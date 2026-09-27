"""
tests/test_rotation.py — 52주 신고가 근접 로테이션 (STRATEGY=near_high) 통합 점검

한투 API 대신 가짜 시세·주문을 끼워 순위 → 첫 매수 → 보유 → 21거래일 교체 → 빈 슬롯 복구 → 마감 흐름을 본다.
API 키 없이 돈다.

    python tests/test_rotation.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.DATA_DIR = tempfile.mkdtemp(prefix="nh-test-")
config.STATE_BACKEND = "local"
config.DRY_RUN = True
config.TELEGRAM_BOT_TOKEN = ""
config.TELEGRAM_ALLOWED_CHAT_IDS = []
config.STRATEGY = "near_high"
config.NH_UNIVERSE_TOP = 200
config.NH_SLOTS = 2
config.NH_HOLD_DAYS = 21
config.NH_HIGH_LOOKBACK = 250
config.NH_MOM_DAYS = 60
config.NH_MIN_PRICE = 1000
config.NH_MIN_VALUE = 1_000_000_000
config.NH_REFILL = True
config.NH_STOP_LOSS_PCT = 0.0
config.NH_MAX_DAILY_GAIN_PCT = 10.0
config.ENTRY_LIMIT_TICKS = 2

from core import nearhigh, state                 # noqa: E402
from core import universe as universe_mod        # noqa: E402
from core.kis import quotes, trading              # noqa: E402
from core.kis.tick import offset_ticks           # noqa: E402
from jobs import rotation                        # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'  ok  ' if cond else ' FAIL '} {name}" + (f"  — {detail}" if detail else ""))


# ══════════════════════════════════════════════════════════════
# 가짜 시장
# ══════════════════════════════════════════════════════════════
N = 262


def series(last: float, peak_ratio: float, mom: float = 0.2, value: float = 5e10, n: int = N) -> list[dict]:
    """마지막 종가 last, 250일 최고가 = last / peak_ratio, 60일 전 종가 = last / (1+mom)."""
    peak = last / peak_ratio
    start = last / (1 + mom)
    out = []
    for i in range(n):
        c = start + (last - start) * max(0, i - (n - 61)) / 60 if i >= n - 61 else start
        out.append({"date": f"2025{(i // 28) % 12 + 1:02d}{i % 28 + 1:02d}{i:03d}"[:8] + f"{i:04d}",
                    "open": c, "high": c, "low": c, "close": c, "volume": 1, "value": value})
    out[n // 2]["high"] = peak          # 중간에 최고가 한 번
    out[-1]["close"] = last
    out[-1]["high"] = max(out[-1]["high"], last)
    for k, x in enumerate(out):
        x["date"] = f"{20240101 + k:08d}"   # 정렬만 되면 된다
    return out


MARKET = {
    # 코드: (이름, 시총(억), 일봉)
    "000010": ("신고가A", 900_000, series(50_000, 1.00)),
    "000020": ("근접B", 800_000, series(30_000, 0.95)),
    "000030": ("비싼C", 700_000, series(700_000, 0.99)),     # 30만원 슬롯으로 1주도 못 산다
    "000040": ("하락D", 600_000, series(20_000, 0.98, mom=-0.1)),
    "000050": ("거래적음E", 500_000, series(20_000, 0.99, value=1e8)),
    "000060": ("신규상장F", 400_000, series(20_000, 1.00, n=100)),
    "000070": ("후보G", 300_000, series(10_000, 0.90)),
}
PRICES: dict[str, float] = {}
ORDERS: list[dict] = []


def fake_universe(top_n, force=False):
    return [{"ticker": t, "name": v[0], "market": "KOSPI", "marcap": v[1]} for t, v in MARKET.items()][:top_n]


def fake_history(ticker, bars=260):
    return MARKET[ticker][2][-bars:]


def fake_price(ticker, market="J"):
    p = PRICES.get(ticker, MARKET[ticker][2][-1]["close"])
    return {"ticker": ticker, "price": p, "prev_close": p, "is_halted": False, "is_watch": False,
            "open": p, "high": p, "low": p, "change_pct": 0, "volume": 1, "value": 1, "name": "",
            "upper_limit": 0, "lower_limit": 0, "status_code": "00", "market": market}


def fake_order(ticker, qty, price, side, market=False):
    ORDERS.append({"ticker": ticker, "qty": qty, "price": price, "side": side})
    return {"success": True, "order_no": f"T{len(ORDERS):04d}", "org_no": "0", "message": "TEST",
            "ticker": ticker, "qty": qty, "price": price, "side": side, "dry_run": config.DRY_RUN}


CASH = {"v": 600_000}


def fake_buyable(ticker, price):
    return {"cash": CASH["v"], "qty_no_margin": int(CASH["v"] // max(price, 1)), "amount_no_margin": CASH["v"],
            "qty_max": 0, "amount_max": 0}


def fake_balance():
    hold = [{"ticker": p["ticker"], "name": p.get("name", ""), "qty": int(p["qty"]), "sellable_qty": int(p["qty"]),
             "avg_price": float(p["entry_price"]), "price": 0, "eval_amount": 0, "pnl_amount": 0, "pnl_pct": 0}
            for p in state.get_positions()]
    return {"holdings": hold, "cash": CASH["v"], "d2_cash": CASH["v"], "total_eval": 600_000,
            "net_asset": 600_000, "pnl_amount": 0}


def install():
    universe_mod.get_large_universe = fake_universe
    rotation.universe = universe_mod
    quotes.get_daily_history = fake_history
    quotes.get_price = fake_price
    quotes.is_open_day = lambda d=None: True
    trading.place_order = fake_order
    trading.buy = lambda t, q, p, market=False: fake_order(t, q, p, "buy", market)
    trading.sell = lambda t, q, p, market=False: fake_order(t, q, p, "sell", market)
    trading.get_buyable = fake_buyable
    trading.get_balance = fake_balance
    trading.get_pending_orders = lambda: []
    trading.cancel_order = lambda *a, **k: {"success": True}


def next_day(days: int = 1):
    """하루(또는 며칠) 지난 것처럼: 카운트 표시를 지우고 당일 매매·순위 캐시를 비운다."""
    for _ in range(days):
        d = state.load()
        rot = d.get("rotation") or {}
        rot["counted"] = "2000-01-01"
        rot["ranking"] = {}
        if rot.get("rebalanced"):
            rot["rebalanced"] = "2000-01-01"
        if rot.get("cycle_start") == rotation._today():
            rot["cycle_start"] = "2000-01-01"
        d["rotation"] = rot
        d["traded_today"] = {}
        state.save(d)
        rotation.bump_day()


def held() -> list[str]:
    return sorted(p["ticker"] for p in state.get_positions())


# ══════════════════════════════════════════════════════════════
def main() -> int:
    install()

    print("\n── 점수·필터 ─────────────────────────────────────")
    picks = [nearhigh.evaluate(t, v[0], v[2], marcap=v[1]) for t, v in MARKET.items()]
    by = {p.ticker: p for p in picks}
    ranked = nearhigh.rank(picks)
    order = [p.ticker for p in ranked]
    check("점수 = 전일 종가 / 250일 최고가", abs(by["000020"].score - 0.95) < 1e-6, f"{by['000020'].score:.4f}")
    check("순위: 신고가A > 비싼C > 근접B > 후보G", order == ["000010", "000030", "000020", "000070"], str(order))
    check("60일 수익률 ≤ 0 탈락", not by["000040"].eligible and "수익률" in by["000040"].reject, by["000040"].reject)
    check("거래대금 부족 탈락", not by["000050"].eligible and "거래대금" in by["000050"].reject, by["000050"].reject)
    check("일봉 250개 미만 탈락", not by["000060"].eligible and "일봉 부족" in by["000060"].reject, by["000060"].reject)
    jump = series(40_000, 0.99)
    jump[-5]["close"] = jump[-6]["close"] * 1.12          # 5일 전에 하루 +12%
    pj = nearhigh.evaluate("000099", "급등이력", jump, marcap=1)
    check("20일 내 +10%↑ 급등일 있으면 탈락 (테마주 회피)", not pj.eligible and "급등이력" in pj.reject, pj.reject)
    config.NH_MAX_DAILY_GAIN_PCT = 0
    check("NH_MAX_DAILY_GAIN_PCT=0 이면 필터 끔", nearhigh.evaluate("000099", "급등이력", jump, marcap=1).eligible)
    config.NH_MAX_DAILY_GAIN_PCT = 10.0

    print("\n── 첫날: 준비 → 첫 매수 ─────────────────────────")
    r = rotation.prep(force=True)
    check("첫날은 교체일(첫 매수)", r["due"] is True)
    check("순위 캐시", rotation.load_ranking(build_if_missing=False)[0][0].ticker == "000010")
    ORDERS.clear()
    e = rotation.entry(force=True)
    check("2종목 매수: 신고가A + 근접B (비싼C 건너뜀)", held() == ["000010", "000020"], f"{held()} / {e.get('skipped')}")
    check("비싼C 가 순위 2위여도 유지 목록은 A·B (살 수 있는 상위 2)", e.get("kept") == [] and len(e.get("entered", [])) == 2)
    check("비싼 종목은 순위에서 빠져 아예 시도하지 않음", not any(o["ticker"] == "000030" for o in ORDERS), str(ORDERS))
    a = state.get_position("000010")
    lim = offset_ticks(50_000, config.ENTRY_LIMIT_TICKS)
    check("수량 = 슬롯 30만원 / 지정가", a["qty"] == 300_000 // lim, f"{a['qty']}주 @{lim}")
    rot = rotation.get_rot()
    check("교체 주기 시작 기록", rot.get("cycle_start") == rotation._today() and rot.get("day_idx") == 0, str(rot))
    check("같은 날 다시 돌려도 추가 매수 없음", rotation.entry(force=True).get("skipped") == "교체일 아님" and len(ORDERS) == 2)

    print("\n── 보유 기간: 교체일 전엔 아무것도 안 판다 ───────")
    next_day(1)
    PRICES["000010"] = 50_000 * 0.8            # −20% 여도
    ORDERS.clear()
    rotation.monitor_now = None
    rotation._now_hhmm = lambda: "1100"
    m = rotation.monitor(force=True)
    check("손절 없음(기본): −20% 에도 안 판다", not m["closed"] and held() == ["000010", "000020"], str(m))
    config.NH_STOP_LOSS_PCT = 10.0
    m = rotation.monitor(force=True)
    check("NH_STOP_LOSS_PCT=10 이면 손절", len(m["closed"]) == 1 and m["closed"][0]["ticker"] == "000010",
          str([c["exit_reason"] for c in m["closed"]]))
    config.NH_STOP_LOSS_PCT = 0.0
    PRICES.pop("000010")
    check("같은 날 손절한 종목은 되사지 않는다 (빈 슬롯은 다른 종목)", "000010" not in held(), str(held()))
    check("빈 슬롯은 같은 회차에 다음 순위로 채움", len(held()) == 2 and "000070" in held(), str(held()))

    print("\n── 21거래일 교체 ─────────────────────────────────")
    # 처음 상태로: A, B 보유
    for p in state.get_positions():
        state.close_position(p["ticker"], p["entry_price"], "테스트 정리")
    d = state.load()
    d["rotation"] = {}
    d["traded_today"] = {}
    state.save(d)
    rotation.prep(force=True)
    rotation.entry(force=True)
    check("다시 A, B 보유", held() == ["000010", "000020"], str(held()))
    next_day(20)
    check("20거래일째는 교체일 아님", not rotation.is_rebalance_due(), str(rotation.get_rot().get("day_idx")))
    check("entry 는 아무것도 안 함", rotation.entry(force=True).get("skipped") == "교체일 아님")
    # 교체일 전에 B 가 밀려나고 G 가 신고가로 올라온다
    MARKET["000020"] = ("근접B", 800_000, series(30_000, 0.80))
    MARKET["000070"] = ("후보G", 300_000, series(10_000, 1.00))
    next_day(1)
    check("21거래일째는 교체일", rotation.is_rebalance_due(), str(rotation.get_rot().get("day_idx")))
    ORDERS.clear()
    rotation.prep(force=True)
    e = rotation.entry(force=True)
    sells = [o["ticker"] for o in ORDERS if o["side"] == "sell"]
    buys = [o["ticker"] for o in ORDERS if o["side"] == "buy"]
    check("순위 밖 B 만 매도", sells == ["000020"], str(sells))
    check("A 는 유지(매매 없음), G 신규 매수", buys == ["000070"] and held() == ["000010", "000070"], f"{buys} / {held()}")
    check("매도가 매수보다 먼저", ORDERS and ORDERS[0]["side"] == "sell")
    check("교체 주기 다시 0", rotation.get_rot().get("day_idx") == 0)
    hist = state.get_history()
    check("이력에 교체 매도 기록", any("교체" in h.get("exit_reason", "") for h in hist))

    print("\n── 교체일 누락 복구 / 컷오프 / 일시정지 ──────────")
    next_day(21)
    rotation._now_hhmm = lambda: "1000"
    ORDERS.clear()
    m = rotation.monitor(force=True)
    check("09:05 가 안 돌았어도 감시가 교체 실행", "rebalance" in m and rotation.get_rot().get("day_idx") == 0, str(m.keys()))
    state.close_position("000010", 50_000, "테스트: 수동 매도")
    rotation._now_hhmm = lambda: "1445"
    m = rotation.monitor(force=True)
    check("컷오프(14:30) 이후엔 빈 슬롯 매수 안 함", len(held()) == 1 and "refill" not in m, str(held()))
    rotation._now_hhmm = lambda: "1300"
    state.set_paused(True)
    rotation.monitor(force=True)
    check("일시정지면 매수 안 함", len(held()) == 1)
    state.set_paused(False)
    next_day(1)
    rotation.monitor(force=True)
    check("다음 날 빈 슬롯 복구 (NH_REFILL)", len(held()) == 2, str(held()))

    print("\n── 마감 / 설정 요약 / 실주문 모드 매도 대기 ───────")
    c = rotation.close(force=True, send_report=False)
    check("마감 실행", "cancelled" in c)
    check("요약에 전략명", "52주 신고가" in config.summary())
    check("설정 검증 통과", not [p for p in config.validate() if "STRATEGY" in p or "NH_" in p])
    config.DRY_RUN = False
    check("실주문 모드: 판 종목이 잔고에서 빠지면 대기 끝", rotation._wait_sold({"999999"}, timeout=1))
    config.DRY_RUN = True

    print("\n" + "═" * 60)
    print(f"통과 {len(PASS)} / 실패 {len(FAIL)}")
    for f in FAIL:
        print("  실패:", f)
    shutil.rmtree(config.DATA_DIR, ignore_errors=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
