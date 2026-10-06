"""
tests/test_panic.py — 패닉 모드 (시장 급락일 다음 날 과매도주 교체) 점검. API 키 없이 돈다.

    python tests/test_panic.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.DATA_DIR = tempfile.mkdtemp(prefix="panic-test-")
config.STATE_BACKEND = "local"
config.DRY_RUN = True
config.TELEGRAM_BOT_TOKEN = ""
config.TELEGRAM_ALLOWED_CHAT_IDS = []
config.STRATEGY = "near_high"
config.NH_SLOTS = 2
config.NH_SURGE_EXIT_PCT = 10.0
config.ENTRY_LIMIT_TICKS = 2
config.PANIC_ENABLED = True
config.PANIC_MKT_DROP_PCT = 4.0
config.PANIC_SLOTS = 5
config.PANIC_HOLD_DAYS = 5
config.PANIC_UNIVERSE_TOP = 1500
config.PANIC_PICK_TOP = 1000
config.PANIC_MKT_MIN_VALUE = 3e9
config.PANIC_PICK_MIN_VALUE = 2e9

from core import panic as P, state                # noqa: E402
from core import universe as universe_mod        # noqa: E402
from core.kis import quotes, trading              # noqa: E402
from jobs import panic, rotation                 # noqa: E402

PASS, FAIL = [], []
TODAY = date.today().strftime("%Y%m%d")


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'  ok  ' if cond else ' FAIL '} {name}" + (f"  — {detail}" if detail else ""))


def bars(ret_today: float, ret5: float, value: float = 5e9, today_value: float | None = None,
         ibs: float = 0.3, price: float = 10_000) -> list[dict]:
    """30일 일봉. 마지막 날(오늘) 등락률 ret_today, 5일 수익률 ret5, 오늘 종가 위치 ibs."""
    out = []
    base = price / (1 + ret5)
    for i in range(29):
        out.append({"date": f"{20240101 + i:08d}", "open": base, "high": base, "low": base, "close": base,
                    "volume": 1, "value": value})
    prev = price / (1 + ret_today)
    out[-1]["close"] = prev
    rng = price * 0.1                                 # 오늘 고가 − 저가
    out.append({"date": TODAY, "open": prev, "high": price + (1 - ibs) * rng, "low": price - ibs * rng,
                "close": price, "volume": 1, "value": value if today_value is None else today_value})
    return out


MARKET: dict[str, tuple] = {}
PRICES: dict[str, float] = {}
ORDERS: list[dict] = []


def build_market(drop: float) -> None:
    """시총 순서대로 200종목. 전부 오늘 drop 만큼 빠지고, 몇 종목은 특별한 모양."""
    MARKET.clear()
    for k in range(200):
        t = f"{100000 + k:06d}"
        MARKET[t] = (f"종목{k}", bars(drop, -0.02))
    MARKET["100010"] = ("최대낙폭A", bars(drop, -0.35))
    MARKET["100011"] = ("낙폭B", bars(drop, -0.30))
    MARKET["100012"] = ("낙폭C", bars(drop, -0.25))
    MARKET["100013"] = ("고가권마감X", bars(drop, -0.40, ibs=0.9))              # 종가 위치 높음 → 제외
    MARKET["100014"] = ("거래대금폭증Y", bars(drop, -0.38, today_value=2e10))    # 20일 평균의 4배 → 제외
    MARKET["100015"] = ("거래적음Z", bars(drop, -0.45, value=1e9))              # 거래대금 부족 → 제외
    MARKET["100016"] = ("낙폭D", bars(drop, -0.20))
    MARKET["100017"] = ("낙폭E", bars(drop, -0.18))
    MARKET["100018"] = ("낙폭F", bars(drop, -0.16))


def fake_universe(top_n, force=False):
    return [{"ticker": t, "name": v[0], "market": "KOSDAQ", "marcap": 1e12 - i}
            for i, (t, v) in enumerate(MARKET.items())][:top_n]


def fake_history(ticker, bars=260):
    return MARKET[ticker][1][-bars:]


def fake_price(ticker, market="J"):
    p = PRICES.get(ticker, MARKET[ticker][1][-1]["close"] if ticker in MARKET else 10_000)
    return {"ticker": ticker, "price": p, "prev_close": p, "is_halted": False, "is_watch": False, "open": p,
            "high": p, "low": p, "change_pct": 0, "volume": 1, "value": 1, "name": "", "upper_limit": 0,
            "lower_limit": 0, "status_code": "00", "market": market}


def fake_order(ticker, qty, price, side, market=False):
    ORDERS.append({"ticker": ticker, "qty": qty, "price": price, "side": side})
    return {"success": True, "order_no": f"T{len(ORDERS):04d}", "org_no": "0", "message": "TEST",
            "ticker": ticker, "qty": qty, "price": price, "side": side, "dry_run": True}


def fake_balance():
    hold = [{"ticker": p["ticker"], "name": p.get("name", ""), "qty": int(p["qty"]), "sellable_qty": int(p["qty"]),
             "avg_price": float(p["entry_price"]), "price": 0, "eval_amount": 0, "pnl_amount": 0, "pnl_pct": 0}
            for p in state.get_positions()]
    return {"holdings": hold, "cash": 1e8, "d2_cash": 1e8, "total_eval": 1e8, "net_asset": 1e8, "pnl_amount": 0}


def install():
    universe_mod.get_large_universe = fake_universe
    quotes.get_daily_history = fake_history
    quotes.get_price = fake_price
    quotes.is_open_day = lambda d=None: True
    trading.buy = lambda t, q, p, market=False: fake_order(t, q, p, "buy", market)
    trading.sell = lambda t, q, p, market=False: fake_order(t, q, p, "sell", market)
    trading.place_order = fake_order
    trading.get_buyable = lambda t, p: {"cash": 1e8, "qty_no_margin": int(1e8 // max(p, 1)), "amount_no_margin": 1e8,
                                        "qty_max": 0, "amount_max": 0}
    trading.get_balance = fake_balance
    trading.get_pending_orders = lambda: []
    trading.cancel_order = lambda *a, **k: {"success": True}


def new_day():
    """다음 거래일인 척: 카운트 표시를 지운다."""
    d = state.load()
    pn = d.get("panic") or {}
    if pn.get("counted"):
        pn["counted"] = "2000-01-01"
    if pn.get("entry_date") == panic._today():
        pn["entry_date"] = "2000-01-01"
    d["panic"] = pn
    rot = d.get("rotation") or {}
    rot["counted"] = "2000-01-01"
    d["rotation"] = rot
    d["traded_today"] = {}
    state.save(d)


def tickers(kind: str | None = None) -> list[str]:
    return sorted(p["ticker"] for p in state.get_positions() if kind is None or p.get("strategy") == kind)


def main() -> int:
    install()

    print("\n── 계산 ─────────────────────────────────────────")
    build_market(-0.05)
    snaps = [P.snapshot(t, v[0], "KOSDAQ", i + 1, v[1], TODAY) for i, (t, v) in enumerate(MARKET.items())]
    avg, n = P.market_drop([s for s in snaps if s])
    check("시장 평균 = 거래대금 30억↑ 종목의 오늘 등락률 평균", abs(avg + 0.05) < 1e-9 and n == 199, f"{avg:+.4f}, {n}종목")
    check("−5% 는 급락일", P.is_panic(avg, n))
    check("−3.9% 는 급락일 아님", not P.is_panic(-0.039, 500))
    check("종목 100개 미만이면 판정 안 함", not P.is_panic(-0.10, 50))
    top = [s.ticker for s in P.pick([s for s in snaps if s], 5)]
    check("후보 = 5일 수익률 나쁜 순, 고가권 마감 · 거래대금 폭증 · 거래대금 부족 제외",
          top == ["100010", "100011", "100012", "100016", "100017"], str(top))
    stale = P.snapshot("x", "x", "", 1, MARKET["100010"][1][:-1], TODAY)
    check("오늘 일봉이 없으면 계산 안 함 (장중에 돌린 경우)", stale is None)

    print("\n── 평소 보유 → 급락 판정 → 다음 날 교체 ─────────────")
    for t in ("900001", "900002"):
        MARKET[t] = ("평소보유", bars(-0.05, 0.1, price=50_000))
        state.add_position({"ticker": t, "name": "평소보유", "qty": 3, "entry_price": 50_000, "entry_date": "2000-01-01",
                            "strategy": "near_high", "filled": True})
    rotation.set_rot(cycle_start="2000-01-01", day_idx=3, counted="2000-01-01")
    config.PANIC_ENABLED = False
    check("PANIC_ENABLED=false 면 판정 안 함", panic.scan().get("skipped") == "PANIC_ENABLED=false")
    config.PANIC_ENABLED = True
    r = panic.scan()
    check("급락일 판정 → 대기", r.get("panic") is True and panic.get()["status"] == "pending", str(r.get("avg")))
    check("대기 중엔 평소 전략이 사고팔지 않음", panic.blocks_base())
    new_day()
    ORDERS.clear()
    rotation.prep(force=True)
    e = rotation.entry(force=True)
    sells = [o["ticker"] for o in ORDERS if o["side"] == "sell"]
    check("평소 보유 2종목 매도", sorted(sells) == ["900001", "900002"], str(sells))
    check("패닉 5종목 매수", tickers("panic") == ["100010", "100011", "100012", "100016", "100017"], str(tickers()))
    check("상태 = 보유 1일째", panic.get()["status"] == "active" and panic.get()["day_idx"] == 1, str(panic.get()))
    pos = state.get_position("100010")
    check("종목당 금액 = 순자산 / 5", pos["qty"] * pos["entry_price"] <= 2e7 and pos["qty"] * pos["entry_price"] > 1.9e7,
          f"{pos['qty']}주 × {pos['entry_price']}")
    check("같은 날 다시 돌려도 추가 매수 없음", len([o for o in ORDERS if o["side"] == "buy"]) == 5
          and not rotation.entry(force=True)["panic"]["refill"]["entered"])
    check("보유 중엔 새 급락 신호 안 받음", panic.scan().get("skipped", "").startswith("패닉 보유 중"))

    print("\n── 보유 · 청산 ───────────────────────────────────")
    rotation._surge_exits()
    check("급등 매도 규칙은 패닉 종목에 안 걸림", len(tickers("panic")) == 5)
    for d in range(2, 6):
        new_day()
        rotation.prep(force=True)
        rotation.close(force=True, send_report=False)
        if d < 5:
            check(f"{d}일째: 보유 유지", len(tickers("panic")) == 5 and panic.get()["day_idx"] == d)
    check("5일째 마감에 전부 매도", tickers("panic") == [] and panic.get()["status"] == "idle", str(tickers()))
    check("다음 날 평소 전략 재정렬 예약", rotation.is_rebalance_due())
    check("패닉 끝나면 평소 전략이 다시 움직임", not panic.blocks_base())

    print("\n── 급락 아닌 날 ─────────────────────────────────")
    build_market(-0.02)
    r = panic.scan()
    check("−2% 는 대기 안 만듦", r.get("panic") is False and panic.get()["status"] == "idle", f"{r.get('avg'):+.4f}")
    build_market(-0.05)
    panic.scan()
    build_market(-0.01)
    new_day()
    r = panic.scan()
    check("대기 신호가 있어도 다음 판정이 급락 아니면 버림 (진입이 안 돈 경우)", panic.get()["status"] == "idle")

    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
