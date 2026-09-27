"""
tests/test_closebet.py — 종가 베팅 (STRATEGY=closebet) 흐름 점검. API 키 없이 돈다.

    python tests/test_closebet.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.DATA_DIR = tempfile.mkdtemp(prefix="cb-test-")
config.STATE_BACKEND = "local"
config.DRY_RUN = True
config.TELEGRAM_BOT_TOKEN = ""
config.TELEGRAM_ALLOWED_CHAT_IDS = []
config.STRATEGY = "closebet"
config.CB_SLOTS = 2
config.CB_RANK_TOP = 30
config.CB_MIN_CHANGE_PCT = 5.0
config.CB_MAX_CHANGE_PCT = 29.0
config.CB_MIN_IBS = 0.9
config.CB_MIN_VALUE = 5e9
config.CB_REGIME = True
config.CB_BUY_TICKS = 5

from core import state                            # noqa: E402
from core import universe as universe_mod         # noqa: E402
from core.kis import quotes, trading              # noqa: E402
from core.kis.tick import offset_ticks            # noqa: E402
from jobs import closebet                         # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'  ok  ' if cond else ' FAIL '} {name}" + (f"  — {detail}" if detail else ""))


# 코드: (이름, 현재가, 저가, 고가, 등락률, 거래대금, 상한가)
QUOTES = {
    "100010": ("강세마감A", 10_900, 9_800, 11_000, 12.0, 3e11, 12_600),   # IBS 0.92 → 후보, 거래대금 1위
    "100020": ("강세마감B", 5_200, 4_900, 5_220, 6.1, 1e11, 6_300),       # IBS 0.94 → 후보
    "100030": ("강세마감C", 20_000, 18_000, 20_100, 8.0, 8e10, 25_000),   # IBS 0.95 → 후보지만 슬롯 부족
    "100040": ("윗꼬리D", 10_000, 9_000, 11_500, 9.0, 2e11, 12_000),     # IBS 0.4 → 탈락
    "100050": ("상한가E", 13_000, 10_000, 13_000, 29.9, 4e11, 13_000),   # 상한가 → 탈락
    "100060": ("약보합F", 10_000, 9_900, 10_050, 1.0, 5e11, 13_000),     # 등락률 → 탈락
    "100075": ("우선주G", 10_000, 9_000, 10_000, 10.0, 1e11, 13_000),    # 우선주 → 탈락
}
INDEX = {"closes": [800 + i for i in range(110)]}      # 상승 추세 → 100일선 위
ORDERS = []


def fake_rank(market="0000"):
    return [{"ticker": t, "name": v[0], "price": v[1], "change_pct": v[4], "value": v[5]}
            for t, v in sorted(QUOTES.items(), key=lambda kv: -kv[1][5])]


def fake_price(t, market="J"):
    name, p, lo, hi, chg, val, up = QUOTES[t]
    return {"ticker": t, "name": name, "price": p, "prev_close": p / (1 + chg / 100), "open": lo, "high": hi, "low": lo,
            "change_pct": chg, "volume": 1, "value": val, "upper_limit": up, "lower_limit": round(p * 0.7),
            "status_code": "00", "is_halted": False, "is_watch": False, "market": market}


def fake_order(t, q, p, side, market=False):
    ORDERS.append({"ticker": t, "qty": q, "price": p, "side": side})
    return {"success": True, "order_no": f"C{len(ORDERS)}", "org_no": "0", "message": "T", "ticker": t,
            "qty": q, "price": p, "side": side, "dry_run": config.DRY_RUN}


def fake_balance():
    hold = [{"ticker": p["ticker"], "name": p["name"], "qty": p["qty"], "sellable_qty": p["qty"],
             "avg_price": p["entry_price"], "price": 0, "eval_amount": 0, "pnl_amount": 0, "pnl_pct": 0}
            for p in state.get_positions()]
    return {"holdings": hold, "cash": 600_000, "d2_cash": 600_000, "total_eval": 600_000, "net_asset": 600_000,
            "pnl_amount": 0}


def install():
    quotes.get_value_rank = fake_rank
    quotes.get_price = fake_price
    quotes.get_index_daily = lambda code="1001", bars=120: [{"date": f"{i:08d}", "close": c}
                                                            for i, c in enumerate(INDEX["closes"])]
    quotes.is_open_day = lambda d=None: True
    universe_mod.get_large_universe = lambda n, force=False: [{"ticker": t} for t in QUOTES if t.endswith("0")]
    closebet.universe = universe_mod
    trading.buy = lambda t, q, p, market=False: fake_order(t, q, p, "buy")
    trading.sell = lambda t, q, p, market=False: fake_order(t, q, p, "sell")
    trading.get_buyable = lambda t, p: {"cash": 600_000, "qty_no_margin": 600_000 // p, "amount_no_margin": 0,
                                        "qty_max": 0, "amount_max": 0}
    trading.get_balance = fake_balance
    trading.get_pending_orders = lambda: []
    trading.cancel_order = lambda *a, **k: {"success": True}


def main() -> int:
    install()
    print("\n── 후보 판정 ─────────────────────────────────────")
    c, stats = closebet.candidates()
    codes = [x["ticker"] for x in c]
    check("후보 = IBS·등락률 통과 종목, 거래대금 큰 순", codes == ["100010", "100020", "100030"], str(codes))
    rj = stats["rejects"]
    check("윗꼬리(IBS 낮음) 탈락", rj.get("IBS") == 1, str(rj))
    check("상한가 탈락 (못 산다)", rj.get("상한가", 0) + rj.get("등락률", 0) >= 2, str(rj))
    check("우선주 탈락", rj.get("보통주아님") == 1, str(rj))

    print("\n── 국면 ─────────────────────────────────────────")
    check("코스닥이 100일선 위면 on", closebet.regime_on()[0])
    INDEX["closes"] = [900 - i for i in range(110)]
    on, why = closebet.regime_on()
    check("100일선 아래면 off", not on, why)
    r = closebet.buy_close(force=True)
    check("국면 off 면 매수 안 함", r.get("skipped") == "국면" and not ORDERS)
    INDEX["closes"] = [800 + i for i in range(110)]
    quotes.get_index_daily = lambda code="1001", bars=120: (_ for _ in ()).throw(RuntimeError("API 오류"))
    check("지수 조회 실패면 off (안전 쪽)", not closebet.regime_on()[0])
    quotes.get_index_daily = lambda code="1001", bars=120: [{"date": f"{i:08d}", "close": c}
                                                            for i, c in enumerate(INDEX["closes"])]

    print("\n── 15:21 종가 매수 ──────────────────────────────")
    r = closebet.buy_close(force=True)
    held = sorted(p["ticker"] for p in state.get_positions())
    check("2슬롯: 거래대금 1·2위 후보 매수", held == ["100010", "100020"], str(held))
    a = state.get_position("100010")
    lim = offset_ticks(10_900, config.CB_BUY_TICKS)
    check("지정가 = 현재가 + 5틱", a["entry_price"] == lim, f"{a['entry_price']} vs {lim}")
    check("수량 = 슬롯 30만원 / 지정가", a["qty"] == 300_000 // lim, str(a["qty"]))
    check("다시 돌려도 추가 매수 없음 (보유한도)", closebet.buy_close(force=True).get("skipped") == "보유한도")

    print("\n── 다음 날 08:45 시가 매도 ───────────────────────")
    ORDERS.clear()
    r = closebet.sell_open(force=True)
    check("오늘 산 종목은 오늘 안 판다", not r["sold"] and not ORDERS)
    d = state.load()
    for p in d["positions"]:
        p["entry_date"] = "2000-01-01"
    d["traded_today"] = {}
    state.save(d)
    r = closebet.sell_open(force=True)
    sells = [o for o in ORDERS if o["side"] == "sell"]
    check("전일 매수분 전량 시가 매도 주문", len(sells) == 2 and not state.get_positions(), str(sells))
    check("매도 지정가 = 하한가 (동시호가에서 시가 체결)", sells and sells[0]["price"] == round(10_900 * 0.7), str(sells[:1]))
    check("이력에 기록", sum("종가 베팅" in h["exit_reason"] for h in state.get_history()) == 2)
    check("09:05 점검 (DRY_RUN 은 할 일 없음)", closebet.check_open(force=True) == {"left": []})
    check("리포트 실행", closebet.report(force=True) == {"ok": True})

    print("\n── 설정 ─────────────────────────────────────────")
    check("요약에 종가 베팅", "종가 베팅" in config.summary())
    check("STRATEGY=closebet 검증 통과", not [p for p in config.validate() if "STRATEGY" in p])

    print("\n" + "═" * 60)
    print(f"통과 {len(PASS)} / 실패 {len(FAIL)}")
    for f in FAIL:
        print("  실패:", f)
    shutil.rmtree(config.DATA_DIR, ignore_errors=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
