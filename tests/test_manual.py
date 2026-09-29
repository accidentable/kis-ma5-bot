"""
tests/test_manual.py — 수동 매매 (/buy /sell /bal /orders /cancel /fills /progress) 점검. API 키 없이 돈다.

    python tests/test_manual.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.DATA_DIR = tempfile.mkdtemp(prefix="manual-test-")
config.STATE_BACKEND = "local"
config.DRY_RUN = True
config.TELEGRAM_BOT_TOKEN = ""
config.TELEGRAM_ALLOWED_CHAT_IDS = [1]

from core import commands, manual, state        # noqa: E402
from core.kis import quotes, trading            # noqa: E402

PASS, FAIL = [], []
ORDERS: list[dict] = []
CANCELS: list[str] = []
FILLS: dict[date, list[dict]] = {}


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'  ok  ' if cond else ' FAIL '} {name}" + (f"  — {detail}" if detail else ""))


def fake_price(ticker, market="J"):
    p = 70_000
    return {"ticker": ticker, "price": p, "prev_close": p, "is_halted": ticker == "999999", "is_watch": False,
            "open": p, "high": p, "low": p, "change_pct": 1.23, "volume": 1, "value": 1, "name": "", "upper_limit": 0,
            "lower_limit": 0, "status_code": "00", "market": market}


def fake_order(ticker, qty, price, side, market=False):
    ORDERS.append({"ticker": ticker, "qty": qty, "price": price, "side": side, "market": market})
    return {"success": True, "order_no": f"T{len(ORDERS):04d}", "org_no": "0", "message": "TEST",
            "ticker": ticker, "qty": qty, "price": price, "side": side, "dry_run": True}


def fake_balance():
    return {"holdings": [{"ticker": "005930", "name": "삼성전자", "qty": 100, "sellable_qty": 80, "avg_price": 65_000,
                          "price": 70_000, "eval_amount": 7e6, "pnl_amount": 5e5, "pnl_pct": 7.69}],
            "cash": 3e7, "d2_cash": 3e7, "total_eval": 3.7e7, "net_asset": 1e8, "pnl_amount": 5e5}


PENDING = [
    {"order_no": "0000012345", "org_no": "00950", "ticker": "005930", "name": "삼성전자", "side": "buy", "qty": 10,
     "remain_qty": 7, "price": 69_000, "order_time": "093015", "order_dvsn": "00"},
    {"order_no": "0000012346", "org_no": "00950", "ticker": "000660", "name": "SK하이닉스", "side": "sell", "qty": 3,
     "remain_qty": 3, "price": 200_000, "order_time": "101500", "order_dvsn": "00"},
]


def install():
    quotes.get_price = fake_price
    trading.buy = lambda t, q, p, market=False: fake_order(t, q, p, "buy", market)
    trading.sell = lambda t, q, p, market=False: fake_order(t, q, p, "sell", market)
    trading.get_buyable = lambda t, p: {"cash": 3e7, "qty_no_margin": int(3e7 // max(p, 1)), "amount_no_margin": 3e7,
                                        "qty_max": 0, "amount_max": 0}
    trading.get_balance = fake_balance
    trading.get_pending_orders = lambda: list(PENDING)
    trading.cancel_order = lambda no, org, q, order_dvsn="00": CANCELS.append(no) or {"success": True}
    trading.get_today_fills = lambda d=None: list(FILLS.get(d or date.today(), []))
    manual._name = lambda t: {"005930": "삼성전자"}.get(t, t)


def last() -> dict:
    return ORDERS[-1]


def main() -> int:
    install()

    print("\n── 입력 해석 ─────────────────────────────────────")
    cases = {"1000만": 1e7, "5천만원": 5e7, "1억": 1e8, "1억5천만": 1.5e8, "20,000,000원": 2e7, "500만원": 5e6,
             "2.5억": 2.5e8, "3천5백만": 3.5e7}
    for s, v in cases.items():
        got = manual.parse_qty_or_amount(s)
        check(f"금액 {s}", got == (None, v), str(got))
    check("수량 10", manual.parse_qty_or_amount("10") == (10, None))
    check("수량 10주", manual.parse_qty_or_amount("10주") == (10, None))
    for bad in ("0", "abc", "-5"):
        try:
            manual.parse_qty_or_amount(bad)
            check(f"잘못된 수량 {bad} 거부", False)
        except ValueError:
            check(f"잘못된 수량 {bad} 거부", True)
    check("가격 생략 = 시장가", manual.parse_price(None) is None and manual.parse_price("시장가") is None)
    check("가격 71,500", manual.parse_price("71,500") == 71_500)

    print("\n── 매수 ─────────────────────────────────────────")
    r = manual.buy("005930", "10")
    check("10주 시장가", last() == {"ticker": "005930", "qty": 10, "price": 0, "side": "buy", "market": True}, r)
    check("응답에 DRY_RUN 표시", "DRY_RUN" in r and "삼성전자" in r)
    manual.buy("A005930", "1000만")
    check("1000만원 시장가 → 현재가+1% 로 수량", last()["qty"] == int(1e7 // (70_000 * 1.01)), str(last()))
    manual.buy("005930", "1000만", "71,530")
    check("지정가는 호가 단위로 맞추고 그 가격으로 수량", last()["price"] == 71_500 and last()["qty"] == 139
          and not last()["market"], str(last()))
    n = len(ORDERS)
    r = manual.buy("005930", "1억")
    check("현금 부족이면 주문 안 함", len(ORDERS) == n and "현금 부족" in r, r)
    r = manual.buy("005930", "5만")
    check("금액이 1주보다 작으면 주문 안 함", len(ORDERS) == n and "수량 0" in r, r)
    r = manual.buy("999999", "10")
    check("거래정지 종목 거부", len(ORDERS) == n and "거래정지" in r, r)
    for bad in ("5930", "abcdef"):
        try:
            manual.buy(bad, "10")
            check(f"종목코드 {bad} 거부", False)
        except ValueError:
            check(f"종목코드 {bad} 거부", True)

    print("\n── 매도 ─────────────────────────────────────────")
    manual.sell("005930")
    check("수량 생략 = 매도 가능 수량 전부 시장가", last() == {"ticker": "005930", "qty": 80, "price": 0, "side": "sell",
                                                  "market": True}, str(last()))
    manual.sell("005930", "5", "72000")
    check("5주 지정가", last()["qty"] == 5 and last()["price"] == 72_000 and not last()["market"])
    manual.sell("005930", "all")
    check("all = 전량", last()["qty"] == 80)
    manual.sell("005930", "350만")
    check("금액으로 매도 = 현재가 기준 수량", last()["qty"] == 50, str(last()))
    n = len(ORDERS)
    r = manual.sell("005930", "81")
    check("매도 가능 수량 초과 거부", len(ORDERS) == n and "매도 가능" in r, r)
    r = manual.sell("000660")
    check("보유 없는 종목 거부", len(ORDERS) == n and "보유 없음" in r, r)

    print("\n── 조회 · 취소 ──────────────────────────────────")
    check("잔고", "삼성전자" in manual.balance() and "순자산" in manual.balance())
    o = manual.orders()
    check("미체결 목록", "0000012345" in o and "09:30" in o and "7/10" in o, o)
    manual.cancel("12345")
    check("주문번호 앞 0 없이도 취소", CANCELS == ["0000012345"], str(CANCELS))
    CANCELS.clear()
    manual.cancel("all")
    check("all = 전부 취소", len(CANCELS) == 2)
    check("없는 주문번호", "없다" in manual.cancel("1"))

    print("\n── 체결 · 대회 조건 ─────────────────────────────")
    today = date.today()
    check("체결 없음", "없음" in manual.fills())
    days = [today - timedelta(days=k) for k in range(0, 20) if (today - timedelta(days=k)).weekday() < 5][:6]
    start = min(days)
    config.CONTEST_START = start.isoformat()
    for i, d in enumerate(days):
        t = f"{100000 + i:06d}"
        FILLS[d] = [{"order_no": "1", "ticker": t, "name": f"종목{i}", "side": "buy", "qty": 1000, "avg_price": 50_000,
                     "amount": 5e7, "time": "090101"},
                    {"order_no": "2", "ticker": t, "name": f"종목{i}", "side": "sell", "qty": 1000, "avg_price": 51_000,
                     "amount": 5.1e7, "time": "150101"}]
    f = manual.fills(today)
    check("오늘 체결 합계", "매수 50,000,000원" in f and "매도 51,000,000원" in f, f)
    p = manual.progress()
    check("대회 조건: 매수+매도 합계 · 일수 · 종목수", "606,000,000원" in p and "매매일수 6일" in p and "매매종목 6개" in p, p)
    check("대회 조건 전부 충족 표시", p.count("✅") == 3 and "❌" not in p)
    FILLS.clear()
    FILLS[today] = [{"order_no": "1", "ticker": "005930", "name": "삼성전자", "side": "buy", "qty": 10,
                     "avg_price": 70_000, "amount": 7e5, "time": "090101"}]
    p = manual.progress()
    check("미달이면 ❌", p.count("❌") == 3, p)

    print("\n── 텔레그램 명령 ────────────────────────────────")
    n = len(ORDERS)
    r = commands.handle("/buy 005930 10 70000", 1)
    check("/buy 가 주문을 낸다", len(ORDERS) == n + 1 and last()["side"] == "buy" and last()["price"] == 70_000, r)
    r = commands.handle("/sell 005930 all", 1)
    check("/sell 가 주문을 낸다", last()["side"] == "sell" and last()["qty"] == 80, r)
    check("/buy 인자 부족이면 도움말", "예)" in commands.handle("/buy 005930", 1))
    check("잘못된 입력은 ⚠️ 로 답함", commands.handle("/buy 005930 abc", 1).startswith("⚠️"))
    check("/bal", "순자산" in commands.handle("/bal", 1))
    check("/orders", "미체결" in commands.handle("/orders", 1))
    check("/progress", "대회 조건" in commands.handle("/progress", 1))
    check("/manual 도움말", commands.handle("/manual", 1) == manual.HELP)
    n = len(ORDERS)
    check("허용 안 된 chat_id 는 무시", commands.handle("/buy 005930 10", 2) == "" and len(ORDERS) == n)
    check("수동 매매는 자동매매 상태에 기록 안 함", state.get_positions() == [])

    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
