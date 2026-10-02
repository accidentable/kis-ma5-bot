"""
core/manual.py — 수동 매매 (한투 OpenAPI 로 직접 사고판다). 대회 'OpenAPI 거래' 조건을 사람이 채우기 위한 창구.

텔레그램 (/buy /sell /bal /orders /cancel /fills /progress) 과 CLI (python cli.py buy ...) 가 같은 함수를 쓴다.
자동매매 상태(state.json) 에는 기록하지 않는다 → 자동 전략이 수동 매수 종목을 팔거나 슬롯으로 세지 않는다.

순서는 종목코드 · 가격 · 수량

가격 표기
  71500     지정가 71,500원 (호가 단위로 맞춘다)
  시장가 · m · 0   시장가
수량 · 금액 표기
  10        10주
  1000만    1,000만원어치 (현재가 기준으로 수량 계산)      500만원 · 1억 · 20000000원 도 된다
  all       (매도) 매도 가능 수량 전부 — 생략해도 전부
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import config
from core.kis import quotes, trading
from core.kis.tick import round_to_tick

logger = logging.getLogger(__name__)

HELP = """수동 매매 (한투 OpenAPI 주문)

/buy 종목코드 가격 수량|금액
   예) /buy 005930 71500 10      71,500원에 10주
       /buy 005930 시장가 1000만  시장가로 1,000만원어치
/sell 종목코드 가격 [수량|all]
   예) /sell 005930 72000 5      72,000원에 5주
       /sell 005930 시장가       시장가로 전량
   (가격에 시장가 · m · 0 을 쓰면 시장가)
/bal       잔고 · 예수금
/orders    미체결 주문
/cancel 주문번호|all   미체결 취소
/fills     오늘 체결
/progress  대회 조건 진행 (매매금액 · 매매일수 · 종목수)"""


# ══════════════════════════════════════════════════════════════
# 입력 해석
# ══════════════════════════════════════════════════════════════
def _parse_money(s: str) -> float | None:
    """정확한 한글 금액 해석: 억 · 천만 · 만 · 원 조합."""
    t = s.replace(",", "").replace(" ", "").strip()
    if not t or not re.search(r"(원|만|억)$", t):
        return None
    t = t[:-1] if t.endswith("원") else t
    total, num = 0.0, ""
    units = {"억": 1e8, "만": 1e4}
    buf = 0.0
    for ch in t:
        if ch.isdigit() or ch == ".":
            num += ch
            continue
        if ch == "천":
            buf += float(num or 1) * 1000
        elif ch == "백":
            buf += float(num or 1) * 100
        elif ch in units:
            buf += float(num or 0)
            total += (buf or 1) * units[ch]
            buf = 0.0
        else:
            return None
        num = ""
    total += buf + (float(num) if num else 0.0)
    return total if total > 0 else None


def parse_qty_or_amount(s: str) -> tuple[int | None, float | None]:
    """(수량, 금액) 중 하나. '10' → (10, None), '1000만' → (None, 1e7)."""
    amt = _parse_money(s)
    if amt:
        return None, amt
    t = s.replace(",", "").rstrip("주")
    if t.isdigit() and int(t) > 0:
        return int(t), None
    raise ValueError(f"수량/금액을 못 읽었다: {s} (예: 10 · 1000만 · 5천만원 · 1억)")


def parse_price(s: str | None) -> int | None:
    """None/'시장가'/'m'/'0' → None(시장가), 숫자 → 지정가."""
    if s is None or s.lower() in ("시장가", "m", "mkt", "market", "0"):
        return None
    t = s.replace(",", "").rstrip("원")
    if not t.isdigit():
        raise ValueError(f"가격을 못 읽었다: {s}")
    return int(t)


def _name(ticker: str) -> str:
    try:
        from core import universe
        return universe.get_name(ticker) or ticker
    except Exception:
        return ticker


def _norm_ticker(s: str) -> str:
    t = s.strip().upper().lstrip("A")
    if not (t.isdigit() and len(t) == 6):
        raise ValueError(f"종목코드는 숫자 6자리: {s}")
    return t


# ══════════════════════════════════════════════════════════════
# 주문
# ══════════════════════════════════════════════════════════════
def buy(ticker: str, price: str | None, qty_or_amount: str) -> str:
    ticker = _norm_ticker(ticker)
    limit = parse_price(price)
    qty, amount = parse_qty_or_amount(qty_or_amount)
    q = quotes.get_price(ticker)
    name = _name(ticker)
    if q["is_halted"]:
        return f"⛔ {name}({ticker}) 거래정지"
    ref = limit or q["price"]
    if ref <= 0:
        return f"⚠️ {ticker} 현재가 조회 실패"
    if limit:
        limit = int(round_to_tick(limit, "nearest"))
    if qty is None:
        unit = (limit or q["price"] * 1.01)          # 시장가는 1% 여유를 두고 수량을 잡는다
        qty = int(amount // unit)
    info = trading.get_buyable(ticker, int(limit or q["price"]))
    if qty > info["qty_no_margin"]:
        return (f"⚠️ 현금 부족: {qty:,}주 요청, 미수 없이 살 수 있는 수량 {info['qty_no_margin']:,}주 "
                f"(주문가능 {info['cash']:,.0f}원)")
    if qty <= 0:
        return "⚠️ 수량 0 — 금액이 1주 가격보다 작다"
    r = trading.buy(ticker, qty, limit or 0, market=limit is None)
    kind = "시장가" if limit is None else f"지정가 {limit:,}원"
    if not r["success"]:
        return f"❌ 매수 주문 거부: {name}({ticker}) {qty:,}주 {kind}\n{r['message']}"
    logger.info("수동 매수 %s %d주 %s 주문번호 %s", ticker, qty, kind, r["order_no"])
    return (f"🟢 매수 주문: {name}({ticker}) {qty:,}주 {kind}\n"
            f"약 {qty * ref:,.0f}원 | 현재가 {q['price']:,.0f}원 ({q['change_pct']:+.2f}%) | 주문번호 {r['order_no']}"
            + ("\n(DRY_RUN — 실제 주문 아님)" if r.get("dry_run") else ""))


def sell(ticker: str, price: str | None = None, qty_or_all: str | None = None) -> str:
    ticker = _norm_ticker(ticker)
    limit = parse_price(price)
    hold = next((h for h in trading.get_balance()["holdings"] if h["ticker"] == ticker), None)
    if hold is None:
        return f"⚠️ {ticker} 보유 없음"
    can = int(hold["sellable_qty"] or hold["qty"])
    if qty_or_all in (None, "all", "전량", "전부"):
        qty = can
    else:
        qty, amount = parse_qty_or_amount(qty_or_all)
        if qty is None:
            qty = int(amount // max(hold["price"], 1))
    if qty <= 0 or qty > can:
        return f"⚠️ 매도 가능 수량 {can:,}주 (요청 {qty:,}주)"
    if limit:
        limit = int(round_to_tick(limit, "nearest"))
    r = trading.sell(ticker, qty, limit or 0, market=limit is None)
    kind = "시장가" if limit is None else f"지정가 {limit:,}원"
    if not r["success"]:
        return f"❌ 매도 주문 거부: {hold['name']}({ticker}) {qty:,}주 {kind}\n{r['message']}"
    logger.info("수동 매도 %s %d주 %s 주문번호 %s", ticker, qty, kind, r["order_no"])
    return (f"🔴 매도 주문: {hold['name']}({ticker}) {qty:,}주 {kind}\n"
            f"평단 {hold['avg_price']:,.0f} → 현재 {hold['price']:,.0f}원 ({hold['pnl_pct']:+.2f}%) | 주문번호 {r['order_no']}"
            + ("\n(DRY_RUN — 실제 주문 아님)" if r.get("dry_run") else ""))


def balance() -> str:
    b = trading.get_balance()
    lines = [f"💰 순자산 {b['net_asset']:,.0f}원 | 예수금 {b['cash']:,.0f}원 (D+2 {b['d2_cash']:,.0f}) | "
             f"평가손익 {b['pnl_amount']:+,.0f}원"]
    if not b["holdings"]:
        lines.append("보유 없음")
    for h in b["holdings"]:
        lines.append(f"· {h['name']}({h['ticker']}) {h['qty']:,}주 (매도가능 {h['sellable_qty']:,}) "
                     f"{h['avg_price']:,.0f} → {h['price']:,.0f} {h['pnl_pct']:+.2f}% ({h['pnl_amount']:+,.0f}원)")
    return "\n".join(lines)


def orders() -> str:
    os_ = trading.get_pending_orders()
    if not os_:
        return "미체결 없음"
    return "\n".join(["⏳ 미체결"] + [
        f"· {o['order_no']} {'매수' if o['side'] == 'buy' else '매도'} {o['name']}({o['ticker']}) "
        f"{o['remain_qty']:,}/{o['qty']:,}주 @{o['price']:,} ({o['order_time'][:2]}:{o['order_time'][2:4]})"
        for o in os_])


def cancel(target: str) -> str:
    os_ = trading.get_pending_orders()
    pick = os_ if target in ("all", "전부") else [o for o in os_ if o["order_no"].lstrip("0") == target.lstrip("0")]
    if not pick:
        return f"취소할 미체결이 없다 ({target})"
    out = []
    for o in pick:
        r = trading.cancel_order(o["order_no"], o["org_no"], o["remain_qty"], order_dvsn=o["order_dvsn"])
        out.append(f"{'✅' if r.get('success') else '❌'} {o['order_no']} {o['name']} {o['remain_qty']:,}주 "
                   f"{'' if r.get('success') else r.get('message', '')}")
    return "취소\n" + "\n".join(out)


def fills(d: date | None = None) -> str:
    fs = trading.get_today_fills(d)
    if not fs:
        return "오늘 체결 없음"
    buy_amt = sum(f["amount"] for f in fs if f["side"] == "buy")
    sell_amt = sum(f["amount"] for f in fs if f["side"] == "sell")
    lines = [f"✅ 체결 {len(fs)}건 | 매수 {buy_amt:,.0f}원 · 매도 {sell_amt:,.0f}원"]
    for f in fs:
        lines.append(f"· {f['time'][:2]}:{f['time'][2:4]} {'매수' if f['side'] == 'buy' else '매도'} "
                     f"{f['name']}({f['ticker']}) {f['qty']:,}주 @{f['avg_price']:,.0f} = {f['amount']:,.0f}원")
    return "\n".join(lines)


def progress() -> str:
    """대회 시작일(CONTEST_START)부터 오늘까지 체결을 모아 수상 조건과 비교한다."""
    start = date.fromisoformat(config.CONTEST_START) if config.CONTEST_START else date.today().replace(day=1)
    d, total, days, names = start, 0.0, set(), {}
    while d <= date.today():
        if d.weekday() < 5:
            try:
                fs = trading.get_today_fills(d)
            except Exception as e:
                logger.warning("%s 체결 조회 실패: %s", d, e)
                fs = []
            if fs:
                days.add(d)
                for f in fs:
                    total += f["amount"]
                    names[f["ticker"]] = f["name"]
        d += timedelta(days=1)
    ok = lambda c: "✅" if c else "❌"
    return (f"🏁 대회 조건 진행 ({start.isoformat()} ~ 오늘, 체결 기준)\n"
            f"{ok(total >= config.CONTEST_MIN_AMOUNT)} 매매금액 {total:,.0f}원 / {config.CONTEST_MIN_AMOUNT:,.0f}원 (매수+매도)\n"
            f"{ok(len(days) >= config.CONTEST_MIN_DAYS)} 매매일수 {len(days)}일 / {config.CONTEST_MIN_DAYS}일\n"
            f"{ok(len(names) >= config.CONTEST_MIN_STOCKS)} 매매종목 {len(names)}개 / {config.CONTEST_MIN_STOCKS}개 "
            f"(코스피200 · 코스닥150 편입 여부는 직접 확인)\n"
            + (", ".join(f"{n}({t})" for t, n in names.items()) if names else ""))
