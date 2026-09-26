"""
jobs/close.py — 장 마감 정리 (15:15 전후)

  1. 미체결 주문 취소 (한투 주문은 당일 소멸이지만 명시적으로 정리해 상태를 맞춘다)
  2. 보유기간 만료 / 5일선 종가 이탈 점검 후 청산
  3. 일일 리포트 발송
"""
from __future__ import annotations

import logging
from datetime import date

import config
from core import notify, state, trader
from core.kis import quotes, trading

logger = logging.getLogger(__name__)


def _report(closed: list[dict]) -> str:
    lines = [f"📕 마감 리포트 {date.today().isoformat()}"]

    try:
        bal = trading.get_balance()
        lines.append(
            f"순자산 {bal['net_asset']:,.0f}원 | 예수금 {bal['cash']:,.0f}원 | "
            f"평가손익 {bal['pnl_amount']:+,.0f}원"
        )
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")

    positions = state.get_positions()
    if positions:
        lines.append("\n[보유]")
        for p in positions:
            try:
                price = quotes.get_price(p["ticker"])["price"]
                entry = float(p.get("entry_price", 0) or 0)
                pnl = (price / entry - 1) * 100 if entry else 0
                lines.append(
                    f"· {p.get('name', '')}({p['ticker']}) {p['qty']:,}주 "
                    f"{entry:,.0f} → {price:,.0f} {pnl:+.2f}% | "
                    f"{p.get('hold_days', 1)}일차 | 5일선 {float(p.get('breakout_price', 0)):,.0f}"
                )
            except Exception:
                lines.append(f"· {p.get('name', '')}({p['ticker']}) 시세 조회 실패")
    else:
        lines.append("\n[보유] 없음")

    if closed:
        lines.append("\n[오늘 청산]")
        for r in closed:
            lines.append(
                f"· {r.get('name', '')}({r['ticker']}) {r['pnl_pct']:+.2f}% "
                f"({r['pnl_amount']:+,}원) — {r['exit_reason']}"
            )

    hist = state.get_history(limit=20)
    if hist:
        wins = sum(1 for h in hist if h.get("pnl_pct", 0) > 0)
        total = sum(h.get("pnl_amount", 0) for h in hist)
        lines.append(
            f"\n최근 {len(hist)}건: {wins}승 {len(hist) - wins}패 | 누적 {total:+,}원"
        )

    return "\n".join(lines)


def run(force: bool = False, send_report: bool = True) -> dict:
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}

    # 10:00 정리가 어떤 이유로 안 돌았어도 마감 전엔 분할 상태를 반드시 닫는다
    trader.finalize_accumulation()
    cancelled = trader.cancel_all_pending()
    trader.sync_fills()

    closed = []
    for pos in state.get_positions():
        try:
            price = quotes.get_price(pos["ticker"])["price"]
        except Exception as e:
            logger.error("%s 현재가 조회 실패: %s", pos["ticker"], e)
            continue

        reason = trader.evaluate_exit(pos, price, closing=True)
        if reason:
            record = trader.exit_position(pos, reason, price=price)
            if record:
                closed.append(record)

    if send_report:
        notify.send(_report(closed))

    logger.info("마감 정리: 미체결 취소 %d건, 청산 %d건", len(cancelled), len(closed))
    return {"cancelled": cancelled, "closed": closed}
