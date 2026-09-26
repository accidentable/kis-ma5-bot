"""
core/commands.py — 텔레그램 명령어 처리

LLM 없이 정해진 명령만 받는다. 매매 판단은 전부 코드가 하고,
여기는 조회와 비상 개입(일시정지 / 강제청산)을 위한 창구다.

웹훅 람다와 로컬 폴링이 같은 함수를 쓴다.
"""
from __future__ import annotations

import logging

import config
from core import notify, state, trader
from core.kis import quotes, trading

logger = logging.getLogger(__name__)

HELP = """MA5 돌파 역발상 봇

/status   보유 포지션 + 손익
/scan     지금 기준 시그널 스캔 (주문 안 함)
/history  최근 매매 이력
/config   현재 설정
/pause    자동매매 일시정지
/resume   자동매매 재개
/sell 종목코드  강제 청산"""


def _cmd_status() -> str:
    lines = []
    try:
        bal = trading.get_balance()
        lines.append(
            f"💰 순자산 {bal['net_asset']:,.0f}원 | 예수금 {bal['cash']:,.0f}원\n"
            f"평가손익 {bal['pnl_amount']:+,.0f}원"
        )
    except Exception as e:
        lines.append(f"잔고 조회 실패: {e}")

    positions = state.get_positions()
    if not positions:
        lines.append("\n보유 포지션 없음")
    else:
        lines.append("")
        for p in positions:
            try:
                price = quotes.get_price(p["ticker"])["price"]
            except Exception:
                price = 0
            entry = float(p.get("entry_price", 0) or 0)
            pnl = (price / entry - 1) * 100 if (entry and price) else 0
            breakout = float(p.get("breakout_price", 0) or 0)
            intraday_line = trader.intraday_exit_line(p)
            tp_close = trader.take_profit_target(p)
            tp_intra = trader.intraday_tp_line(p)
            lines.append(
                f"📌 {p.get('name', '')}({p['ticker']}) {p['qty']:,}주\n"
                f"   진입 {entry:,.0f} → 현재 {price:,.0f}  {pnl:+.2f}%\n"
                f"   {p.get('hold_days', 1)}/{config.MAX_HOLD_TRADING_DAYS}일차\n"
                f"   익절: 종가 {tp_close:,.0f} / 장중 {tp_intra:,.0f}\n"
                f"   이탈: 종가 {breakout:,.0f}(5일선) / 장중 {intraday_line:,.0f}"
            )

    if state.is_paused():
        lines.append("\n⏸ 자동매매 일시정지 중")
    return "\n".join(lines)


def _cmd_scan() -> str:
    from jobs import scan as scan_job
    ranked, stats = scan_job.scan()
    return scan_job.format_result(ranked, stats)


def _cmd_history() -> str:
    hist = state.get_history(limit=15)
    if not hist:
        return "매매 이력 없음"

    lines = ["📜 최근 매매"]
    for h in reversed(hist):
        lines.append(
            f"· {h.get('exit_date', '')} {h.get('name', '')}({h['ticker']}) "
            f"{h.get('pnl_pct', 0):+.2f}% ({h.get('pnl_amount', 0):+,}원) — {h.get('exit_reason', '')}"
        )
    wins = sum(1 for h in hist if h.get("pnl_pct", 0) > 0)
    total = sum(h.get("pnl_amount", 0) for h in hist)
    lines.append(f"\n{wins}승 {len(hist) - wins}패 | 누적 {total:+,}원")
    return "\n".join(lines)


def _cmd_sell(args: list[str]) -> str:
    if not args:
        return "종목코드를 넣어라. 예: /sell 005930"

    ticker = args[0].strip()
    pos = state.get_position(ticker)
    if pos is None:
        held = ", ".join(p["ticker"] for p in state.get_positions()) or "없음"
        return f"{ticker} 보유 포지션이 없다. 현재 보유: {held}"

    record = trader.exit_position(pos, "수동 청산")
    return "매도 주문을 넣었다." if record else "매도 주문 실패 — 로그를 확인해라."


def handle(text: str, chat_id: int) -> str:
    """명령 문자열을 처리해 응답 텍스트를 반환한다."""
    if not notify.is_allowed(chat_id):
        logger.warning("허용되지 않은 chat_id: %s", chat_id)
        return ""

    parts = text.strip().split()
    if not parts:
        return ""

    cmd = parts[0].lower().lstrip("/").split("@")[0]
    args = parts[1:]

    try:
        if cmd in ("start", "help"):
            return HELP
        if cmd == "status":
            return _cmd_status()
        if cmd == "scan":
            return _cmd_scan()
        if cmd == "history":
            return _cmd_history()
        if cmd == "config":
            return "⚙️ 설정\n" + config.summary()
        if cmd == "pause":
            state.set_paused(True)
            return "⏸ 자동매매를 멈췄다. 보유 종목 청산 감시는 계속된다."
        if cmd == "resume":
            state.set_paused(False)
            return "▶️ 자동매매를 재개했다."
        if cmd == "sell":
            return _cmd_sell(args)
    except Exception as e:
        logger.exception("명령 처리 실패: %s", text)
        return f"⚠️ 처리 중 오류\n{type(e).__name__}: {e}"

    return f"모르는 명령: /{cmd}\n\n{HELP}"
