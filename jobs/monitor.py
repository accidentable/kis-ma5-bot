"""
jobs/monitor.py — 장중 감시 (10분 간격)

  1. 체결 동기화
  2. SPLIT_CANCEL_AT 이 지났으면 분할매수 정리 — 미체결 취소, 포지션 확정 또는 제거
  3. 보유 종목 청산 점검 — 익절은 항상, 급이탈·손절은 STOP_BLACKOUT_UNTIL 이후만
  4. 슬롯이 비었으면 재진입 — 캐시된 후보의 현재가만 조회해 돌파 판정, 한 번에 매수

2→4 순서가 중요하다. 10:00 에 분할이 전량 미체결로 지워지면 같은 틱의 4 가 현재가로 다시 산다.
보유기간 만료는 장 마감 작업(close.py)에서만 본다.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import config
from core import notify, state, trader
from core.kis import quotes
from jobs import entry as entry_job
from jobs import scan as scan_job

logger = logging.getLogger(__name__)


def _now_hhmm() -> str:
    # 서버 시간대가 KST 로 맞춰져 있어야 한다 (setup.sh 가 timedatectl 로 설정).
    return datetime.now().strftime("%H%M")


def _check_exits(stop_blackout: bool) -> list[dict]:
    closed: list[dict] = []
    for pos in state.get_positions():
        try:
            price = quotes.get_price(pos["ticker"])["price"]
        except Exception as e:
            logger.error("%s 현재가 조회 실패: %s", pos["ticker"], e)
            notify.send_error(f"{pos['ticker']} 감시 중 현재가 조회 실패", e)
            continue

        reason = trader.evaluate_exit(pos, price, closing=False, stop_blackout=stop_blackout)
        if reason:
            record = trader.exit_position(pos, reason, price=price)
            if record:
                closed.append(record)
        else:
            entry = float(pos.get("entry_price", 0) or 0)
            pnl = (price / entry - 1) * 100 if entry else 0
            logger.info(
                "%s %s: %s원 %+.2f%% (5일선 %s, 급이탈선 %s%s)",
                pos["ticker"], pos.get("name", ""),
                f"{price:,.0f}", pnl,
                f"{float(pos.get('breakout_price', 0)):,.0f}",
                f"{trader.intraday_exit_line(pos):,.0f}",
                " — 손절 유예 중" if stop_blackout else "",
            )
    return closed


def _try_reentry() -> dict:
    """슬롯이 비어 있고 시간대가 맞으면 캐시된 후보로 재진입을 시도한다."""
    if not config.INTRADAY_REENTRY:
        return {"skipped": "비활성"}
    if state.is_paused():
        return {"skipped": "일시정지"}
    if len(state.get_positions()) >= config.MAX_POSITIONS:
        return {"skipped": "보유한도"}

    now = _now_hhmm()
    if now < config.REENTRY_START:
        return {"skipped": f"시작 전 ({now})"}
    if now >= config.REENTRY_CUTOFF:
        return {"skipped": f"컷오프 ({now})"}

    if not entry_job.entry_lock.acquire(blocking=False):
        logger.info("진입 작업이 실행 중 — 이번 재진입 건너뜀")
        return {"skipped": "잠금"}
    try:
        # 캐시가 없으면(08:40 작업 실패 등) 여기서 한 번 만든다. 그 회차만 느리다.
        eligible, _, from_cache = scan_job.load_or_build_candidates()
        if not from_cache:
            logger.warning("당일 후보 캐시가 없어 감시 작업에서 새로 만들었다")

        ranked, stats = scan_job.find_breakouts(eligible, exclude=scan_job.excluded_tickers())
        if not ranked:
            logger.info("재진입: 돌파 종목 없음 (%s)", stats.get("rejects"))
            return {"entered": [], "stats": stats}

        dip = entry_job.morning_dip_window(now)
        result = entry_job.enter_best(ranked, source=("장중 눌림대기" if dip else "장중 재진입"), dip=dip)
    finally:
        entry_job.entry_lock.release()

    if result["entered"]:
        logger.info("재진입 매수: %s", [p["ticker"] for p in result["entered"]])
    elif result["skipped"]:
        logger.info("재진입: 후보 전부 자금 부족 %s", result["skipped"])
    return {**result, "stats": stats}


def run(force: bool = False) -> dict:
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}

    now = _now_hhmm()
    stop_blackout = now < config.STOP_BLACKOUT_UNTIL

    trader.sync_fills()

    finalized: list[dict] = []
    if now >= config.SPLIT_CANCEL_AT:
        finalized = trader.finalize_accumulation()

    closed = _check_exits(stop_blackout)
    reentry = _try_reentry()

    return {
        "now": now,
        "stop_blackout": stop_blackout,
        "finalized": finalized,
        "closed": closed,
        "reentry": reentry,
    }
