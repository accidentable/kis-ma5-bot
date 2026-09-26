"""
jobs/entry.py — 진입

  run()          09:05 작업. 일봉을 받아 후보를 만들고(당일 캐시), 돌파 종목을 매수한다.
  enter_best()   돌파 종목 목록에서 순위대로 매수를 시도한다. 장중 재진입(monitor)도 이걸 쓴다.

두 경로가 동시에 매수하지 않도록 잠금을 둔다. 스케줄러가 스레드풀이라
09:05 작업이 길어지면 09:10 감시와 겹칠 수 있다.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime

import config
from core import notify, state, trader
from core.kis import quotes
from core.strategy import Candidate
from jobs import scan as scan_job

logger = logging.getLogger(__name__)

# 진입 작업 상호배제. 잠금을 못 잡으면 그 회차는 건너뛴다(대기하지 않음).
entry_lock = threading.Lock()


def _now_hhmm() -> str:
    return datetime.now().strftime("%H%M")


def morning_dip_window(now_hhmm: str) -> bool:
    """
    SPLIT_CANCEL_AT 전이면 '장 초반 눌림 대기' 구간이다. 이 구간의 진입은 즉시 매수가 아니라
    현재가 − 0.5ATR 지정가를 걸어두고 급락을 기다린다.
    """
    return config.PREMARKET_SPLIT_ENTRY and now_hhmm < config.SPLIT_CANCEL_AT


def enter_best(ranked: list[Candidate], *, source: str, dip: bool = False) -> dict:
    """
    순위대로 매수를 시도한다. 자금 부족이면 다음 순위로, 성공하면 멈춘다.
    source 는 알림에 붙는 출처 라벨. dip=True 면 즉시 매수 대신 눌림 대기 주문을 건다.
    """
    open_slots = config.MAX_POSITIONS - len(state.get_positions())
    if open_slots <= 0:
        return {"entered": [], "skipped": [], "reason": "보유한도"}

    limit_rank = config.ENTRY_FALLBACK_MAX_RANK or len(ranked)
    pool = ranked[:limit_rank]

    entered: list[dict] = []
    skipped: list[str] = []

    for rank, c in enumerate(pool, 1):
        if len(entered) >= open_slots:
            break

        if dip:
            pos, reason = trader.enter_split(c, c.price, source=source)
        else:
            pos, reason = trader.enter(c)
        if pos:
            pos["rank"] = rank
            pos["source"] = source
            state.update_position(c.ticker, rank=rank, source=source)
            entered.append(pos)
            if rank > 1:
                notify.send(
                    f"ℹ️ [{source}] 1~{rank - 1}순위는 자금 부족으로 건너뛰고 {rank}순위를 매수했다.\n"
                    + "\n".join(f"  {i}. {s}" for i, s in enumerate(skipped, 1))
                )
            continue

        if reason == "자금부족":
            skipped.append(f"{c.name}({c.ticker}) {c.price:,.0f}원")
            continue
        if reason in ("일시정지", "보유한도"):
            break
        # 주문거부는 이미 알림이 나갔다. 다음 후보로.

    return {"entered": entered, "skipped": skipped, "reason": ""}


def run(force: bool = False) -> dict:
    """09:05 진입 작업."""
    today = date.today()

    if not force and not quotes.is_open_day(today):
        logger.info("%s 은 휴장일 — 진입 작업 생략", today)
        return {"skipped": "휴장일"}

    # 보유 포지션의 기준선을 오늘 값으로 맞춘다. 빼먹으면 어제 5일선으로 청산 판정한다.
    trader.refresh_breakout_prices()
    trader.sync_fills()

    if state.is_paused():
        notify.send("⏸ 자동매매 일시정지 상태 — 진입 생략")
        return {"skipped": "일시정지"}

    # 08:50 선주문이 걸려 있으면: 정리 시각이 됐으면 여기서 정리한다 (미체결 → 취소하고 아래로 진행,
    # 체결 → 확정하고 물러선다). 아직 정리 시각 전이면 물러선다.
    if any(p.get("accumulating") for p in state.get_positions()):
        if _now_hhmm() >= config.SPLIT_CANCEL_AT:
            finalized = trader.finalize_accumulation()
            logger.info("단일 진입 전 선주문 정리: %s", finalized)
        if any(p.get("accumulating") for p in state.get_positions()):
            logger.info("선주문 진행 중 — 단일 진입 생략")
            return {"skipped": "분할진행중"}

    if not entry_lock.acquire(blocking=False):
        logger.warning("다른 진입 작업이 실행 중 — 단일 진입 생략")
        return {"skipped": "잠금"}
    try:
        # 08:40 준비가 만든 당일 캐시를 쓴다 (날짜가 다르면 자동으로 새로 만든다).
        ranked, stats = scan_job.scan(use_cache=True)

        if len(state.get_positions()) >= config.MAX_POSITIONS:
            logger.info("보유 한도 도달 — 후보만 캐시하고 매수 생략")
            return {"skipped": "보유한도", "stats": stats}

        if not ranked:
            notify.send(scan_job.format_result(ranked, stats)
                        + ("\n\n장중 10분마다 다시 본다." if config.INTRADAY_REENTRY else ""))
            return {"entered": [], "stats": stats}

        dip = morning_dip_window(_now_hhmm())
        result = enter_best(ranked, source=("09:05 눌림대기" if dip else "09:05 진입"), dip=dip)
    finally:
        entry_lock.release()

    if not result["entered"]:
        msg = scan_job.format_result(ranked, stats)
        if result["skipped"]:
            msg += (
                f"\n\n⚠️ 매수 실패 — 후보 {len(result['skipped'])}종목 모두 자금 부족\n"
                + "\n".join(f"  · {s}" for s in result["skipped"])
            )
        else:
            msg += "\n\n⚠️ 돌파 종목은 있었으나 매수가 접수되지 않음"
        notify.send(msg)

    return {**result, "stats": stats}
