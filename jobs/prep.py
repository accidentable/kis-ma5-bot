"""
jobs/prep.py — 개장 전 작업

  run()               08:40  일봉을 받아 1차 후보를 계산하고 그날 캐시한다. 느린 작업(99종목 일봉,
                             유량제한 때문에 2~3분)을 개장 전으로 빼서 이후 판정은 현재가만 본다.
  premarket_entry()   08:50  프리마켓(NXT) 가격으로 돌파를 판정하고, 1순위에 본장 전 분할 매수를 걸어둔다.
                             아무것도 못 걸었으면 09:05 단일 진입이 대체한다.
"""
from __future__ import annotations

import logging
from datetime import date

import config
from core import notify, state, strategy, trader
from core.kis import quotes
from jobs import entry as entry_job
from jobs import scan as scan_job

logger = logging.getLogger(__name__)


def run(force: bool = False) -> dict:
    """08:40 — 후보 계산·캐시 + 보유 종목 기준선 갱신."""
    if not force and not quotes.is_open_day(date.today()):
        logger.info("휴장일 — 준비 작업 생략")
        return {"skipped": "휴장일"}

    trader.refresh_breakout_prices()
    trader.sync_fills()

    eligible, stats = scan_job.build_candidates()
    state.set_daily_candidates([strategy.to_dict(c) for c in eligible])

    held = ", ".join(f"{p.get('name', '')}({p['ticker']})" for p in state.get_positions()) or "없음"
    notify.send(
        f"🌅 개장 전 준비 {date.today().isoformat()}\n"
        f"유니버스 {stats.get('universe', 0)} → 1차 통과 {len(eligible)}종목\n"
        f"보유: {held}"
        + (f"\n오류 {stats['errors']}건" if stats.get("errors") else "")
    )
    return {"eligible": len(eligible), "stats": stats}


def premarket_entry(force: bool = False) -> dict:
    """08:50 — 프리마켓 가격으로 판정 → 분할 매수."""
    if not config.PREMARKET_SPLIT_ENTRY:
        return {"skipped": "비활성"}
    if not force and not quotes.is_open_day(date.today()):
        return {"skipped": "휴장일"}
    if state.is_paused():
        return {"skipped": "일시정지"}
    if len(state.get_positions()) >= config.MAX_POSITIONS:
        logger.info("보유 한도 도달 — 프리마켓 진입 생략")
        return {"skipped": "보유한도"}

    if not entry_job.entry_lock.acquire(blocking=False):
        return {"skipped": "잠금"}
    try:
        eligible, _, from_cache = scan_job.load_or_build_candidates()
        if not from_cache:
            logger.warning("준비 캐시가 없어 08:50 에 새로 만들었다 (08:40 작업이 안 돌았나?)")

        exclude = scan_job.excluded_tickers()
        hits: list[strategy.Candidate] = []
        no_price: list[str] = []
        for c in eligible:
            if c.ticker in exclude:
                continue
            price = quotes.get_premarket_price(c.ticker)
            if price is None:
                no_price.append(c.ticker)
                continue
            c.premarket_price = price
            if strategy.check_breakout(c, price):
                hits.append(c)

        ranked = strategy.rank(hits)
        if no_price:
            logger.warning("프리마켓 가격 없음 %d종목: %s", len(no_price), no_price[:10])

        if not ranked:
            notify.send(
                f"🔭 08:50 프리마켓 — 5일선 위 종목 없음 "
                f"(후보 {len(eligible)}, 가격 없음 {len(no_price)})\n"
                f"{config.ENTRY_TIME[:2]}:{config.ENTRY_TIME[2:]} 단일 진입으로 넘어간다."
            )
            return {"entered": [], "checked": len(eligible), "no_price": len(no_price)}

        limit_rank = config.ENTRY_FALLBACK_MAX_RANK or len(ranked)
        entered: list[dict] = []
        skipped: list[str] = []
        for rank, c in enumerate(ranked[:limit_rank], 1):
            pos, reason = trader.enter_split(c, c.premarket_price, source="08:50 눌림대기")
            if pos:
                state.update_position(c.ticker, rank=rank)
                entered.append(pos)
                if rank > 1:
                    notify.send(f"ℹ️ 1~{rank - 1}순위는 자금 부족 — {rank}순위에 분할매수\n"
                                + "\n".join(f"  {i}. {s}" for i, s in enumerate(skipped, 1)))
                break
            if reason == "자금부족":
                skipped.append(f"{c.name}({c.ticker}) {c.premarket_price:,.0f}원")
                continue
            if reason in ("일시정지", "보유한도"):
                break
    finally:
        entry_job.entry_lock.release()

    if not entered:
        notify.send(
            "🔭 08:50 프리마켓 — 돌파 종목은 있었으나 분할매수를 못 걸었다\n"
            + ("\n".join(f"  · {s}" for s in skipped) if skipped else "주문 거부")
        )
    return {"entered": entered, "skipped": skipped, "ranked": [c.ticker for c in ranked]}
