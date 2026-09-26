"""
jobs/scan.py — 시그널 스캔

09:05 기준 흐름
  1. KOSPI100 유니버스 로드 (관리종목·거래정지 등 배제)
  2. 종목별 일봉 조회 → 전일까지 확정되는 조건 판정 (5일선 아래 체류 / 추세 / 유동성)
  3. 통과한 종목만 현재가 조회 → 5일선 상향돌파 판정
  4. 점수화 후 정렬

2번에서 대부분이 탈락하므로 3번의 호출 수가 크게 줄어든다.
"""
from __future__ import annotations

import logging
from datetime import date

import config
from core import state, strategy, universe
from core.kis import quotes
from core.strategy import Candidate

logger = logging.getLogger(__name__)


def excluded_tickers() -> set[str]:
    """
    오늘 다시 사면 안 되는 종목: 보유 중이거나 당일 매매한 것.
    익절 직후 같은 판정으로 더 비싸게 되사는 걸 막는다.
    """
    held = {p["ticker"] for p in state.get_positions()}
    traded = state.get_traded_today() if config.NO_SAME_DAY_REENTRY else set()
    return held | traded


def _drop_today(candles: list[dict]) -> list[dict]:
    """장중 조회 시 섞이는 당일 미완성 봉 제거."""
    today = date.today().strftime("%Y%m%d")
    return [c for c in candles if c["date"] != today]


def build_candidates(tickers: list[dict] | None = None) -> tuple[list[Candidate], dict]:
    """전일 종가까지로 확정되는 조건을 통과한 종목 목록."""
    stocks = tickers if tickers is not None else universe.get_universe()
    eligible: list[Candidate] = []
    rejects: dict[str, int] = {}
    errors = 0

    need = strategy.required_bars()

    for s in stocks:
        try:
            candles = _drop_today(quotes.get_daily_candles(s["ticker"], days=need + 5))
            c = strategy.prepare(s["ticker"], s["name"], candles)
        except Exception as e:
            errors += 1
            logger.warning("%s(%s) 일봉 조회 실패: %s", s["name"], s["ticker"], e)
            continue

        if c.eligible:
            eligible.append(c)
        else:
            key = c.reject.split("(")[0].strip()
            rejects[key] = rejects.get(key, 0) + 1

    stats = {"universe": len(stocks), "eligible": len(eligible), "errors": errors, "rejects": rejects}
    logger.info(
        "1차 필터: %d종목 중 %d종목 통과 (오류 %d) %s",
        stats["universe"], stats["eligible"], errors, rejects,
    )
    return eligible, stats


def load_or_build_candidates() -> tuple[list[Candidate], dict, bool]:
    """
    오늘 캐시가 있으면 그걸 쓰고, 없으면 일봉을 받아 만들고 캐시한다.
    반환: (후보, 통계, 캐시에서 왔는지)

    1차 조건은 전일 종가까지로 확정되므로 장중에 바뀌지 않는다.
    그래서 하루 한 번만 계산하고, 이후 재판정은 현재가만 본다.
    """
    cached = state.get_daily_candidates()
    if cached is not None:
        eligible = [strategy.from_dict(d) for d in cached]
        logger.info("당일 후보 캐시 사용: %d종목", len(eligible))
        return eligible, {"universe": "-", "eligible": len(eligible), "errors": 0, "rejects": {}}, True

    eligible, stats = build_candidates()
    state.set_daily_candidates([strategy.to_dict(c) for c in eligible])
    return eligible, stats, False


def find_breakouts(
    candidates: list[Candidate],
    exclude: set[str] | None = None,
) -> tuple[list[Candidate], dict]:
    """현재가를 조회해 5일선 상향돌파한 종목만 추린다. exclude 에 든 종목은 건너뛴다."""
    hits: list[Candidate] = []
    rejects: dict[str, int] = {}
    exclude = exclude or set()

    for c in candidates:
        if c.ticker in exclude:
            rejects["보유/당일매매"] = rejects.get("보유/당일매매", 0) + 1
            continue
        try:
            q = quotes.get_price(c.ticker)
        except Exception as e:
            logger.warning("%s(%s) 현재가 조회 실패: %s", c.name, c.ticker, e)
            continue

        if q["is_halted"]:
            rejects["거래정지"] = rejects.get("거래정지", 0) + 1
            continue
        if q["is_watch"]:
            rejects["투자주의/경고"] = rejects.get("투자주의/경고", 0) + 1
            continue

        if strategy.check_breakout(c, q["price"]):
            hits.append(c)
        else:
            key = c.reject.split("(")[0].strip()
            rejects[key] = rejects.get(key, 0) + 1

    ranked = strategy.rank(hits)
    stats = {"checked": len(candidates), "breakouts": len(ranked), "rejects": rejects}
    logger.info("2차 판정: %d종목 중 %d종목 돌파 %s", len(candidates), len(ranked), rejects)
    return ranked, stats


def scan(use_cache: bool = True) -> tuple[list[Candidate], dict]:
    """
    전체 스캔. (정렬된 돌파 종목, 통계)
    use_cache=False 면 일봉을 다시 받아 후보를 새로 만든다 (09:05 첫 스캔).
    """
    if use_cache:
        eligible, s1, from_cache = load_or_build_candidates()
    else:
        eligible, s1 = build_candidates()
        state.set_daily_candidates([strategy.to_dict(c) for c in eligible])
        from_cache = False
    ranked, s2 = find_breakouts(eligible, exclude=excluded_tickers())
    return ranked, {**s1, **s2, "from_cache": from_cache}


def format_result(ranked: list[Candidate], stats: dict, limit: int = 5) -> str:
    """텔레그램용 요약."""
    head = (
        f"📡 스캔 결과 {date.today().isoformat()}\n"
        f"유니버스 {stats.get('universe', 0)} → 1차통과 {stats.get('eligible', 0)} "
        f"→ 돌파 {stats.get('breakouts', 0)}"
    )
    if not ranked:
        return head + "\n\n조건 충족 종목 없음"

    lines = [head, ""]
    for i, c in enumerate(ranked[:limit], 1):
        lines.append(f"{i}. {strategy.describe(c)}")
    if len(ranked) > limit:
        lines.append(f"... 외 {len(ranked) - limit}종목")
    return "\n".join(lines)
