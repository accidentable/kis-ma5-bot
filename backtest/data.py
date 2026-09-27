"""
backtest/data.py — 과거 일봉 수집 (한투 API, 종목별 캐시)

한투 일봉 API 는 한 번에 100건까지라, 날짜 구간을 뒤로 밀어가며 여러 번 받는다.
실전 봇과 같은 소스·같은 수정주가(FID_ORG_ADJ_PRC=0)라 판정이 실전과 일치한다.
받은 결과는 data/bt_cache/<종목>.json 에 저장해 재실행 때 다시 받지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta

from core.kis import client
from core.kis.quotes import TR_DAILY_CHART, _PATH_DAILY, _f, _i

logger = logging.getLogger(__name__)

_WINDOW_DAYS = 140   # 한 번 요청하는 달력일 구간 (≈ 95 거래일, 100건 제한 안쪽)


def _fetch_window(ticker: str, start: date, end: date) -> list[dict]:
    data = client.get(
        _PATH_DAILY, TR_DAILY_CHART,
        {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": end.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        },
    )
    out = []
    for r in data.get("output2") or []:
        d = str(r.get("stck_bsop_date", "")).strip()
        close = _f(r.get("stck_clpr"))
        if not d or close <= 0:
            continue
        out.append({
            "date": d,
            "open": _f(r.get("stck_oprc")) or close,
            "high": _f(r.get("stck_hgpr")) or close,
            "low": _f(r.get("stck_lwpr")) or close,
            "close": close,
            "volume": _i(r.get("acml_vol")),
            "value": _f(r.get("acml_tr_pbmn")),
        })
    return out


def fetch_history(ticker: str, since: date, until: date) -> list[dict]:
    """since~until 일봉 전체 (날짜 오름차순). 상장 전 구간은 자연히 비어서 멈춘다."""
    by_date: dict[str, dict] = {}
    end = until
    while end >= since:
        start = max(since, end - timedelta(days=_WINDOW_DAYS))
        rows = _fetch_window(ticker, start, end)
        for r in rows:
            by_date[r["date"]] = r
        if not rows:
            # 이 구간에 거래가 없다 — 상장 전이거나 장기 정지. 더 과거로는 안 간다.
            break
        oldest = min(datetime.strptime(r["date"], "%Y%m%d").date() for r in rows)
        # 100건 제한에 걸려 구간 앞쪽이 잘렸으면 oldest 바로 전날부터 이어서 받는다
        end = min(start, oldest) - timedelta(days=1)
    return [by_date[k] for k in sorted(by_date)]


def load_all(tickers: list[str], cache_dir: str, since: date, until: date,
             refresh: bool = False, extend: bool = False) -> dict[str, list[dict]]:
    """extend=True 면 .since 표시가 없거나 요청보다 짧은 캐시를 다시 받는다 (fetch 용). 읽기만 할 땐 False."""
    os.makedirs(cache_dir, exist_ok=True)
    out: dict[str, list[dict]] = {}
    todo = []
    for t in tickers:
        p = os.path.join(cache_dir, f"{t}.json")
        mark = p[:-5] + ".since"
        # 예전에 더 짧은 기간으로 받은 캐시면 다시 받는다 (.since 에 그때 요청한 시작일을 적어 둔다)
        short = os.path.exists(mark) and open(mark).read().strip() > since.strftime("%Y%m%d")
        if not refresh and os.path.exists(p) and not short and (os.path.exists(mark) or not extend):
            with open(p, encoding="utf-8") as f:
                out[t] = json.load(f)
        else:
            todo.append(t)

    logger.info("일봉: 캐시 %d종목 / 새로 받을 종목 %d", len(out), len(todo))
    t0 = time.monotonic()
    for n, t in enumerate(todo, 1):
        try:
            bars = fetch_history(t, since, until)
        except Exception as e:
            logger.warning("%s 일봉 수집 실패: %s", t, e)
            continue
        with open(os.path.join(cache_dir, f"{t}.json"), "w", encoding="utf-8") as f:
            json.dump(bars, f)
        with open(os.path.join(cache_dir, f"{t}.since"), "w") as f:
            f.write(since.strftime("%Y%m%d"))
        out[t] = bars
        if n % 10 == 0 or n == len(todo):
            el = time.monotonic() - t0
            logger.info("일봉 수집 %d/%d (%s: %d봉) — 남은 시간 약 %.0f분", n, len(todo), t, len(bars),
                        el / n * (len(todo) - n) / 60)
    return out
