"""
core/nearhigh.py — 52주 신고가 근접 로테이션 (STRATEGY=near_high)

규칙 (백테스트 backtest/lab 과 같은 정의)
  유니버스  코스피+코스닥 보통주 시총 상위 NH_UNIVERSE_TOP (관리·정지·경고 종목 제외)
  필터      전일 종가 ≥ NH_MIN_PRICE, 20일 평균 거래대금 ≥ NH_MIN_VALUE, NH_MOM_DAYS 일 수익률 > 0,
            일봉이 NH_HIGH_LOOKBACK 개 이상 (신고가를 정의할 수 있는 종목만)
  점수      전일 종가 / 최근 NH_HIGH_LOOKBACK 거래일 최고가 (전일 포함). 1 이면 신고가, 높을수록 먼저
  보유      상위 NH_SLOTS 종목을 계좌 1/NH_SLOTS 씩. 비싸서 1주도 못 사면 다음 순위
  교체      NH_HOLD_DAYS 거래일마다 다시 순위를 매겨, 상위 NH_SLOTS 밖으로 밀린 종목만 팔고 새로 산다

근거 (KRX 전종목 2010 ~ 2026, 상장폐지 포함, 60만원·정수 주식·매도세 0.20% 반영, 한 달짜리 구간)
  2011 ~ 2018 에서 고르고, 2019 ~ 2024.9 에서 계열 비교, 2024.10 ~ 에서 한 번 확인한 결과가 이것이다.
  한 달 평균 +0.5% / +0.9% / +0.9%. 시장(동일가중)보다 약간 낮다 — 확실한 초과수익이 아니라,
  검증한 후보 중 모든 구간에서 플러스였고 거래가 적어 비용이 작은 전략이라서 골랐다.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date

import config

logger = logging.getLogger(__name__)


@dataclass
class Pick:
    ticker: str
    name: str
    market: str = ""
    marcap: float = 0.0          # 억원
    prev_close: float = 0.0
    high_n: float = 0.0          # 최근 N 거래일 최고가
    score: float = 0.0           # prev_close / high_n
    mom: float = 0.0             # NH_MOM_DAYS 일 수익률
    avg_value: float = 0.0       # 20일 평균 거래대금
    bars: int = 0
    eligible: bool = False
    reject: str = ""


def evaluate(ticker: str, name: str, candles: list[dict], *, market: str = "", marcap: float = 0.0) -> Pick:
    """candles: 날짜 오름차순, 마지막이 전일. 당일 미완성 봉은 호출부에서 뺀다."""
    p = Pick(ticker=ticker, name=name, market=market, marcap=marcap, bars=len(candles))
    n, m = config.NH_HIGH_LOOKBACK, config.NH_MOM_DAYS
    if len(candles) < max(n, m + 1, 20):
        p.reject = f"일봉 부족 ({len(candles)}/{n})"
        return p
    closes = [c["close"] for c in candles]
    p.prev_close = closes[-1]
    p.high_n = max(c["high"] for c in candles[-n:])
    p.score = p.prev_close / p.high_n if p.high_n > 0 else 0.0
    p.mom = closes[-1] / closes[-1 - m] - 1 if closes[-1 - m] > 0 else 0.0
    p.avg_value = sum(c["value"] for c in candles[-20:]) / 20
    if p.prev_close < config.NH_MIN_PRICE:
        p.reject = f"주가 {p.prev_close:,.0f}원 < {config.NH_MIN_PRICE:,.0f}"
    elif p.avg_value < config.NH_MIN_VALUE:
        p.reject = f"거래대금 {p.avg_value / 1e8:,.1f}억 부족"
    elif p.mom <= 0:
        p.reject = f"{m}일 수익률 {p.mom * 100:+.1f}% ≤ 0"
    else:
        p.eligible = True
    return p


def rank(picks: list[Pick]) -> list[Pick]:
    """적격 종목을 점수 내림차순으로. 동점은 시총 큰 순."""
    ok = [p for p in picks if p.eligible]
    return sorted(ok, key=lambda p: (-p.score, -p.marcap, p.ticker))


def describe(p: Pick) -> str:
    return (f"{p.name}({p.ticker}) {p.prev_close:,.0f}원 | {config.NH_HIGH_LOOKBACK}일 최고가 대비 "
            f"{p.score * 100:.1f}% | {config.NH_MOM_DAYS}일 {p.mom * 100:+.1f}% | 시총 {p.marcap / 1e4:,.1f}조")


def to_dict(p: Pick) -> dict:
    return asdict(p)


def from_dict(d: dict) -> Pick:
    return Pick(**{k: v for k, v in d.items() if k in Pick.__dataclass_fields__})


def today_str() -> str:
    return date.today().isoformat()
