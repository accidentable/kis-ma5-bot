"""
core/strategy.py — MA5 상향돌파 역발상 전략

진입
  1) 직전 BELOW_LOOKBACK 거래일 중 BELOW_MIN_DAYS 일 이상 종가가 그날의 5일선 아래
  2) 전일 종가 < 전일 5일선            (어제까지도 5일선 아래에 있었다)
  3) 현재가 > 직전 4거래일 종가 평균    (지금 5일선을 위로 넘었다)
  4) 전일 종가 대비 상승률 ≤ MAX_CHASE_PCT  (이미 급등한 건 추격하지 않는다)
  5) 20일선 > 60일선                  (골든크로스 상태 — 중장기 추세가 살아있다)

(3) 이 왜 5일선 돌파와 같은 말인지:
  오늘 5일선 = (C1+C2+C3+C4+P)/5   (C1~C4 = 직전 4거래일 종가, P = 현재가)
  P > (C1+C2+C3+C4+P)/5  ⟺  4P > C1+C2+C3+C4  ⟺  P > (C1+C2+C3+C4)/4
  즉 장중 이동평균을 재계산할 필요 없이 직전 4일 종가 평균 하나로 판정된다.
  청산 조건인 '5일선 이탈' 도 부등호만 뒤집으면 된다.

(5) 를 "종가 > 60일선" 대신 "20일선 > 60일선" 으로 두는 이유:
  하락추세 종목도 하루 반등이면 종가가 60일선을 잠깐 넘는다. 20일선이 60일선 위에
  있으려면 최근 한 달의 평균이 석 달 평균보다 높아야 하므로, 반등 하루로는 못 속인다.

청산
  +TAKE_PROFIT_PCT% 익절 | 5일선 이탈 | MAX_HOLD_TRADING_DAYS 거래일 경과
  (손절은 config.USE_STOP_LOSS 로 선택)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from statistics import fmean

import config

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """전일 종가까지의 정보로 확정되는 종목 상태. 장 시작 전에 미리 계산해둔다."""
    ticker: str
    name: str

    breakout_price: float = 0.0    # 직전 4거래일 종가 평균 = 오늘의 5일선 돌파 기준가
    prev_close: float = 0.0
    prev_ma5: float = 0.0
    prev_disparity: float = 100.0  # 전일 5일 이격도 (종가/5일선×100)
    below_days: int = 0            # 최근 lookback 중 5일선 아래였던 날 수
    atr: float = 0.0
    avg_value: float = 0.0         # 20일 평균 거래대금 (원)

    # 중장기 추세 (전일 기준)
    ma_trend_short: float = 0.0    # 20일선
    ma_trend_long: float = 0.0     # 60일선
    ma_spread: float = 0.0         # (20일선/60일선 − 1) × 100 — 추세 강도
    trend_disparity: float = 100.0  # 전일 60일 이격도 (참고용)
    cross_age: int = -1            # 골든크로스 후 경과 거래일 (참고용, 랭킹엔 안 씀)

    # 수렴 → 발산 자리인가 (전일 기준)
    ma_squeeze: float = 0.0        # 5·20·60일선 최고−최저 / 주가 × 100. 작을수록 뭉쳐 있다
    vol_contraction: float = 1.0   # ATR(5)/ATR(20). 1 미만이면 최근 변동폭이 줄고 있다

    eligible: bool = False         # 전일까지의 조건을 통과했는가
    reject: str = ""               # 탈락 사유

    # 장중 판정 결과
    premarket_price: float = 0.0   # 08:50 프리마켓 가격 (분할 진입 기준가)
    price: float = 0.0
    change_pct: float = 0.0
    thrust: float = 0.0            # 돌파 강도 (%)
    score: float = 0.0
    score_detail: dict = field(default_factory=dict)


def required_bars() -> int:
    """prepare() 가 필요로 하는 최소 일봉 개수."""
    need = max(config.MA_PERIOD + config.BELOW_LOOKBACK, 20)
    if config.USE_TREND_FILTER:
        # 60일선을 CROSS_LOOKBACK 일 전까지 거슬러 계산해야 크로스 시점을 찾는다
        need = max(need, config.TREND_MA_LONG + config.CROSS_LOOKBACK)
    return need


def _sma(values: list[float], period: int) -> float:
    return fmean(values[-period:]) if len(values) >= period else 0.0


def _sma_at(values: list[float], idx: int, period: int) -> float:
    """values[idx] 를 마지막으로 하는 period 일 단순이동평균. 데이터 부족이면 0."""
    if idx + 1 < period:
        return 0.0
    return fmean(values[idx - period + 1: idx + 1])


def _atr(candles: list[dict], period: int = 14) -> float:
    """Wilder 의 True Range 단순평균."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for prev, cur in zip(candles[-(period + 1):-1], candles[-period:]):
        tr = max(
            cur["high"] - cur["low"],
            abs(cur["high"] - prev["close"]),
            abs(cur["low"] - prev["close"]),
        )
        trs.append(tr)
    return fmean(trs) if trs else 0.0


def _golden_cross_age(closes: list[float], short: int, long: int, lookback: int) -> int:
    """
    가장 최근 골든크로스(단기선이 장기선을 위로 넘은 날)가 며칠 전인지.
    0 = 어제 넘음. lookback 안에 없으면 lookback 을 돌려준다 (오래된 추세).
    """
    last = len(closes) - 1
    for k in range(lookback):
        idx = last - k
        if idx - 1 < long - 1:
            break
        s_now, l_now = _sma_at(closes, idx, short), _sma_at(closes, idx, long)
        s_prev, l_prev = _sma_at(closes, idx - 1, short), _sma_at(closes, idx - 1, long)
        if l_now <= 0 or l_prev <= 0:
            break
        if s_now > l_now and s_prev <= l_prev:
            return k
    return lookback


def prepare(ticker: str, name: str, candles: list[dict]) -> Candidate:
    """
    일봉(날짜 오름차순)으로 전일까지 확정되는 조건을 계산한다.
    candles 의 마지막 원소는 **전일** 이어야 한다. 장중에 호출하면 당일 미완성
    봉이 섞일 수 있으므로 호출부에서 잘라내고 넘긴다.
    """
    c = Candidate(ticker=ticker, name=name)

    need = required_bars()
    if len(candles) < need:
        c.reject = f"일봉 부족 ({len(candles)}/{need})"
        return c

    closes = [x["close"] for x in candles]
    p = config.MA_PERIOD

    c.prev_close = closes[-1]
    c.prev_ma5 = _sma(closes, p)
    # 오늘의 돌파 기준가: 직전 (MA_PERIOD - 1) 개 종가의 평균
    c.breakout_price = fmean(closes[-(p - 1):])
    c.atr = _atr(candles)
    c.avg_value = fmean([x["value"] for x in candles[-20:]])

    if c.prev_ma5 > 0:
        c.prev_disparity = c.prev_close / c.prev_ma5 * 100

    if config.USE_TREND_FILTER:
        s, l = config.TREND_MA_SHORT, config.TREND_MA_LONG
        c.ma_trend_short = _sma(closes, s)
        c.ma_trend_long = _sma(closes, l)
        if c.ma_trend_long > 0:
            c.ma_spread = (c.ma_trend_short / c.ma_trend_long - 1) * 100
            c.trend_disparity = c.prev_close / c.ma_trend_long * 100
        c.cross_age = _golden_cross_age(closes, s, l, config.CROSS_LOOKBACK)

    # 수렴도: 5·20·60일선이 얼마나 한 점에 모였나. 삼각수렴 꼭짓점에선 이 값이 작다.
    mas = [m for m in (c.prev_ma5, c.ma_trend_short, c.ma_trend_long) if m > 0]
    if len(mas) >= 2 and c.prev_close > 0:
        c.ma_squeeze = (max(mas) - min(mas)) / c.prev_close * 100

    # 변동성 축소: 최근 5일 평균 변동폭이 20일 평균보다 작아졌나. 봉이 작아지며 조여드는 구간.
    atr5, atr20 = _atr(candles, 5), _atr(candles, 20)
    if atr20 > 0:
        c.vol_contraction = atr5 / atr20

    # 최근 lookback 거래일 각각에 대해 "그날 종가 < 그날 MA5" 를 센다
    below = 0
    for i in range(1, config.BELOW_LOOKBACK + 1):
        idx = len(closes) - i
        if idx < p - 1:
            break
        ma = fmean(closes[idx - p + 1: idx + 1])
        if ma > 0 and closes[idx] < ma:
            below += 1
    c.below_days = below

    # ── 전일까지 확정되는 조건 검사 ─────────────────────────
    if c.prev_close >= c.prev_ma5:
        c.reject = f"전일 종가가 이미 5일선 위 (이격도 {c.prev_disparity:.1f})"
        return c
    if below < config.BELOW_MIN_DAYS:
        c.reject = f"5일선 아래 체류 부족 ({below}/{config.BELOW_MIN_DAYS}일)"
        return c
    if config.USE_TREND_FILTER:
        s, l = config.TREND_MA_SHORT, config.TREND_MA_LONG
        if c.ma_trend_short <= c.ma_trend_long:
            c.reject = f"{s}일선이 {l}일선 아래 (데드크로스 상태, 이격 {c.ma_spread:+.1f}%)"
            return c
        if config.TREND_REQUIRE_PRICE_ABOVE and c.trend_disparity < 100:
            c.reject = f"{l}일선 아래 (이격도 {c.trend_disparity:.1f})"
            return c
    if c.avg_value < config.MIN_TRADING_VALUE:
        c.reject = f"거래대금 부족 ({c.avg_value / 1e8:.0f}억 < {config.MIN_TRADING_VALUE / 1e8:.0f}억)"
        return c

    c.eligible = True
    return c


def check_breakout(c: Candidate, price: float) -> bool:
    """
    장중 현재가로 5일선 상향돌파 여부를 판정한다.
    통과하면 c.price / c.change_pct / c.thrust 를 채운다.
    """
    if not c.eligible or price <= 0 or c.breakout_price <= 0:
        return False

    c.price = price
    c.change_pct = (price / c.prev_close - 1) * 100 if c.prev_close > 0 else 0.0
    c.thrust = (price / c.breakout_price - 1) * 100

    if price <= c.breakout_price:
        c.reject = f"아직 5일선 아래 (기준 {c.breakout_price:,.0f} / 현재 {price:,.0f})"
        return False
    if c.change_pct > config.MAX_CHASE_PCT:
        c.reject = f"이미 급등 ({c.change_pct:+.1f}% > {config.MAX_CHASE_PCT:.1f}%)"
        return False

    # 캐시된 후보를 하루에 여러 번 재판정하므로, 이전 판정의 탈락 사유가 남지 않게 지운다
    c.reject = ""
    return True


def is_ma5_broken(breakout_price: float, price: float) -> bool:
    """보유 종목의 5일선 이탈 여부. 돌파 판정의 부등호를 뒤집은 것."""
    return breakout_price > 0 and price < breakout_price


def _percentile_ranks(values: list[float]) -> list[float]:
    """값 목록을 0~1 백분위로. 동점은 같은 값을 받는다."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        pr = (i + j) / 2 / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = pr
        i = j + 1
    return ranks


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """
    돌파한 후보들에 점수를 매겨 내림차순 정렬한다.

    squeeze      이평선 수렴도가 작을수록 ↑ — 5·20·60일선이 한 점에 모인 자리 (1순위)
    contraction  ATR(5)/ATR(20) 이 작을수록 ↑ — 봉이 작아지며 조여든 구간
    depth        전일 5일 이격도가 낮을수록 ↑ — 얼마나 깊이 눌렸다 올라오는가
    thrust       돌파 강도 ↑
    trend        20일선/60일선 이격 ↑ — 추세 강도
    value        20일 평균 거래대금 ↑ — 전액 매수를 소화할 유동성

    수렴(squeeze+contraction)이 1순위인 이유: 이평선이 뭉치고 변동폭이 줄어든 자리에서
    5일선을 뚫는 건 "수렴 후 발산" 의 시작이다. 흩어진 이평선 사이를 오르내리는 돌파와는
    질이 다르다.

    각 지표를 후보 집단 내 백분위로 환산한 뒤 가중합한다. 절대값이 아니라
    상대순위를 쓰기 때문에 지표 간 단위 차이에 영향받지 않는다.
    """
    if not candidates:
        return []

    w = config.RANK_WEIGHTS
    metrics = {
        "squeeze": _percentile_ranks([-c.ma_squeeze for c in candidates]),
        "contraction": _percentile_ranks([-c.vol_contraction for c in candidates]),
        "depth": _percentile_ranks([-c.prev_disparity for c in candidates]),
        "thrust": _percentile_ranks([c.thrust for c in candidates]),
        "trend": _percentile_ranks([c.ma_spread for c in candidates]),
        "value": _percentile_ranks([c.avg_value for c in candidates]),
    }

    for i, c in enumerate(candidates):
        detail = {k: round(metrics[k][i], 3) for k in metrics}
        c.score_detail = detail
        c.score = round(sum(detail[k] * w.get(k, 0.0) for k in detail), 4)

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def describe(c: Candidate) -> str:
    """텔레그램/로그용 한 줄 설명."""
    return (
        f"{c.name}({c.ticker}) {c.price:,.0f}원 {c.change_pct:+.2f}%\n"
        f"  5일선 {c.breakout_price:,.0f} 돌파 (+{c.thrust:.2f}%) | "
        f"전일 이격도 {c.prev_disparity:.1f} | {c.below_days}일 체류\n"
        f"  수렴 {c.ma_squeeze:.1f}% | 변동폭 ATR5/20 {c.vol_contraction:.2f} | "
        f"20/60 이격 {c.ma_spread:+.1f}% | 거래대금 {c.avg_value / 1e8:,.0f}억 | 점수 {c.score:.3f}"
    )


def to_dict(c: Candidate) -> dict:
    """상태 저장용 직렬화. 장중 재진입 캐시에 쓴다."""
    return asdict(c)


def from_dict(d: dict) -> Candidate:
    return Candidate(**d)
