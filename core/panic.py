"""
core/panic.py — 시장 급락일 과매도주 매수 (패닉 모드) 계산부. 주문은 jobs/panic.py 가 한다.

규칙 (backtest: 1억 · 5종목 · 2011~2019 봇 대비 +1.4%p/월, 2020~ +1.7%p/월 — 결과를 여러 번 본 뒤 고른 설정)
  급락일   장 마감 뒤, 시총 상위 PANIC_UNIVERSE_TOP 중 20일 평균 거래대금 PANIC_MKT_MIN_VALUE 이상 종목의
           오늘 등락률(±30% 자름) 단순 평균이 −PANIC_MKT_DROP_PCT% 이하
           (백테스트 기준 '거래대금 30억↑ 전 종목 평균' 과 상관 0.998, 급락일 61일 중 60일 일치)
  후보     시총 PANIC_PICK_TOP 위 안 · 주가 1,000원↑ · 20일 평균 거래대금(오늘 제외) PANIC_PICK_MIN_VALUE 이상
           · 오늘 상한가 아님 · 종가 위치(IBS) < PANIC_IBS_MAX · 오늘 거래대금 < 20일 평균 × PANIC_VR_MAX
  순위     최근 5거래일 수익률이 가장 나쁜 순
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import config


@dataclass
class Snap:
    ticker: str
    name: str
    market: str
    rank: int              # 시총 순위 (1부터)
    price: float           # 오늘 종가
    ret1: float            # 오늘 등락률
    ret5: float            # 5거래일 수익률
    val20: float           # 20일 평균 거래대금 (오늘 제외)
    ibs: float             # (종가 − 저가) / (고가 − 저가)
    vr: float              # 오늘 거래대금 / val20
    limit_up: bool

    def to_dict(self) -> dict:
        return asdict(self)


def snapshot(ticker: str, name: str, market: str, rank: int, candles: list[dict], today: str) -> Snap | None:
    """오늘 일봉이 들어 있는 일봉 목록(날짜 오름차순)으로 계산. 오늘 일봉이 없거나 너무 짧으면 None."""
    if len(candles) < 26 or candles[-1]["date"] != today:
        return None
    c = [float(x["close"]) for x in candles]
    t = candles[-1]
    prev = c[-2]
    if prev <= 0 or c[-6] <= 0:
        return None
    hist = [float(x.get("value", 0) or 0) for x in candles[-21:-1]]
    val20 = sum(hist) / len(hist) if hist else 0.0
    h, l = float(t["high"]), float(t["low"])
    ibs = (c[-1] - l) / (h - l) if h > l else 0.5
    ret1 = c[-1] / prev - 1
    return Snap(ticker=ticker, name=name, market=market, rank=rank, price=c[-1], ret1=ret1, ret5=c[-1] / c[-6] - 1,
                val20=val20, ibs=ibs, vr=(float(t.get("value", 0) or 0) / val20) if val20 > 0 else 0.0,
                limit_up=ret1 >= 0.295)


def market_drop(snaps: list[Snap]) -> tuple[float, int]:
    """거래대금 조건을 넘는 종목의 오늘 등락률 단순 평균 (±30% 자름) 과 종목 수."""
    xs = [max(-0.3, min(0.3, s.ret1)) for s in snaps if s.val20 >= config.PANIC_MKT_MIN_VALUE]
    return (sum(xs) / len(xs), len(xs)) if xs else (0.0, 0)


def is_panic(avg: float, n: int) -> bool:
    return n >= 100 and avg <= -config.PANIC_MKT_DROP_PCT / 100


def eligible(s: Snap) -> str:
    """빈 문자열이면 후보. 아니면 탈락 사유."""
    if s.rank > config.PANIC_PICK_TOP:
        return "시총순위"
    if s.price < 1000:
        return "저가주"
    if s.val20 < config.PANIC_PICK_MIN_VALUE:
        return "거래대금"
    if s.limit_up:
        return "상한가"
    if s.ibs >= config.PANIC_IBS_MAX:
        return "고가권마감"
    if s.vr >= config.PANIC_VR_MAX:
        return "거래대금급증"
    return ""


def pick(snaps: list[Snap], n: int) -> list[Snap]:
    """후보를 5일 수익률이 나쁜 순으로 n 개."""
    ok = [s for s in snaps if not eligible(s)]
    ok.sort(key=lambda s: (s.ret5, s.ticker))
    return ok[:n]


def describe(s: Snap | dict) -> str:
    d = s if isinstance(s, dict) else s.to_dict()
    return (f"{d['name']}({d['ticker']}) {d['price']:,.0f}원 | 5일 {d['ret5'] * 100:+.1f}% · 오늘 {d['ret1'] * 100:+.1f}% "
            f"| 시총 {d['rank']}위 · 거래대금 {d['val20'] / 1e8:,.0f}억")
