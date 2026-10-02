"""
core/contest.py — 대회 모드(STRATEGY=contest) 계산부. 주문·상태·알림은 jobs/contest.py 가 한다.

목표: 한 달 안에 +CT_LOCK_PCT% 를 한 번 찍을 확률을 최대로 (평균 수익이 아니다). 근거 backtest/results/contest_tail_20261002.md.

규칙
  대상    코스피 시총 CT_KOSPI_N + 코스닥 시총 CT_KOSDAQ_N 보통주 (관리·정지·경고 제외), 주가 1,000원↑
  순위    CT_LOOKBACK 거래일 수익률(오늘 현재가 / N일 전 종가 − 1)이 대상 상위 CT_TOP_PCT% 안인 종목 중 높은 순 CT_SLOTS 개
          제외: 오늘 상한가, 최근 20일에 하한가(−29.5%↓)가 있었던 종목, 일봉이 부족한 종목
  매수    장 마감 동시호가 (15:20, 현재가 + CT_BUY_TICKS 틱 지정가, 상한가 이내)
  매도    ① 손절: 종가 ≤ 매수가 × (1 − CT_STOP_PCT%)      ② 추적: 종가 ≤ 보유 중 최고 종가 × (1 − CT_TRAIL_PCT%)
          ③ 보유 CT_HOLD_DAYS 거래일 경과                 → 전부 다음 날 장 시작 동시호가에 매도 (하한가 지정가 → 시가 체결)
  목표 락  마감 뒤 순자산이 월초 기준 × (1 + CT_LOCK_PCT%) 이상이면 다음 날 시가에 전량 매도, 월말까지 현금
  폭락 전환  대상 평균 등락이 −CT_CRASH_MKT_PCT% 이하이고 직전 20일 시장 일간 변동성 × CT_CRASH_SIGMA 이하면,
            보유를 종가에 팔고 그날 −CT_CRASH_DROP_PCT% 이하 빠진 종목(하한가 · 거래대금 20일 평균 CT_CRASH_VR_MAX 배↑ ·
            종목 낙폭이 시장 낙폭의 CT_CRASH_IDIO_MULT 배↑ 제외) 중 가장 많이 빠진 CT_SLOTS 개로 전환, CT_CRASH_HOLD 거래일 뒤 시가 매도
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import config


@dataclass
class Snap:
    ticker: str
    name: str
    market: str
    price: float                 # 지금 가격 (15:05 현재가 또는 종가)
    prev_close: float
    change_pct: float            # 오늘 등락률 (%)
    upper_limit: float
    lower_limit: float
    value: float                 # 오늘 거래대금
    closes: list = field(default_factory=list)     # 어제까지 종가 (오름차순)
    values: list = field(default_factory=list)     # 어제까지 거래대금 (오름차순)

    @property
    def ret_n(self) -> float | None:
        n = config.CT_LOOKBACK
        if len(self.closes) < n or self.closes[-n] <= 0 or self.price <= 0:
            return None
        return self.price / self.closes[-n] - 1

    @property
    def limit_up(self) -> bool:
        return self.upper_limit > 0 and self.price >= self.upper_limit

    @property
    def limit_down(self) -> bool:
        return self.lower_limit > 0 and self.price <= self.lower_limit

    @property
    def had_limit_down_20(self) -> bool:
        c = self.closes[-21:]
        return any(c[i] > 0 and c[i + 1] / c[i] - 1 <= -0.295 for i in range(len(c) - 1))

    @property
    def val_ratio(self) -> float | None:
        v = [x for x in self.values[-20:] if x > 0]
        return (self.value / (sum(v) / len(v))) if v else None


def split_universe(stocks: list[dict], kospi_n: int, kosdaq_n: int) -> list[dict]:
    """시총순 보통주 목록(market 포함)에서 코스피 상위 kospi_n + 코스닥 상위 kosdaq_n."""
    out, n = [], {"KOSPI": 0, "KOSDAQ": 0}
    lim = {"KOSPI": kospi_n, "KOSDAQ": kosdaq_n}
    for s in sorted(stocks, key=lambda s: -float(s.get("marcap", 0) or 0)):
        m = s.get("market", "KOSPI")
        if n.get(m, 0) < lim.get(m, 0):
            n[m] += 1
            out.append(dict(s, rank=n[m]))
    return out


def market_history(snaps: list[Snap], days: int = 20) -> list[float]:
    """직전 days 일의 시장(대상 평균) 일간 등락률. 어제까지 종가로 계산."""
    out = []
    for k in range(days, 0, -1):
        rs = []
        for s in snaps:
            c = s.closes
            if len(c) > k and c[-k - 1] > 0 and c[-k] > 0:
                rs.append(max(-0.3, min(0.3, c[-k] / c[-k - 1] - 1)))
        if rs:
            out.append(sum(rs) / len(rs))
    return out


def market_today(snaps: list[Snap]) -> tuple[float, int]:
    xs = [max(-0.3, min(0.3, s.change_pct / 100)) for s in snaps if s.price > 0 and s.prev_close > 0]
    return (sum(xs) / len(xs), len(xs)) if xs else (0.0, 0)


def crash_signal(snaps: list[Snap]) -> tuple[bool, dict]:
    """T5: 시장 −CT_CRASH_MKT_PCT% 이하 AND 직전 20일 변동성 × CT_CRASH_SIGMA 이하."""
    mkt, n = market_today(snaps)
    hist = market_history(snaps, 20)
    sigma = (sum(x * x for x in hist) / len(hist)) ** 0.5 if len(hist) >= 10 else None
    if sigma is not None:
        mean = sum(hist) / len(hist)
        sigma = (sum((x - mean) ** 2 for x in hist) / len(hist)) ** 0.5
    on = (config.CT_CRASH_ENABLED and n >= config.CT_CRASH_MIN_N and sigma is not None
          and mkt <= -config.CT_CRASH_MKT_PCT / 100 and mkt <= -config.CT_CRASH_SIGMA * sigma)
    return on, {"mkt": mkt, "n": n, "sigma": sigma, "threshold": (-config.CT_CRASH_SIGMA * sigma) if sigma else None}


def eligible(s: Snap) -> str:
    """빈 문자열이면 모멘텀 후보. 아니면 탈락 사유."""
    if s.price < 1000:
        return "저가주"
    if s.limit_up:
        return "상한가"
    if s.ret_n is None:
        return "일봉부족"
    if s.had_limit_down_20:
        return "하한가이력"
    return ""


def rank_momentum(snaps: list[Snap]) -> tuple[list[Snap], dict]:
    """CT_LOOKBACK 일 수익률 상위 CT_TOP_PCT% 안에서 높은 순. 반환 (후보 목록, 통계)."""
    ok, rejects = [], {}
    for s in snaps:
        why = eligible(s)
        if why:
            rejects[why] = rejects.get(why, 0) + 1
        else:
            ok.append(s)
    ok.sort(key=lambda s: (-s.ret_n, s.ticker))
    top = max(1, math.ceil(len(ok) * config.CT_TOP_PCT / 100)) if ok else 0
    cands = ok[:top]
    cut = cands[-1].ret_n if cands else None
    return cands, {"eligible": len(ok), "top": top, "cutoff": cut, "rejects": rejects}


def crash_candidates(snaps: list[Snap], mkt: float) -> tuple[list[Snap], dict]:
    """폭락일 급락주: −CT_CRASH_DROP_PCT% 이하, 하한가 · 거래 터짐 · 혼자 유독 빠짐 · 하한가 이력 제외, 더 빠진 순."""
    out, rejects = [], {}

    def rej(k):
        rejects[k] = rejects.get(k, 0) + 1

    for s in snaps:
        r = s.change_pct / 100
        if r > -config.CT_CRASH_DROP_PCT / 100:
            continue
        if s.price < 1000:
            rej("저가주"); continue
        if s.limit_down:
            rej("하한가"); continue
        if s.had_limit_down_20:
            rej("하한가이력"); continue
        vr = s.val_ratio
        if vr is not None and vr >= config.CT_CRASH_VR_MAX:
            rej("거래폭발"); continue
        if mkt < 0 and r <= mkt * config.CT_CRASH_IDIO_MULT:
            rej("종목악재의심"); continue
        out.append(s)
    out.sort(key=lambda s: (s.change_pct, s.ticker))
    return out, {"rejects": rejects}


def exit_reasons(pos: dict, close: float, today_idx: int) -> list[str]:
    """마감 뒤 판정. pos: entry_price, peak_close, entry_idx(거래일 번호), kind. 반환 사유 목록 (비면 보유)."""
    out = []
    entry = float(pos.get("entry_price", 0) or 0)
    peak = max(float(pos.get("peak_close", 0) or 0), close)
    held = today_idx - int(pos.get("entry_idx", today_idx))
    if pos.get("kind") == "crash":
        if held >= config.CT_CRASH_HOLD:
            out.append(f"폭락 전환 {config.CT_CRASH_HOLD}일 만료")
        return out
    if pos.get("kind") == "filler":
        out.append("조건용 1주 매도")
        return out
    if entry > 0 and close <= entry * (1 - config.CT_STOP_PCT / 100):
        out.append(f"손절 −{config.CT_STOP_PCT:g}%")
    if peak > 0 and close <= peak * (1 - config.CT_TRAIL_PCT / 100):
        out.append(f"추적 −{config.CT_TRAIL_PCT:g}% (고점 {peak:,.0f})")
    if held >= config.CT_HOLD_DAYS:
        out.append(f"보유 {config.CT_HOLD_DAYS}일 만료")
    return out


def lock_hit(nav: float, anchor: float) -> bool:
    return anchor > 0 and nav >= anchor * (1 + config.CT_LOCK_PCT / 100)


def describe(s: Snap) -> str:
    r = s.ret_n
    return (f"{s.name}({s.ticker}) {s.price:,.0f}원 | {config.CT_LOOKBACK}일 {r * 100:+.1f}% · 오늘 {s.change_pct:+.1f}%"
            if r is not None else f"{s.name}({s.ticker}) {s.price:,.0f}원 | 오늘 {s.change_pct:+.1f}%")
