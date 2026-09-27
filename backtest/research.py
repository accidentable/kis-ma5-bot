"""
backtest/research.py — 대안 전략 탐색 (표본 내 선택 → 표본 외 검증)

  python -m backtest.research            전체 실행 + backtest/results/research_<날짜>.md

실전 봇 코드와 독립이다. snapshot.json.gz 만 읽는다 (한투 키 불필요).

과적합을 막으려고 순서를 고정했다.
  1. 전략 계열과 파라미터 격자를 먼저 정한다 (아래 FAMILIES). 결과를 보고 격자를 늘리지 않는다.
  2. 표본 내(IS) 구간에서만 계열별 최선 설정을 고른다. 기준은 5슬롯 포트폴리오 샤프.
  3. 고른 설정만 표본 외(OOS) 구간에서 한 번 돌려 본다. OOS 로 다시 고르지 않는다.
  4. 같은 설정을 다른 유니버스·슬롯 수·비용 가정에 돌려 결론이 버티는지 본다.

체결 모델 (일봉으로 실제로 구현 가능한 것만)
  진입   신호는 d일 종가까지의 데이터로 계산, d+1일 시가(장전 동시호가)에 산다
  청산   종가 판정은 그날 종가(장마감 동시호가)에 판다
         손절선은 장중 저가가 닿으면 손절가에, 시가가 이미 아래면 시가에 판다
         로테이션 교체는 다음 날 시가에 판다
  비용   수수료 편도 0.015%, 매도세 0.20%, 동시호가 체결 슬리피지 편도 0.05%, 장중 손절 슬리피지 0.2%
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Callable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = os.path.join(ROOT, "backtest", "snapshot.json.gz")
OUT = os.path.join(ROOT, "backtest", "results")

# ── 구간 ─────────────────────────────────────────────────────
IS_START, IS_END = "20231004", "20250630"     # 표본 내: 전략 선택에만 쓴다
OOS_START = "20250701"                         # 표본 외: 선택 뒤 한 번만 본다

# ── 비용 ─────────────────────────────────────────────────────
COST = {"fee": 0.00015, "tax": 0.0020, "slip_auction": 0.0005, "slip_stop": 0.0020}

MIN_VALUE = 5e9    # 20일 평균 거래대금 50억 미만은 거른다


# ════════════════════════════════════════════════════════════
# 데이터 + 지표
# ════════════════════════════════════════════════════════════
@dataclass
class Series:
    t: str
    d: list
    o: list
    h: list
    l: list
    c: list
    val: list
    idx: dict = field(default_factory=dict)
    ind: dict = field(default_factory=dict)


def _sma(x: list, n: int) -> list:
    out = [math.nan] * len(x)
    s = 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def _rsi(c: list, n: int) -> list:
    out = [math.nan] * len(c)
    up = dn = 0.0
    for i in range(1, len(c)):
        ch = c[i] - c[i - 1]
        g, lo = max(ch, 0.0), max(-ch, 0.0)
        if i <= n:
            up += g / n
            dn += lo / n
        else:
            up = (up * (n - 1) + g) / n
            dn = (dn * (n - 1) + lo) / n
        if i >= n:
            out[i] = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
    return out


def _atr(h: list, l: list, c: list, n: int = 14) -> list:
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    return _sma(tr, n)


def _roll(x: list, n: int, fn: Callable) -> list:
    """직전 n개 (오늘 제외) 의 fn. 돌파 판정용."""
    return [math.nan if i < n else fn(x[i - n:i]) for i in range(len(x))]


def _ret(c: list, n: int, skip: int = 0) -> list:
    return [math.nan if i < n else c[i - skip] / c[i - n] - 1 for i in range(len(c))]


def load() -> tuple[dict, dict]:
    with gzip.open(SNAPSHOT, "rt", encoding="utf-8") as f:
        snap = json.load(f)
    univ = {k: {s["ticker"] for s in v} for k, v in snap["universes"].items()}
    names = {s["ticker"]: s["name"] for v in snap["universes"].values() for s in v}
    data = {}
    for t, b in snap["bars"].items():
        s = Series(t, [x["date"] for x in b], [x["open"] for x in b], [x["high"] for x in b],
                   [x["low"] for x in b], [x["close"] for x in b], [x.get("value") or 0.0 for x in b])
        s.idx = {d: i for i, d in enumerate(s.d)}
        c = s.c
        s.ind = {
            "ma5": _sma(c, 5), "ma20": _sma(c, 20), "ma60": _sma(c, 60), "ma120": _sma(c, 120),
            "rsi2": _rsi(c, 2), "atr": _atr(s.h, s.l, c),
            "hh20": _roll(s.h, 20, max), "hh55": _roll(s.h, 55, max),
            "ll10": _roll(s.l, 10, min), "ll20": _roll(s.l, 20, min),
            "ret5": _ret(c, 5), "mom60": _ret(c, 60, 5), "mom120": _ret(c, 120, 5),
            "val20": _sma(s.val, 20),
        }
        data[t] = s
    univ["KOSPI200+KOSDAQ150"] = univ["KOSPI200"] | univ["KOSDAQ150"]
    return data, {"univ": univ, "names": names}


def calendar(data: dict) -> list:
    return sorted({d for s in data.values() for d in s.d})


def ew_index(data: dict, members: set, cal: list) -> dict:
    """유니버스 동일가중 지수 (일간 종가 수익률 평균의 누적). 시장 국면 필터용."""
    lvl, out = 1.0, {}
    for k, d in enumerate(cal):
        rs = []
        if k:
            p = cal[k - 1]
            for t in members:
                s = data.get(t)
                if s and d in s.idx and p in s.idx:
                    rs.append(s.c[s.idx[d]] / s.c[s.idx[p]] - 1)
        lvl *= 1 + (fmean(rs) if rs else 0.0)
        out[d] = lvl
    return out


# ════════════════════════════════════════════════════════════
# 전략 — candidates(d) 는 d일 종가 기준 신규 매수 후보, check_exit 는 보유 중 청산 판정
# ════════════════════════════════════════════════════════════
@dataclass
class Pos:
    t: str
    entry_d: str
    entry_px: float     # 비용 전 체결가
    cost: float         # 투입금 (수수료 포함)
    qty: float
    days: int = 0
    peak: float = 0.0
    stop: float = 0.0


class Strategy:
    name = "base"

    def __init__(self, ctx: dict, members: set, **p):
        # 정렬된 목록으로 들고 있어야 동점 후보의 순서가 실행마다 같다 (set 순서는 PYTHONHASHSEED 에 따라 바뀐다)
        self.ctx, self.members, self.p = ctx, sorted(members), p
        self.data = ctx["data"]
        self.regime_on = True

    def label(self) -> str:
        return f"{self.name}(" + ", ".join(f"{k}={v}" for k, v in self.p.items()) + ")"

    def begin_day(self, d: str) -> None:
        if self.p.get("regime"):
            ix, ma = self.ctx["ew"], self.ctx["ew_ma"]
            self.regime_on = ix[d] > ma.get(d, 0)

    def _liquid(self, s: Series, i: int) -> bool:
        v = s.ind["val20"][i]
        return v == v and v >= MIN_VALUE

    def candidates(self, d: str) -> list:
        return []

    def init_pos(self, pos: Pos, s: Series, i: int) -> None:
        pass

    # 반환: None | ("stop", 가격, 사유) | ("close", 사유) | ("next_open", 사유)
    def check_exit(self, pos: Pos, s: Series, i: int):
        return None


class MA5Cross(Strategy):
    """원래 아이디어의 깔끔한 버전: 종가가 5일선 아래→위로 올라오면 다음 날 시가 매수, 종가 5일선 이탈 청산."""
    name = "MA5교차"

    def candidates(self, d):
        if not self.regime_on:
            return []
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i):
                continue
            m5 = s.ind["ma5"]
            if not (s.c[i] > m5[i] and s.c[i - 1] <= m5[i - 1]):
                continue
            tf = self.p["trend"]
            if tf and not s.c[i] > s.ind[tf][i]:
                continue
            out.append((t, -(s.c[i] / m5[i] - 1)))   # 5일선에 가까울수록 먼저 (덜 추격)
        return sorted(out, key=lambda x: -x[1])

    def check_exit(self, pos, s, i):
        if s.c[i] < s.ind["ma5"][i]:
            return ("close", "5일선 이탈")
        if pos.days >= self.p["hold"]:
            return ("close", "만료")
        return None


class Reversal(Strategy):
    """단기 반전: 상승 추세(장기선 위) 종목이 RSI(2) 과매도로 밀리면 산다. 5일선 회복 또는 N일에 판다."""
    name = "단기반전"

    def candidates(self, d):
        if not self.regime_on:
            return []
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i):
                continue
            if not s.c[i] > s.ind[self.p["trend"]][i]:
                continue
            r = s.ind["rsi2"][i]
            if r < self.p["rsi"]:
                out.append((t, -r))
        return sorted(out, key=lambda x: -x[1])

    def check_exit(self, pos, s, i):
        if pos.days >= 1 and s.c[i] > s.ind["ma5"][i]:
            return ("close", "5일선 회복")
        if pos.days >= self.p["hold"]:
            return ("close", "만료")
        return None


class Breakout(Strategy):
    """돌파 추세추종: 종가가 N일 신고가를 넘으면 산다. M일 신저가 이탈 또는 3ATR 추적손절에 판다."""
    name = "돌파추종"

    def candidates(self, d):
        if not self.regime_on:
            return []
        n = self.p["n"]
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i):
                continue
            hh = s.ind[f"hh{n}"][i]
            if not (s.c[i] > hh and s.c[i - 1] <= s.ind[f"hh{n}"][i - 1]):
                continue
            if not s.c[i] > s.ind["ma120"][i]:
                continue
            out.append((t, s.ind["mom60"][i]))
        return sorted(out, key=lambda x: -x[1])

    def init_pos(self, pos, s, i):
        pos.peak = pos.entry_px
        pos.stop = pos.entry_px - 3 * s.ind["atr"][i - 1]

    def check_exit(self, pos, s, i):
        if s.o[i] <= pos.stop:
            return ("stop", s.o[i], "추적손절(갭)")
        if s.l[i] <= pos.stop:
            return ("stop", pos.stop, "추적손절")
        pos.peak = max(pos.peak, s.c[i])
        pos.stop = max(pos.stop, pos.peak - 3 * s.ind["atr"][i])
        m = self.p["m"]
        if s.c[i] < s.ind[f"ll{m}"][i]:
            return ("close", f"{m}일 신저가")
        return None


class Momentum(Strategy):
    """모멘텀 로테이션: R거래일마다 L일 수익률(최근 5일 제외) 상위 K개를 들고, 순위 2K 밖으로 밀리면 교체."""
    name = "모멘텀로테이션"

    def __init__(self, ctx, members, **p):
        super().__init__(ctx, members, **p)
        self.k = 0          # 포트폴리오 슬롯 수 (run 이 넣는다)
        self.n_day = 0
        self.keep: set = set()
        self.rebal = False

    def begin_day(self, d):
        super().begin_day(d)
        self.rebal = self.n_day % self.p["r"] == 0
        self.n_day += 1

    def _ranked(self, d):
        key = f"mom{self.p['L']}"
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i):
                continue
            m = s.ind[key][i]
            if s.c[i] > getattr(self, "cap", math.inf):
                continue   # 주가 상한은 순위를 매기기 전에 건다 (상위 K개를 고른 뒤 거르면 빈손이 된다)
            if m == m and m > 0 and s.c[i] > s.ind["ma60"][i]:
                out.append((t, m))
        return sorted(out, key=lambda x: -x[1])

    def candidates(self, d):
        if not self.rebal:
            return []
        if not self.regime_on:
            self.keep = set()
            return []
        r = self._ranked(d)
        self.keep = {t for t, _ in r[: 2 * self.k]}
        return r[: self.k]

    def check_exit(self, pos, s, i):
        # candidates 가 같은 날 먼저 불리도록 run 에서 순서를 맞춘다
        if self.rebal and pos.t not in self.keep:
            return ("next_open", "순위 밖" if self.regime_on else "국면 off")
        return None


# ════════════════════════════════════════════════════════════
# 포트폴리오 시뮬레이션
# ════════════════════════════════════════════════════════════
@dataclass
class Trade:
    t: str
    entry_d: str
    exit_d: str
    net: float
    days: int
    reason: str


def run(strat: Strategy, cal: list, start: str, end: str, k: int, cost: dict = COST) -> dict:
    data = strat.data
    if isinstance(strat, Momentum):
        strat.k = k
    days = [d for d in cal if start <= d <= end]
    cash, pos = 1.0, {}
    pending: list = []            # 다음 날 시가 매수 [(t, score)]
    pend_exit: dict = {}          # 다음 날 시가 매도 {t: 사유}
    trades, curve = [], []
    buy_mult = 1 + cost["fee"] + cost["slip_auction"]

    def sell(p: Pos, px: float, slip: float, d: str, reason: str) -> float:
        net_px = px * (1 - slip) * (1 - cost["fee"] - cost["tax"])
        proceeds = p.qty * net_px
        trades.append(Trade(p.t, p.entry_d, d, proceeds / p.cost - 1, p.days, reason))
        return proceeds

    last_eq = 1.0
    for d in days:
        strat.begin_day(d)
        # ① 시가: 매도 먼저, 그다음 매수
        for t, why in list(pend_exit.items()):
            s = data[t]
            i = s.idx.get(d)
            if i is None:
                continue
            cash += sell(pos.pop(t), s.o[i], cost["slip_auction"], d, why)
            del pend_exit[t]
        if pending:
            for t, _ in pending:
                if len(pos) >= k or t in pos:
                    continue
                s = data[t]
                i = s.idx.get(d)
                if i is None:
                    continue
                alloc = min(cash, last_eq / k)
                if alloc <= 1e-9:
                    break
                px = s.o[i]
                qty = alloc / (px * buy_mult)
                p = Pos(t, d, px, alloc, qty)
                strat.init_pos(p, s, i)
                pos[t] = p
                cash -= alloc
            pending = []
        # ② 장중·종가 청산
        cands = strat.candidates(d)      # 로테이션은 여기서 keep 을 정한다
        for t in list(pos):
            p = pos[t]
            s = data[t]
            i = s.idx.get(d)
            if i is None:
                continue
            p.days += 1
            ex = strat.check_exit(p, s, i)
            if ex is None:
                continue
            if ex[0] == "stop":
                cash += sell(pos.pop(t), ex[1], cost["slip_stop"], d, ex[2])
            elif ex[0] == "close":
                cash += sell(pos.pop(t), s.c[i], cost["slip_auction"], d, ex[1])
            else:
                pend_exit[t] = ex[1]
        # ③ 평가
        eq = cash
        for t, p in pos.items():
            s = data[t]
            i = s.idx.get(d)
            px = s.c[i] if i is not None else s.c[max(j for j, x in enumerate(s.d) if x <= d)]
            eq += p.qty * px
        curve.append((d, eq))
        last_eq = eq
        # ④ 다음 날 매수 후보
        free = k - len(pos) + len(pend_exit)
        if free > 0 and cands:
            pending = [(t, sc) for t, sc in cands if t not in pos or t in pend_exit][: free]

    # 기간 끝 미청산은 마지막 종가로 평가만 한다 (거래 목록엔 넣는다)
    for t, p in pos.items():
        s = data[t]
        i = max(j for j, x in enumerate(s.d) if x <= days[-1])
        trades.append(Trade(t, p.entry_d, s.d[i], p.qty * s.c[i] * (1 - cost["fee"] - cost["tax"]) / p.cost - 1,
                            p.days, "기간 끝"))
    return {"trades": trades, "curve": curve}


def stats(r: dict) -> dict:
    nets = [t.net for t in r["trades"]]
    curve = r["curve"]
    eqs = [v for _, v in curve]
    rets = [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs))]
    peak, mdd = 1.0, 0.0
    for v in eqs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    years = len(eqs) / 250
    sd = pstdev(rets) if len(rets) > 1 else 0
    n = len(nets)
    tstat = (fmean(nets) / (pstdev(nets) / math.sqrt(n))) if n > 2 and pstdev(nets) > 0 else 0
    return {
        "n": n, "exp": fmean(nets) if nets else 0, "win": sum(x > 0 for x in nets) / n if n else 0,
        "t": tstat, "total": eqs[-1] - 1 if eqs else 0,
        "cagr": eqs[-1] ** (1 / years) - 1 if eqs and eqs[-1] > 0 else -1,
        "mdd": mdd, "sharpe": fmean(rets) / sd * math.sqrt(250) if sd > 0 else 0,
        "hold": fmean([t.days for t in r["trades"]]) if n else 0,
    }


def bench(data: dict, members: set, start: str, end: str) -> dict:
    """동일가중 보유 (기간 첫날 시가 매수 → 끝날 종가). 매수 비용·매도세 반영."""
    cal = [d for d in calendar(data) if start <= d <= end]
    got = [t for t in members if t in data and cal[0] in data[t].idx and cal[-1] in data[t].idx]
    buy = {t: data[t].o[data[t].idx[cal[0]]] * (1 + COST["fee"]) for t in got}
    curve = []
    for d in cal:
        vals = []
        for t in got:
            s = data[t]
            i = s.idx.get(d)
            if i is None:
                i = max(j for j, x in enumerate(s.d) if x <= d)
            vals.append(s.c[i] / buy[t])
        curve.append((d, fmean(vals)))
    last = curve[-1][1] * (1 - COST["fee"] - COST["tax"])
    curve[-1] = (curve[-1][0], last)
    return stats({"trades": [], "curve": curve})


# ════════════════════════════════════════════════════════════
# 실험
# ════════════════════════════════════════════════════════════
def grid() -> list:
    """사전 등록한 격자. 결과를 보고 늘리지 않는다."""
    out = []
    for trend in (None, "ma60"):
        for hold in (3, 10):
            for reg in (False, True):
                out.append((MA5Cross, {"trend": trend, "hold": hold, "regime": reg}))
    for tr in ("ma60", "ma120"):
        for rsi in (5, 10):
            for hold in (5, 10):
                for reg in (False, True):
                    out.append((Reversal, {"trend": tr, "rsi": rsi, "hold": hold, "regime": reg}))
    for n in (20, 55):
        for m in (10, 20):
            for reg in (False, True):
                out.append((Breakout, {"n": n, "m": m, "regime": reg}))
    for L in (60, 120):
        for r in (5, 20):
            for reg in (False, True):
                out.append((Momentum, {"L": L, "r": r, "regime": reg}))
    return out


def make_ctx(data: dict, members: set, cal: list) -> dict:
    ew = ew_index(data, members, cal)
    vals = [ew[d] for d in cal]
    ma = _sma(vals, 100)
    return {"data": data, "ew": ew, "ew_ma": {d: m for d, m in zip(cal, ma) if m == m}}


def pct(x: float, dg: int = 1) -> str:
    return f"{x * 100:+.{dg}f}%"


class HoldAll(Strategy):
    """사후 추가 (선택 대상 아님): 유니버스 전체 동일가중 보유. regime=True 면 동일가중 지수가 100일선 아래일 때 현금."""
    name = "전체보유"

    def __init__(self, ctx, members, **p):
        super().__init__(ctx, members, **p)
        self.was_on = None

    def candidates(self, d):
        if not self.regime_on:
            return []
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is not None and self._liquid(s, i):
                out.append((t, 0.0))
        return out

    def check_exit(self, pos, s, i):
        return None if self.regime_on else ("next_open", "국면 off")


class MaxPrice(Strategy):
    """다른 전략을 감싸 주가 상한을 건다 (소액 계좌에서 1주도 못 사는 종목 제외)."""

    def __init__(self, inner: Strategy, cap: float):
        self.inner, self.cap = inner, cap
        self.data, self.p, self.name = inner.data, dict(inner.p, cap=cap), inner.name
        inner.cap = cap
        self.intraday = getattr(inner, "intraday", False)
        self.entry_at_close = getattr(inner, "entry_at_close", False)

    def begin_day(self, d):
        self.inner.begin_day(d)

    def intraday_entries(self, d):
        return [x for x in self.inner.intraday_entries(d) if x[1] <= self.cap]

    def candidates(self, d):
        return [(t, sc) for t, sc in self.inner.candidates(d) if self.data[t].c[self.data[t].idx[d]] <= self.cap]

    def init_pos(self, pos, s, i):
        self.inner.init_pos(pos, s, i)

    def check_exit(self, pos, s, i):
        return self.inner.check_exit(pos, s, i)


def extra(data: dict, U: dict, ctxs: dict, cal: list, end: str) -> list:
    """보조 점검. 선택은 이미 끝났고 여기서 설정을 바꾸지 않는다."""
    U0 = "KOSPI100"
    ctx, mem = ctxs[U0], U[U0]
    L = ["", "## ④ 보조 점검 (선택 뒤, 설정 변경 없음)", ""]

    # 엔진 검산: 전체보유를 포트폴리오 엔진으로 돌리면 bench 와 거의 같아야 한다
    b = bench(data, mem, IS_START, end)
    h = stats(run(HoldAll(ctx, mem, regime=False), cal, IS_START, end, 100))
    L += [f"- 엔진 검산 — 동일가중 보유: bench {pct(b['total'])} / 엔진(100슬롯, 첫날 시가 매수) {pct(h['total'])}", ""]

    # 로테이션 날짜 위상: 20일마다 교체면 어느 날 시작하느냐에 따라 결과가 갈린다
    L += ["### 모멘텀 로테이션 — 교체일 위상 20가지 (KOSPI100, L=60, r=20)", "",
          "| 슬롯 | 구간 | 누적 최소 / 중앙 / 최대 | CAGR 중앙 | MDD 중앙 | 샤프 중앙 | 동일가중 보유 누적 |", "|---|---|---|---|---|---|---|"]
    from statistics import median
    for k in (1, 5):
        for tag, a, z in (("전체", IS_START, end), ("OOS", OOS_START, end)):
            days = [d for d in cal if a <= d <= z]
            ms = []
            for off in range(20):
                ms.append(stats(run(Momentum(ctx, mem, L=60, r=20, regime=False), cal, days[off], z, k)))
            tot = sorted(m["total"] for m in ms)
            bb = bench(data, mem, a, z)
            L.append(f"| {k} | {tag} | {pct(tot[0])} / {pct(median(tot))} / {pct(tot[-1])} | {pct(median(m['cagr'] for m in ms))} | "
                     f"{pct(median(m['mdd'] for m in ms))} | {median(m['sharpe'] for m in ms):.2f} | {pct(bb['total'])} |")
            print("phase", k, tag, flush=True)

    # 반기별: 전략이 시장을 이긴 구간이 고르게 있나
    halves = []
    y0 = int(IS_START[:4])
    for y in range(y0, int(end[:4]) + 1):
        for a, z in ((f"{y}0101", f"{y}0630"), (f"{y}0701", f"{y}1231")):
            a, z = max(a, IS_START), min(z, end)
            if a < z and [d for d in cal if a <= d <= z]:
                halves.append((a, z))
    L += ["", "### 반기별 누적 (KOSPI100, 5슬롯, 반기마다 새로 시작)", "",
          "| 반기 | 동일가중 보유 | 모멘텀로테이션 | 돌파추종 | 단기반전 |", "|---|---|---|---|---|"]
    for a, z in halves:
        bb = bench(data, mem, a, z)
        mo = stats(run(Momentum(ctx, mem, L=60, r=20, regime=False), cal, a, z, 5))
        br = stats(run(Breakout(ctx, mem, n=55, m=10, regime=False), cal, a, z, 5))
        rv = stats(run(Reversal(ctx, mem, trend="ma120", rsi=5, hold=10, regime=False), cal, a, z, 5))
        L.append(f"| {a[:6]}~{z[:6]} | {pct(bb['total'])} | {pct(mo['total'])} | {pct(br['total'])} | {pct(rv['total'])} |")

    # 소액 계좌: 60만원 5슬롯이면 슬롯당 12만원
    L += ["", "### 소액 계좌 제약 — 주가 상한 (KOSPI100, 모멘텀 L=60 r=20)", "",
          "| 조건 | 슬롯 | 구간 | 거래 | 누적 | CAGR | MDD | 샤프 |", "|---|---|---|---|---|---|---|---|"]
    for cap, k in ((1e12, 5), (120_000, 5), (600_000, 1)):
        for tag, a, z in (("전체", IS_START, end), ("OOS", OOS_START, end)):
            st = MaxPrice(Momentum(ctx, mem, L=60, r=20, regime=False), cap)
            st.inner.k = k
            m = stats(run(st, cal, a, z, k))
            L.append(f"| {'상한 없음' if cap > 1e11 else f'{cap / 1e4:.0f}만원 이하'} | {k} | {tag} | {m['n']} | {pct(m['total'])} | {pct(m['cagr'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} |")

    # 사후 추가: 전체보유 + 국면 필터
    L += ["", "### 사후 추가 — 전체보유 + 국면 필터 (결과를 본 뒤 떠올린 것이라 선택 근거로 쓰지 않는다)", "",
          "| 설정 | 구간 | 누적 | CAGR | MDD | 샤프 |", "|---|---|---|---|---|---|"]
    for reg in (False, True):
        for tag, a, z in (("IS", IS_START, IS_END), ("OOS", OOS_START, end)):
            m = stats(run(HoldAll(ctx, mem, regime=reg), cal, a, z, 100))
            L.append(f"| 전체보유{' + 100일선 필터' if reg else ''} | {tag} | {pct(m['total'])} | {pct(m['cagr'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} |")
    return L


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    data, meta = load()
    cal = calendar(data)
    end = cal[-1]
    U = meta["univ"]
    ctxs = {u: make_ctx(data, U[u], cal) for u in ("KOSPI100", "KOSPI200", "KOSPI200+KOSDAQ150")}

    L = [f"# 대안 전략 탐색 — 표본 내 선택, 표본 외 검증", "",
         f"- 표본 내(IS) {IS_START} ~ {IS_END} · 표본 외(OOS) {OOS_START} ~ {end}",
         f"- 체결: d일 종가로 신호 → d+1일 시가 매수, 종가 판정은 당일 종가 매도. 비용 수수료 편도 0.015% + 매도세 0.20% + 동시호가 슬리피지 편도 0.05% (손절 0.2%)",
         f"- 선택 기준: IS 에서 KOSPI100·5슬롯 포트폴리오 샤프. 격자는 실행 전에 고정 ({len(grid())}개 설정)", ""]

    # ① IS 격자 (KOSPI100, 5슬롯)
    U0 = "KOSPI100"
    rows = []
    for cls, p in grid():
        st = cls(ctxs[U0], U[U0], **p)
        r5 = stats(run(st, cal, IS_START, IS_END, 5))
        rows.append((cls, p, st.label(), r5))
        print(f"IS {st.label():70s} n={r5['n']:4d} exp={pct(r5['exp'], 2)} sharpe={r5['sharpe']:.2f} cagr={pct(r5['cagr'])} mdd={pct(r5['mdd'])}", flush=True)

    b_is = bench(data, U[U0], IS_START, IS_END)
    b_oos = bench(data, U[U0], OOS_START, end)
    L += ["## ① 표본 내 격자 (KOSPI100, 5슬롯)", "",
          f"기준선 — 동일가중 보유: 누적 {pct(b_is['total'])}, CAGR {pct(b_is['cagr'])}, MDD {pct(b_is['mdd'])}, 샤프 {b_is['sharpe']:.2f}", "",
          "| 설정 | 거래 | 거래당 기대값 | t | 승률 | 누적 | CAGR | MDD | 샤프 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, _, lab, m in rows:
        L.append(f"| {lab} | {m['n']} | {pct(m['exp'], 2)} | {m['t']:.1f} | {m['win'] * 100:.0f}% | {pct(m['total'])} | {pct(m['cagr'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} |")

    # ② 계열별 IS 최선 → OOS 한 번
    best = {}
    for cls, p, lab, m in rows:
        if m["n"] < 30:
            continue
        if cls not in best or m["sharpe"] > best[cls][2]["sharpe"]:
            best[cls] = (p, lab, m)
    L += ["", "## ② 계열별 IS 최선 설정 → 표본 외 (KOSPI100)", "",
          f"기준선 OOS — 동일가중 보유: 누적 {pct(b_oos['total'])}, CAGR {pct(b_oos['cagr'])}, MDD {pct(b_oos['mdd'])}, 샤프 {b_oos['sharpe']:.2f}", "",
          "| 설정 | 슬롯 | 구간 | 거래 | 거래당 기대값 | t | 승률 | 누적 | CAGR | MDD | 샤프 | 평균 보유 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    oos_rows = {}
    for cls, (p, lab, _) in best.items():
        for k in (1, 5):
            for tag, a, z in (("IS", IS_START, IS_END), ("OOS", OOS_START, end)):
                m = stats(run(cls(ctxs[U0], U[U0], **p), cal, a, z, k))
                oos_rows[(cls, k, tag)] = m
                L.append(f"| {lab} | {k} | {tag} | {m['n']} | {pct(m['exp'], 2)} | {m['t']:.1f} | {m['win'] * 100:.0f}% | "
                         f"{pct(m['total'])} | {pct(m['cagr'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} | {m['hold']:.1f} |")
                print(f"{tag} k={k} {lab} {pct(m['exp'], 2)} cagr={pct(m['cagr'])} mdd={pct(m['mdd'])} sh={m['sharpe']:.2f}", flush=True)

    # ③ 강건성: 다른 유니버스, 비용 2배, 전체 구간
    L += ["", "## ③ 강건성 — 같은 설정을 다른 유니버스·비용에 (5슬롯, 전체 구간 " + IS_START + " ~ " + end + ")", "",
          "| 설정 | 유니버스 | 비용 | 거래 | 거래당 기대값 | 누적 | CAGR | MDD | 샤프 |", "|---|---|---|---|---|---|---|---|---|"]
    cost2 = dict(COST, slip_auction=COST["slip_auction"] * 4, slip_stop=COST["slip_stop"] * 2)
    for cls, (p, lab, _) in best.items():
        for u in ("KOSPI100", "KOSPI200", "KOSPI200+KOSDAQ150"):
            for cname, c in (("기본", COST), ("슬리피지↑", cost2)):
                if u != "KOSPI100" and cname != "기본":
                    continue
                m = stats(run(cls(ctxs[u], U[u], **p), cal, IS_START, end, 5, c))
                L.append(f"| {lab} | {u} | {cname} | {m['n']} | {pct(m['exp'], 2)} | {pct(m['total'])} | {pct(m['cagr'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} |")
        print("robust", lab, flush=True)
    for u in ("KOSPI100", "KOSPI200", "KOSPI200+KOSDAQ150"):
        b = bench(data, U[u], IS_START, end)
        L.append(f"| 동일가중 보유 | {u} | 기본 | - | - | {pct(b['total'])} | {pct(b['cagr'])} | {pct(b['mdd'])} | {b['sharpe']:.2f} |")

    L += extra(data, U, ctxs, cal, end)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"research_{end}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
