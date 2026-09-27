"""
backtest/lab/engine.py — 전종목(상장폐지 포함) 행렬 위에서 "한 달짜리 대회"를 흉내 내는 시뮬레이터

데이터: data/lab/market.npz (build_marcap.py). 날짜 × 종목 행렬, 수정주가 o/h/l/c, 원주가 raw, 거래대금 val, 시총 cap.

한 달 = 21거래일. 첫날 시가에 (전날 종가까지의 신호로) 사고, 21일째 종가에 전부 판다.
계좌 60만원, 정수 주식만 (원주가 기준). 슬롯 k 개면 슬롯당 60만/k 원 안에서 살 수 있는 만큼.

비용: 매수 수수료 0.015%, 매도 수수료 0.015% + 거래세 0.20% (2026년 기준을 전 기간에 적용),
      슬리피지 편도 max(0.05%, 0.5틱/가격). 장중 손절 체결은 여기에 0.2% 추가.
시가 상한가(전일 대비 +29.5% 이상)로 열린 날은 못 산다고 본다.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ = os.path.join(ROOT, "data", "lab", "market.npz")

CAPITAL = 600_000
FEE, TAX = 0.00015, 0.0020
MONTH = 21


def tick_size(p: float) -> float:
    if p < 2000: return 1
    if p < 5000: return 5
    if p < 20000: return 10
    if p < 50000: return 50
    if p < 200000: return 100
    if p < 500000: return 500
    return 1000


def slip(raw: float) -> float:
    return max(0.0005, 0.5 * tick_size(raw) / raw) if raw > 0 else 0.001


# ════════════════════════════════════════════════════════════
# 데이터 + 지표 (행렬)
# ════════════════════════════════════════════════════════════
def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    """열마다 이동평균. NaN 이 창 안에 있으면 NaN."""
    T = x.shape[0]
    out = np.full_like(x, np.nan)
    cs = np.nancumsum(np.nan_to_num(x), axis=0, dtype=np.float64)
    cnt = np.cumsum(~np.isnan(x), axis=0)
    s = cs[n - 1:] - np.vstack([np.zeros((1, x.shape[1])), cs[:-n]])
    k = cnt[n - 1:] - np.vstack([np.zeros((1, x.shape[1]), dtype=cnt.dtype), cnt[:-n]])
    out[n - 1:] = np.where(k == n, s / n, np.nan)
    return out


def _roll_max(x: np.ndarray, n: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full_like(x, np.nan)
    w = sliding_window_view(np.nan_to_num(x, nan=-np.inf), n, axis=0)
    m = w.max(axis=-1)
    m[np.isinf(m)] = np.nan
    out[n - 1:] = m
    return out


def _roll_min(x: np.ndarray, n: int) -> np.ndarray:
    return -_roll_max(-x, n)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


@dataclass
class Market:
    dates: np.ndarray
    codes: np.ndarray
    names: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    raw: np.ndarray
    val: np.ndarray
    cap: np.ndarray
    ind: dict = field(default_factory=dict)

    @property
    def T(self):
        return len(self.dates)

    def didx(self, d: str) -> int:
        return int(np.searchsorted(self.dates, d))


def load(npz: str = NPZ) -> Market:
    z = np.load(npz, allow_pickle=True)
    m = Market(z["dates"], z["codes"], z["names"], z["o"], z["h"], z["l"], z["c"], z["raw"], z["val"], z["cap"])
    c = m.c.astype(np.float64)
    I = m.ind
    for n in (5, 10, 20, 60, 120):
        I[f"ma{n}"] = _roll_mean(c, n)
    prev = _shift(c, 1)
    ret1 = c / prev - 1
    I["ret1"] = ret1
    for n in (5, 20, 60, 120, 250):
        I[f"ret{n}"] = c / _shift(c, n) - 1
    I["mom12_1"] = _shift(c, 21) / _shift(c, 250) - 1          # 12-1 모멘텀
    I["hi250"] = _roll_max(m.h.astype(np.float64), 250)
    I["hh20"] = _shift(_roll_max(m.h.astype(np.float64), 20), 1)   # 어제까지 20일 최고가
    I["val20"] = _roll_mean(m.val.astype(np.float64), 20)
    I["vol60"] = np.sqrt(_roll_mean(ret1 ** 2, 60))
    I["max20"] = _roll_max(ret1, 20)                             # 최근 20일 최대 일간 수익률 (MAX 효과)
    # RSI(2) — 단순 평균 근사
    up = np.clip(ret1, 0, None)
    dn = np.clip(-ret1, 0, None)
    au, ad = _roll_mean(up, 2), _roll_mean(dn, 2)
    I["rsi2"] = np.where(ad > 0, 100 - 100 / (1 + au / np.where(ad > 0, ad, 1)), 100.0)
    tr = np.maximum(m.h - m.l, np.maximum(abs(m.h - prev), abs(m.l - prev)))
    I["atr14"] = _roll_mean(tr.astype(np.float64), 14)
    # 시총 순위 (그날 기준, 상장폐지 종목 포함 point-in-time)
    capn = np.nan_to_num(m.cap.astype(np.float64), nan=-1)
    I["caprank"] = (-capn).argsort(axis=1).argsort(axis=1).astype(np.float32) + 1
    I["caprank"][np.isnan(m.cap)] = np.nan
    # 시장 국면: 거래대금 상위 종목 동일가중 지수가 100일선 위인가
    liq = I["val20"] >= 3e9
    mr = np.where(liq & ~np.isnan(ret1), ret1, np.nan)
    ew = np.nancumsum(np.nan_to_num(np.nanmean(np.clip(mr, -0.3, 0.3), axis=1)))
    ewl = np.exp(np.log1p(np.nan_to_num(np.nanmean(np.clip(mr, -0.3, 0.3), axis=1))).cumsum())
    I["mkt"] = ewl
    I["mkt_on"] = ewl > _roll_mean(ewl[:, None], 100)[:, 0]
    I["mkt_ret1"] = np.nan_to_num(np.nanmean(np.clip(mr, -0.3, 0.3), axis=1))
    return m


def universe(m: Market, kind: str) -> np.ndarray:
    """그날 종가 기준 매수 가능 종목 (T × N bool). 전부 그 시점 정보만 쓴다."""
    I = m.ind
    base = (~np.isnan(m.c)) & (m.raw >= 1000) & (I["val20"] >= 1e9) & (m.val > 0)
    if kind == "LARGE":      # 그날 시총 200위 안
        return base & (I["caprank"] <= 200)
    if kind == "MID":        # 시총 201 ~ 1000위, 거래대금 30억 이상
        return base & (I["caprank"] > 200) & (I["caprank"] <= 1000) & (I["val20"] >= 3e9)
    if kind == "LIQ":        # 거래대금 30억 이상 전부
        return base & (I["val20"] >= 3e9)
    raise ValueError(kind)


# ════════════════════════════════════════════════════════════
# 전략 정의
# ════════════════════════════════════════════════════════════
@dataclass
class Spec:
    """
    score: T×N, 클수록 먼저 산다. NaN/−inf 면 후보 아님 (그날 종가 기준 → 다음 날 시가 매수)
    mode:  "rotate"  r 거래일마다 상위 k 로 교체 (보유 종목이 상위 keep 안이면 유지)
           "signal"  빈 슬롯이 생기면 그날 후보 중 상위부터 산다. 청산은 exit 규칙
    exit:  종가 청산 조건 T×N bool (보유 중 그날 종가에 True 면 종가 매도)
    hold:  최대 보유 거래일 (signal 모드)
    stop:  진입가 대비 손절 (예 0.07 → −7% 에 장중 손절). 0 이면 없음
    tp:    진입가 대비 익절 (장중 지정가). 0 이면 없음
    regime: 시장 국면 off 날엔 새로 안 산다 (rotate 는 전부 현금화)
    """
    name: str
    score: np.ndarray
    mode: str = "signal"
    r: int = 21
    keep: int = 0
    exit: np.ndarray | None = None
    hold: int = 0
    stop: float = 0.0
    tp: float = 0.0
    regime: bool = False
    exit_next_open: bool = False       # True 면 청산 신호 다음 날 시가에 판다 (1일 보유 전략용)
    params: dict = field(default_factory=dict)
    top: np.ndarray | None = None      # 날짜별 상위 후보 인덱스 캐시

    def prepare(self, K: int = 12):
        s = np.where(np.isfinite(self.score), self.score, -np.inf)
        idx = np.argpartition(-s, K, axis=1)[:, :K]
        vals = np.take_along_axis(s, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        self.top = [idx[t][np.isfinite(vals[t])] for t in range(len(idx))]
        if self.mode == "rotate" and self.keep:
            kk = self.keep
            ki = np.argpartition(-s, kk, axis=1)[:, :kk]
            kv = np.take_along_axis(s, ki, axis=1)
            self.keepset = [set(ki[t][np.isfinite(kv[t])].tolist()) for t in range(len(ki))]
        return self


@dataclass
class Pos:
    j: int
    qty: int
    entry_adj: float
    cost: float
    t0: int
    peak: float


def run_month(m: Market, sp: Spec, s: int, k: int, capital: float = CAPITAL, length: int = MONTH):
    """s = 첫 거래일 인덱스. 반환: (월 수익률, 거래 목록[(j, 진입일, 청산일, 수익률)])."""
    I = m.ind
    e = min(s + length - 1, m.T - 1)
    cash = float(capital)
    pos: dict[int, Pos] = {}
    trades = []
    pend_exit: set = set()
    slot = capital / k

    def sell(p: Pos, t: int, px_adj: float, extra: float = 0.0):
        nonlocal cash
        raw_px = px_adj * m.raw[t, p.j] / m.c[t, p.j] if m.c[t, p.j] > 0 else px_adj
        net = px_adj * (1 - slip(raw_px) - extra) * (1 - FEE - TAX)
        proceeds = p.cost * net / p.entry_adj
        cash += proceeds
        trades.append((p.j, p.t0, t, proceeds / p.cost - 1))

    def buy(j: int, t: int):
        nonlocal cash
        if j in pos or len(pos) >= k:
            return
        o_adj, c_prev = m.o[t, j], m.c[t - 1, j]
        if not (o_adj > 0) or not (c_prev > 0) or o_adj >= c_prev * 1.295:
            return
        raw_o = o_adj * m.raw[t, j] / m.c[t, j] if m.c[t, j] > 0 else o_adj
        px = raw_o * (1 + slip(raw_o))
        budget = min(cash, slot)
        qty = int(budget / (px * (1 + FEE)))
        if qty < 1:
            return
        cost = qty * px * (1 + FEE)
        cash -= cost
        entry_adj = o_adj * (1 + slip(raw_o)) * (1 + FEE)
        pos[j] = Pos(j, qty, entry_adj, cost, t, o_adj)

    for t in range(s, e + 1):
        sig = t - 1        # 전날 종가 기준 신호
        # ① 시가: 예약 매도
        for j in list(pend_exit):
            if j in pos and m.o[t, j] > 0:
                sell(pos.pop(j), t, float(m.o[t, j]))
            pend_exit.discard(j)
        # ② 시가: 매수
        on = (not sp.regime) or bool(I["mkt_on"][sig])
        if sp.mode == "rotate":
            if (t - s) % sp.r == 0:
                if not on:
                    for j in list(pos):
                        if m.o[t, j] > 0:
                            sell(pos.pop(j), t, float(m.o[t, j]))
                else:
                    keep = sp.keepset[sig] if sp.keep else set(sp.top[sig][:k].tolist())
                    for j in list(pos):
                        if j not in keep and m.o[t, j] > 0:
                            sell(pos.pop(j), t, float(m.o[t, j]))
                    for j in sp.top[sig]:
                        if len(pos) >= k:
                            break
                        buy(int(j), t)
        elif on:
            for j in sp.top[sig]:
                if len(pos) >= k:
                    break
                buy(int(j), t)
        # ③ 장중 손절·익절, 종가 청산
        for j in list(pos):
            p = pos[j]
            o, h, l, c = m.o[t, j], m.h[t, j], m.l[t, j], m.c[t, j]
            if not (c > 0):
                continue
            if sp.stop:
                sl = p.entry_adj * (1 - sp.stop)
                if o <= sl:
                    sell(pos.pop(j), t, float(o), 0.002); continue
                if l <= sl:
                    sell(pos.pop(j), t, float(sl), 0.002); continue
            if sp.tp:
                tp = p.entry_adj * (1 + sp.tp)
                if o >= tp:
                    sell(pos.pop(j), t, float(o)); continue
                if h >= tp:
                    sell(pos.pop(j), t, float(tp)); continue
            held = t - p.t0 + 1
            if sp.mode == "signal":
                hit = (sp.exit is not None and bool(sp.exit[t, j])) or (sp.hold and held >= sp.hold)
                if hit:
                    if sp.exit_next_open:
                        pend_exit.add(j)
                    else:
                        sell(pos.pop(j), t, float(c))
    # ④ 마지막 날 종가에 전부 판다
    for j in list(pos):
        p = pos.pop(j)
        c = m.c[e, j]
        if not (c > 0):
            back = np.where(m.c[:e + 1, j] > 0)[0]
            c = m.c[back[-1], j] if len(back) else p.entry_adj
            sell(p, int(back[-1]) if len(back) else e, float(c))
        else:
            sell(p, e, float(c))
    return cash / capital - 1, trades


def month_dist(m: Market, sp: Spec, starts, k: int) -> dict:
    rets, ntr, wins = [], [], []
    for s in starts:
        r, tr = run_month(m, sp, s, k)
        rets.append(r)
        ntr.append(len(tr))
        wins += [x[3] > 0 for x in tr]
    a = np.array(rets)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p_pos": float((a > 0).mean()), "p10u": float((a >= 0.1).mean()), "p20u": float((a >= 0.2).mean()),
            "p10d": float((a <= -0.1).mean()), "q10": float(np.quantile(a, 0.1)), "q90": float(np.quantile(a, 0.9)),
            "min": float(a.min()), "max": float(a.max()), "trades": float(np.mean(ntr)),
            "win": float(np.mean(wins)) if wins else float("nan"), "rets": a}
