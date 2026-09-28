"""
backtest/lab/panic2_ev_a.py — 패닉 2차 연구 트랙 A (종목 단독 급락) 사건 조사 EA1 ~ EA10 (+ EA3b)

  from backtest.lab import panic2_ev_a as EVA
  lines, summary = EVA.run(F, 'IS')          # VAL · TEST 열은 동결 파일을 쓴 뒤 오케스트레이터가 만든다
  python -m backtest.lab.panic2_ev_a          → IS 만 화면에 출력 (개발용)

정의: backtest/results/lab_panic2_design.json 의 grid.event_studies (EA1 ~ EA10) 와 critique (충돌하면 critique).
신호는 t 종가까지 (open_same 은 t 시가까지) 의 정보만 쓰고, 수익률은 panic2_common.fwd / fwd_path 로만 잰다
(정지일 건너뜀 · 왕복 0.35% · 진입일 = 1일차). 통계는 신호일이 period 안인 사건만 쓴다.

공통 용어
  CF     트랙 A 평온 필터 = ok · x>−2% · t−5..t 에 T0 없음 · ret1>−20% · 하한가 아님 · 배당락 아님 · bad20 아님
  기본   EA1 기본 사건 = CF & 유니버스 & z≤Z & ret1≤−R  (EA2 · EA3 · EA7 은 Z=−3, R=4%)
  초과   같은 날 같은 유니버스 동일가중 (같은 진입 · 청산, 비용 전) 대비
  Δsw    사건 순수익 − 평소 전략 2위 (base_top[t][1]) 비용 전 수익 − 0.35%   (갈아탈 보유 대비)
  위약차 사건 − 같은 날 · 같은 유니버스 · 같은 cr1 5분위 · 같은 vol60(t−1) 3분위 · |z|<1 종목 평균 (비용 전끼리)
  t      날짜 평균 계열의 Newey–West t (시차 h−1). EA4 는 달력 주(월–일) 평균 계열 (시차 ⌈h/5⌉)
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

from backtest.lab import panic2_common as C

COST = C.COST
HZ = (1, 5, 10, 21)
EN = {"close": "종가t", "open": "시가t+1", "open_same": "시가t", "close1": "종가t+1", "close2": "종가t+2",
      "open2": "시가t+2"}
KEYS = ("n", "weeks", "mean", "median", "win", "t_nw", "excess", "t_exc", "dsw", "t_dsw", "plc", "t_plc",
        "top5_share", "maxyear_share", "p5")


# ════════════════════════════════════════════════════════════
# 표 · 숫자
# ════════════════════════════════════════════════════════════
def P(x, d=2):
    return C.pct(x, d)


def TT(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.1f}"


def SH(x):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def SHP(s, k):
    """손익 집중도: 합이 0 이하면 정의 안 됨."""
    return SH(s.get(k)) if s.get("n") and (s.get("mean") or 0) > 0 else ("합≤0" if s.get("n") else "-")


def M(v):
    """NaN 을 뺀 평균 (없으면 NaN)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def tbl(head, rows):
    return (["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""])


def slim(s):
    return {k: v for k, v in s.items() if k in KEYS or k.startswith("exc_E")}


def clean(o):
    """JSON 용: numpy → 파이썬, NaN · inf → None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ════════════════════════════════════════════════════════════
# 기간 문맥 (조각 · 같은 날 EW · 위약 풀 캐시)
# ════════════════════════════════════════════════════════════
class Ctx:
    def __init__(self, F, period: str):
        self.F, self.period = F, period
        self.t0, self.t1 = C.period_range(F, period)
        self.L = self.t1 - self.t0
        self.nyr = self.L / 250
        self.yr = np.array([int(d[:4]) for d in F.dates])
        self.md = np.array([int(d[4:]) for d in F.dates])                      # MMDD
        # 달력 주 (월–일). numpy 의 datetime64[W] 는 목요일 시작이라 3일 밀어서 자른다 (wk_thu = 목요일 시작, 민감도용)
        dd = np.array([np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}") for d in F.dates])
        self.wk = (dd + np.timedelta64(3, "D")).astype("datetime64[W]").astype(np.int64)
        self.wk_thu = dd.astype("datetime64[W]").astype(np.int64)
        self.years = sorted(set(self.yr[self.t0:self.t1].tolist()))
        self.eras = [e for e, (a, z) in C.ERAS.items() if C.didx(F, a) < self.t1 and C.didx(F, z) > self.t0]
        self.era_rng = {e: (C.didx(F, a), C.didx(F, z)) for e, (a, z) in C.ERAS.items()}
        cs = np.concatenate([[0], np.cumsum(np.asarray(F.trig_T0))])
        tt = np.arange(F.T)
        self.recentT0 = (cs[tt + 1] - cs[np.maximum(tt - 5, 0)]) > 0           # t−5..t 에 T0
        self._ew, self._pl, self._cell = {}, {}, {}
        self.cur, self._seen, self._nox = None, {}, {}

    def fwd(self, t, j, entry, h):
        """C.fwd + 진입은 되는데 청산일(데이터 안)부터 끝까지 거래가 없어 빠지는 사건 (상폐 등) 을 조사별로 기록.
        기록 값 = 마지막 거래 종가로 잰 비용 전 수익 (stale mark)."""
        F = self.F
        t, j = np.asarray(t, np.int64), np.asarray(j, np.int64)
        net, g = C.fwd(F, t, j, entry, h)
        if self.cur is None or not len(t):
            return net, g
        self._seen.setdefault(self.cur, []).append(t * F.N + j)
        u = t + h + C.EXIT_OFF[entry]
        miss = ~np.isfinite(net) & (u < F.T)
        if miss.any():
            ep = C.entry_price(F, t[miss], j[miss], entry)
            d = self._nox.setdefault(self.cur, {})
            for tt, jj, uu, e in zip(t[miss], j[miss], u[miss], ep):
                if np.isfinite(e):
                    k = np.where(F.traded[tt:uu + 1, jj])[0]
                    d[int(tt * F.N + jj)] = float(F.c[tt + k[-1], jj] / e - 1) if len(k) else float("nan")
        return net, g

    def s(self, nm, k=0):
        """행 t0..t1−1 에 맞춘 조각 (k 일 전 값)."""
        return np.asarray(getattr(self.F, nm)[self.t0 - k:self.t1 - k])

    def ev(self, m, sc=None, n=None):
        """조각 마스크 → (t, j). n 이 있으면 날짜별 sc 큰 순 n 개."""
        t, j = C.events_from_mask(m, sc, n, 0, self.L)
        return t + self.t0, j

    def ew(self, key, U, entry, h):
        """같은 날 유니버스 동일가중 비용 전 수익 (길이 T). C.same_day_ew 와 같은 정의를 한 번에 계산."""
        k = (key, entry, h)
        if k not in self._ew:
            tt, jj = np.where(U)
            g = C.fwd(self.F, tt + self.t0, jj, entry, h)[1]
            ok = np.isfinite(g)
            s = np.bincount(tt[ok], g[ok], self.L)
            n = np.bincount(tt[ok], minlength=self.L)
            v = np.full(self.F.T, np.nan)
            v[self.t0:self.t1] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            self._ew[k] = v
        return self._ew[k]

    def cells(self, key, U):
        """유니버스 안 cr1 5분위 × vol60(t−1) 3분위 칸 번호 (0..14, 없으면 −1)."""
        if key not in self._cell:
            def qt(v, m, q):
                m = m & np.isfinite(v)
                rk = np.where(m, v, np.inf).argsort(1, kind="stable").argsort(1, kind="stable")
                n = m.sum(1, keepdims=True)
                return np.where(m, np.minimum(rk * q // np.maximum(n, 1), q - 1), -1)
            a = qt(self.s("cr1"), U, 5)
            b = qt(self.s("vol60", 1), U, 3)
            self._cell[key] = np.where((a >= 0) & (b >= 0), a * 3 + b, -1).astype(np.int8)
        return self._cell[key]

    def placebo(self, key, U, entry, h):
        """칸별 위약 평균 비용 전 수익 (L × 15). 사건은 z≤−2.5 라 |z|<1 풀에 들어올 수 없다 (사건 이름 자동 제외)."""
        k = (key, entry, h)
        if k not in self._pl:
            cell = self.cells(key, U)
            pool = (cell >= 0) & (np.abs(self.s("z")) < 1)
            tt, jj = np.where(pool)
            g = C.fwd(self.F, tt + self.t0, jj, entry, h)[1]
            ok = np.isfinite(g)
            kk = tt[ok] * 15 + cell[tt[ok], jj[ok]]
            s = np.bincount(kk, g[ok], self.L * 15)
            n = np.bincount(kk, minlength=self.L * 15)
            self._pl[k] = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(self.L, 15)
        return self._pl[k]

    def st(self, t, j, entry, h, U=None, key=None, net=None, ewv=None, dsw=False, plc=False):
        """사건 통계 = C.ev_stats + 초과 (캐시한 같은 날 EW 또는 사건별 기준 ewv) + 시대별 초과 + Δsw + 위약차."""
        F = self.F
        t, j = np.asarray(t, np.int64), np.asarray(j, np.int64)
        if net is None:
            net = self.fwd(t, j, entry, h)[0]
        net = np.asarray(net, float)
        s = C.ev_stats(F, t, j, net, None, entry, h)
        ok = np.isfinite(net)
        s["_t"], s["_j"], s["_net"] = t[ok], j[ok], net[ok]
        if not ok.any():
            return s
        if ewv is not None:
            ewv = np.asarray(ewv, float)[ok]
        t, j, net = t[ok], j[ok], net[ok]
        lag = max(h - 1, 0)
        s["p5"] = float(np.percentile(net, 5))
        if ewv is None and U is not None:
            ewv = self.ew(key, U, entry, h)[t]
        if ewv is not None:
            exc = net + COST - ewv
            e = np.isfinite(exc)
            s["_exc"] = exc
            if e.any():
                s["excess"] = float(exc[e].mean())
                s["t_exc"] = C.nw_t(C.daily_mean(t[e], exc[e])[1], lag)
                for era in self.eras:
                    a, z = self.era_rng[era]
                    m = e & (t >= a) & (t < z)
                    s[f"exc_{era}"] = float(exc[m].mean()) if m.sum() >= 10 else float("nan")
        if dsw:
            d = net - C.base_gross(F, t, entry, h, (1,)) - COST
            s["dsw"] = M(d)
            s["t_dsw"] = C.nw_t(C.daily_mean(t, d)[1], lag)
        if plc:
            pl = self.placebo(key, U, entry, h)
            cl = self.cells(key, U)[t - self.t0, j]
            pv = np.where(cl >= 0, pl[t - self.t0, np.maximum(cl, 0)], np.nan)
            d = net + COST - pv
            s["plc"] = M(d)
            s["t_plc"] = C.nw_t(C.daily_mean(t, d)[1], lag)
        return s

    def era_mean(self, t, v):
        out = {}
        for era in self.eras:
            a, z = self.era_rng[era]
            m = (t >= a) & (t < z) & np.isfinite(v)
            out[era] = float(v[m].mean()) if m.sum() >= 10 else float("nan")
        return out

    def wk_t(self, t, v, h, thu=False):
        """주 평균 계열의 NW t 와 주 수. h 일 보유는 뒤 ⌈h/5⌉ 주와 겹치므로 시차 ⌈h/5⌉ (5일 → 1, 21일 → 5)."""
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        if ok.sum() < 3:
            return float("nan"), int(ok.sum())
        u, inv = np.unique((self.wk_thu if thu else self.wk)[np.asarray(t)[ok]], return_inverse=True)
        m = np.bincount(inv, v[ok]) / np.bincount(inv)
        return C.nw_t(m, int(np.ceil(h / 5))), len(u)


# ════════════════════════════════════════════════════════════
# 청산 규칙 (EA5 · EA6 · EA9)
# ════════════════════════════════════════════════════════════
def rec_exit(F, t, j, entry, hmax, target="ma5", fail=False):
    """회복 청산: 진입 뒤 1..hmax 일차 종가 중 처음 조건을 만족한 날 매도 (정지일은 못 판다), 없으면 hmax 일차 종가.
    target  ma5: c≥ma5 · pre: c≥c_{t−1} · half: c≥c_t+½(c_{t−1}−c_t).  fail: c<l_t 이면 그날 매도.
    → (순수익, 보유일)."""
    t, j = np.asarray(t, np.int64), np.asarray(j, np.int64)
    if not len(t):
        return np.zeros(0), np.zeros(0)
    Pp = C.fwd_path(F, t, j, entry, hmax)
    u = np.minimum(t[:, None] + C.EXIT_OFF[entry] + np.arange(1, hmax + 1)[None, :], F.T - 1)
    jj = j[:, None]
    cu = F.c[u, jj].astype(float)
    tr = F.traded[u, jj]
    c0, cm1 = F.c[t, j].astype(float), F.c[t - 1, j].astype(float)
    if target == "ma5":
        thr = F.ma5[u, jj].astype(float)
    elif target == "pre":
        thr = cm1[:, None]
    else:
        thr = (c0 + 0.5 * (cm1 - c0))[:, None]
    cond = tr & (cu >= thr)
    if fail:
        cond |= tr & (cu < F.l[t, j].astype(float)[:, None])
    k = np.where(cond.any(1), cond.argmax(1), hmax - 1)
    g = Pp[np.arange(len(t)), k]
    return g - COST, (k + 1).astype(float)


def tpsl_exit(F, t, j, tp, sl, hmax=10):
    """종가 t 진입, 장중 익절 · 손절 (일봉 고가 · 저가). 시가가 선을 넘어 열리면 시가 체결. 손절은 0.2% 추가 미끄러짐,
    같은 날 둘 다 닿으면 손절. 못 닿으면 hmax 일차 종가. → (순수익, 보유일)."""
    t, j = np.asarray(t, np.int64), np.asarray(j, np.int64)
    if not len(t):
        return np.zeros(0), np.zeros(0)
    ep = C.entry_price(F, t, j, "close")
    Pp = C.fwd_path(F, t, j, "close", hmax)
    uu = t[:, None] + np.arange(1, hmax + 1)[None, :]
    u = np.minimum(uu, F.T - 1)
    jj = j[:, None]
    tr = F.traded[u, jj] & (uu < F.T)
    o, h, l = (a[u, jj].astype(float) for a in (F.o, F.h, F.l))
    stp, tgt = ep * (1 - sl), ep * (1 + tp)
    hs = tr & (l <= stp[:, None])
    ht = tr & (h >= tgt[:, None])
    hit = hs | ht
    r = np.arange(len(t))
    any_ = hit.any(1)
    k = np.where(any_, hit.argmax(1), hmax - 1)
    stop = any_ & hs[r, k]
    g = np.where(stop, np.minimum(o[r, k], stp) * (1 - 0.002) / ep - 1,
                 np.where(any_, np.maximum(o[r, k], tgt) / ep - 1, Pp[r, hmax - 1]))
    return g - COST, (k + 1).astype(float)


# ════════════════════════════════════════════════════════════
# EA1 잔차 충격 격자
# ════════════════════════════════════════════════════════════
def ea1(X):
    CF, z, r1, cr1 = X.s("CF"), X.s("z"), X.s("ret1"), X.s("cr1")
    unis = {"U200": X.s("U200"), "MIDQ": X.s("MIDQ")}
    lab = np.asarray(X.F.lab)
    ra, rb, rc, ry, rk, same, sm = [], [], [], [], [], [], {}
    for un, U in unis.items():
        for Z in (-2.5, -3.5, -4.5):
            m7 = CF & U & (z <= Z) & (r1 <= -0.07)
            if not (m7 ^ (CF & U & (z <= Z) & (r1 <= -0.04))).any():
                same.append(f"{un} Z={Z:g}")
            for R in (0.04, 0.07):
                m = CF & U & (z <= Z) & (r1 <= -R)
                tag0 = [un, f"{Z:g}", f"{R * 100:.0f}%"]
                for sel, (t, j) in (("전체", X.ev(m)), ("상위2", X.ev(m, -z, 2))):
                    if sel == "상위2":                   # KOSPI vs KOSDAQ (그날 시장 구분, 종가 t · 5일)
                        lb, kk = lab[t, j], []
                        for code, kn in ((1, "KOSPI"), (2, "KOSDAQ")):
                            s = X.st(t[lb == code], j[lb == code], "close", 5, U, un)
                            kk += [s["n"], P(s.get("mean")), P(s.get("excess")), TT(s.get("t_exc"))]
                            sm[f"{un} Z{Z:g} R{R * 100:.0f} top2 close {kn}"] = {"5": slim(s)}
                        rk.append(tag0 + kk)
                    for en in ("close", "open"):
                        S = {h: X.st(t, j, en, h, U, un, dsw=True, plc=True) for h in HZ}
                        s = S[5]
                        tag = tag0 + [sel, EN[en]]
                        ra.append(tag + [s["n"], s.get("weeks", 0), P(s.get("mean")), P(s.get("median")),
                                         SH(s.get("win")), P(s.get("excess")), TT(s.get("t_exc")), P(s.get("dsw")),
                                         TT(s.get("t_dsw")), P(s.get("plc")), TT(s.get("t_plc")),
                                         SHP(s, "top5_share"), SHP(s, "maxyear_share")]
                                  + [P(s.get(f"exc_{e}")) for e in X.eras])
                        if en == "close":
                            rb.append(tag0 + [sel] + [f"{P(S[h].get('mean'))} / {P(S[h].get('excess'))}" for h in HZ]
                                      + [P(S[21].get("dsw")), P(S[21].get("plc"))])
                        sm[f"{un} Z{Z:g} R{R * 100:.0f} {'all' if sel == '전체' else 'top2'} {en}"] = \
                            {str(h): slim(S[h]) if h in (5, 21) else {k: S[h].get(k) for k in ("n", "mean", "excess")}
                             for h in HZ}
                        if Z == -3.5 and R == 0.04:
                            by = s.get("by_year", {})
                            ry.append([un, sel, EN[en]] + [P(by.get(str(y))) for y in X.years])
                t, j = X.ev(m, -cr1, 2)
                cc = []
                for en in ("close", "open"):
                    s = X.st(t, j, en, 5, U, un)
                    cc += [s["n"], P(s.get("mean")), P(s.get("median")), P(s.get("excess")), TT(s.get("t_exc"))]
                rc.append(tag0 + cc)
    L = ["### EA1 잔차 충격 격자 (Z × R × 유니버스 × 진입)", "",
         "CF & 유니버스 & z≤Z & ret1≤−R 사건의 순수익. 전체 = 모든 사건 동일가중, 상위2 = 날짜별 z 낮은 2개 (대회 방식).", "",
         "**5일 보유 (진입일 = 1일차)**", ""]
    L += tbl(["유니버스", "Z", "R", "선택", "진입", "건수", "주", "평균", "중앙", "승률", "초과", "t초과", "Δsw", "tΔ",
              "위약차", "t위약", "상위5일", "최대연도"] + [f"{e} 초과" for e in X.eras], ra)
    L += ["해석: 위약 풀은 같은 날 유니버스 안 cr1 5분위 × vol60(t−1) 3분위 칸의 |z|<1 종목 (칸이 비면 그 사건은 제외).", "",
          "**보유 기간별 평균 / 초과 (종가 t 진입; 시가 t+1 은 JSON 요약)**", ""]
    L += tbl(["유니버스", "Z", "R", "선택"] + [f"{h}일" for h in HZ] + ["Δsw 21일", "위약차 21일"], rb)
    L += ["**강건성: 날짜별 시총 상위2 (z 대신 cr1 작은 순) · 5일**", ""]
    L += tbl(["유니버스", "Z", "R", "종가 건수", "평균", "중앙", "초과", "t", "시가 건수", "평균", "중앙", "초과", "t"], rc)
    L += ["**연도별 5일 평균 (Z=−3.5, R=4% = SA1 격자점)**", ""]
    L += tbl(["유니버스", "선택", "진입"] + [str(y) for y in X.years], ry)
    L += ["**KOSPI vs KOSDAQ (그날 시장 구분 lab_t) · 상위2 · 종가 t · 5일**", ""]
    L += tbl(["유니버스", "Z", "R", "KOSPI 건수", "평균", "초과", "t", "KOSDAQ 건수", "평균", "초과", "t"], rk)
    L += ["해석: 초과는 두 시장 모두 같은 유니버스 전체의 같은 날 EW 대비."
          + (f" R 4% · 7% 사건이 완전히 같은 칸: {', '.join(same)} (z≤Z 사건이 모두 ret1≤−7%)." if same else ""), ""]
    return L, sm


# ════════════════════════════════════════════════════════════
# EA2 거래대금 마름 vs 폭증
# ════════════════════════════════════════════════════════════
def ea2(X):
    CF, z, r1, vr = X.s("CF"), X.s("z"), X.s("ret1"), X.s("vr")
    U200, MIDQ = X.s("U200"), X.s("MIDQ")
    turn, v60 = X.s("turn1"), X.s("vol60", 1)
    mt = np.nanmedian(np.where(U200, turn, np.nan), 1)[:, None]
    mv = np.nanmedian(np.where(U200, v60, np.nan), 1)[:, None]
    unis = {"U200": U200, "MIDQ": MIDQ, "QLC": U200 & (turn <= mt) & (v60 <= mv)}
    base = CF & (z <= -3) & (r1 <= -0.04)
    rows, sm = [], {}

    def add(tag, t, j, en, U, un, store):
        S = {h: X.st(t, j, en, h, U, un) for h in HZ}
        s = S[5]
        rows.append(tag + [EN[en], s["n"], f"{s['n'] / X.nyr:.0f}"] + [P(S[h].get("mean")) for h in HZ]
                    + [P(s.get("median")), SH(s.get("win")), P(s.get("excess")), TT(s.get("t_exc")),
                       P(S[21].get("excess"))] + [P(s.get(f"exc_{e}")) for e in X.eras])
        sm[store] = {str(h): slim(S[h]) for h in (5, 21)}

    for un, U in unis.items():
        b = base & U
        for a, bb in ((1.2, 3), (1.5, 4)):
            bk = {"DRY": b & (vr <= a), "MID": b & (vr > a) & (vr < bb), "SPIKE": b & (vr >= bb)}
            for bn, m in bk.items():
                for sel in ("전체", "상위2"):
                    t, j = X.ev(m) if sel == "전체" else X.ev(m, -z, 2)
                    ens = ["open"] + (["close"] if bn == "DRY" or un == "QLC" else [])
                    for en in ens:
                        add([un, f"{a}/{bb}", bn, sel], t, j, en, U, un, f"{un} {a}/{bb} {bn} {sel} {en}")
    # 3일 조용한 하락 (grind)
    id3 = X.s("idio") + X.s("idio", 1) + X.s("idio", 2)
    z3 = id3 / (X.s("sig") * np.sqrt(3))
    vmax = np.maximum(np.maximum(X.s("val"), X.s("val", 1)), X.s("val", 2)) / X.s("val20", 3)
    r3 = X.s("ret3")
    for un, U in unis.items():
        for a in (1.2, 1.5):
            m = CF & U & (r3 <= -0.08) & (z3 <= -3) & (vmax <= a)
            add([un, f"≤{a}", "3일 GRIND", "전체"], *X.ev(m), "open", U, un, f"{un} grind a{a} 전체")
    # knife D6 재현 (관심주 & −10%↓ & 거래대금 3배↑, 시가 t+1)
    tr = X.s("tradable")
    t, j = X.ev(tr & X.s("attn20") & (r1 <= -0.10) & (vr >= 3))
    d5, d21 = X.st(t, j, "open", 5, tr, "tradable"), X.st(t, j, "open", 21, tr, "tradable")
    sm["D6"] = {"5": slim(d5), "21": slim(d21)}
    L = ["### EA2 거래대금 마름 vs 폭증 (Z=−3, R=4%)", "",
         "기본 사건을 vr = val/val20(t−1) 구간으로 나눔. DRY vr≤a · MID a<vr<b · SPIKE vr≥b. QLC = U200 중 전날 회전율 · "
         "vol60 모두 U200 중앙값 이하. 3일 GRIND = CF & 3일 −8%↓ & 3일 잔차 z≤−3 & 3일 최대 거래대금/val20(t−3)≤a.", ""]
    L += tbl(["유니버스", "a/b", "구간", "선택", "진입", "건수", "연간"] + [f"{h}일" for h in HZ]
             + ["중앙5", "승률5", "초과5", "t5", "초과21"] + [f"{e} 초과5" for e in X.eras], rows)
    L += ["해석: 상위2 는 구간 안에서 날짜별 z 낮은 2개. QLC 초과는 QLC 자신의 같은 날 EW 대비. "
          "종가 t 진입은 설계대로 DRY · QLC 만 (전체 · 상위2 모두). GRIND 는 하루 사건이 거의 1–2개라 전체만.", "",
          f"knife D6 재현 (tradable & attn20 & ret1≤−10% & vr≥3, 시가 t+1): 건수 {d5['n']}, 5일 평균 {P(d5.get('mean'))} "
          f"(중앙 {P(d5.get('median'))}, 초과 {P(d5.get('excess'))}), 21일 평균 {P(d21.get('mean'))} → "
          f"{'음(−) 재현됨' if (d5.get('mean') or 0) < 0 else '재현 안 됨'}", ""]
    return L, sm


# ════════════════════════════════════════════════════════════
# EA3 하락 모양 · EA3b 시가 갭 되돌림
# ════════════════════════════════════════════════════════════
def overnight(F, t, j):
    """종가 t 매수 → 시가 t+1 매도 순수익 (t+1 정지면 NaN)."""
    t, j = np.asarray(t, np.int64), np.asarray(j, np.int64)
    ep = C.entry_price(F, t, j, "close")
    u = np.minimum(t + 1, F.T - 1)
    o1 = np.where(F.traded[u, j] & (t + 1 < F.T), F.o[u, j].astype(float), np.nan)
    return o1 / ep - 1 - COST


def ea3(X):
    F = X.F
    CF, z, r1 = X.s("CF"), X.s("z"), X.s("ret1")
    gap, intra, ibs, vr = X.s("gap"), X.s("intra"), X.s("ibs"), X.s("vr")
    rows, rows2, sm = [], [], {}

    def add(out, tag, t, j, U, un):
        S = {(en, h): X.st(t, j, en, h, U, un) for en in ("close", "open") for h in (1, 5, 21)}
        s = S[("close", 5)]
        out.append(tag + [s["n"], f"{s['n'] / X.nyr:.0f}", P(M(overnight(F, t, j)))]
                   + [P(S[("close", h)].get("mean")) for h in (1, 5, 21)]
                   + [P(s.get("median")), SH(s.get("win")), P(s.get("excess")), TT(s.get("t_exc")),
                      P(S[("close", 21)].get("excess"))]
                   + [P(S[("open", h)].get(k)) for h, k in ((1, "mean"), (5, "mean"), (5, "excess"), (21, "excess"))])
        for en in ("close", "open"):
            sm[" ".join(tag + [en])] = {str(h): slim(S[(en, h)]) for h in (5, 21)}

    for un in ("U200", "MIDQ"):
        U = X.s(un)
        b = CF & U & (z <= -3) & (r1 <= -0.04)
        for G in (0.03, 0.05):
            NG = b & (gap <= -G) & (intra <= 0.005)
            AB = b & (gap <= -G) & (intra >= 0.01) & (vr >= 2)
            IL = b & (gap >= -0.01) & (intra <= -G) & (ibs <= 0.2)
            IH = b & (gap >= -0.01) & (intra <= -G) & (ibs > 0.2)
            cl = {"NEWS-GAP": NG, "ABSORB": AB, "INTRA-LOW": IL, "INTRA-HI": IH, "OTHER": b & ~(NG | AB | IL | IH)}
            for cn, m in cl.items():
                add(rows, [un, f"{G * 100:.0f}%", cn], *X.ev(m), U, un)
    # 넓은 집합: 평온 & cr1≤500 & ok & ret1≤−8%
    calm = (X.s("x") > -0.02) & ~X.recentT0[X.t0:X.t1] & ~X.s("divx")
    UW = X.s("ok") & (X.s("cr1") <= 500)
    W = UW & calm[:, None] & ~X.s("bad20") & (r1 <= -0.08)
    G3 = W & X.s("limdown") & (X.s("c") <= X.s("l") * (1 + 1e-6))
    cl = {"전체": W, "G1 갭↓회복": W & ~G3 & (gap <= -0.05) & (ibs >= 0.6),
          "G2 장중 밀림": W & ~G3 & (gap > -0.02) & (ibs <= 0.15), "G3 하한가 잠김": G3}
    for cn, m in cl.items():
        add(rows2, [cn], *X.ev(m), UW, "C500")
    head = ["건수", "연간", "밤사이", "종가 1일", "종가 5일", "종가 21일", "중앙5", "승률5", "초과5", "t5", "초과21",
            "시가 1일", "시가 5일", "시가 초과5", "시가 초과21"]
    L = ["### EA3 하락 모양 (갭 · 장중 · 종가 위치)", "",
         "기본 사건 (Z=−3, R=4%) 을 서로 배타적인 모양으로 나눔 (G=I). NEWS-GAP gap≤−G & intra≤+0.5% · ABSORB gap≤−G & "
         "intra≥+1% & vr≥2 · INTRA-LOW gap≥−1% & intra≤−I & ibs≤0.2 · INTRA-HI 같은데 ibs>0.2 · OTHER 나머지.", ""]
    L += tbl(["유니버스", "G=I", "모양"] + head, rows)
    L += ["종가 = 종가 t 진입 (중앙5 · 승률5 · 초과5 · t5 · 초과21 도 종가 기준), 시가 = 시가 t+1 진입. "
          "밤사이 = 종가 t 매수 → 시가 t+1 매도 순수익.", "",
          "**넓은 집합: 평온 & cr1≤500 & ok & ret1≤−8% (G1 gap≤−5% & ibs≥0.6 · G2 gap>−2% & ibs≤0.15 · G3 하한가 & c=l)**", ""]
    L += tbl(["분류"] + head, rows2)
    L += ["해석: 넓은 집합의 평온 = x>−2% · t−5..t 에 T0 없음 · 배당락 아님 · bad20 아님 (ret1 · 하한가 제외는 안 함). "
          "G1 · G2 는 G3 과 겹치지 않게 뺌. 초과는 ok & cr1≤500 의 같은 날 EW 대비.", ""]
    return L, sm


def ea3b(X):
    # 시가 t 에 아는 정보만 쓰도록 여기서 다시 만든다. 캐시 m_SA8 · gz 는 EW 갭을 그날 liquid (그날 거래대금이 든
    # val20_t) 로, 30만원 조건을 그날 종가 raw_t 로 잡아 시가 시점에는 모르는 값이 섞인다.
    gap = X.s("gap")
    ewg = np.nanmean(np.where(X.s("liquid", 1) & np.isfinite(gap), np.clip(gap, -0.3, 0.3), np.nan), 1)
    gz = (gap - X.s("bstar") * ewg[:, None]) / X.s("sig")
    U = X.s("U200", 1) & X.s("traded")
    base = (U & (X.s("raw", 1) <= 300_000) & ~X.s("bad20") & ~X.s("divx")[:, None] & (ewg > -0.01)[:, None]
            & (gz <= -3))
    rows, sm, diff = [], {}, []
    for G, nm in ((0.04, "m_SA8"), (0.07, "m_SA8b")):
        m = base & (gap <= -G)
        mc = X.s(nm)
        diff.append(f"G={G * 100:.0f}% 여기 {int(m.sum())}건 · 캐시 {nm} {int(mc.sum())}건 · 공통 {int((m & mc).sum())}건")
        sm[f"G{G * 100:.0f} vs_cache"] = {"here": int(m.sum()), "cache": int(mc.sum()), "both": int((m & mc).sum())}
        for sel in ("전체", "상위2"):
            t, j = X.ev(m) if sel == "전체" else X.ev(m, -gz, 2)
            for h, xn in ((1, "종가 t"), (2, "종가 t+1"), (5, "FIX5")):
                s = X.st(t, j, "open_same", h, U, "U200p")
                rows.append([f"{G * 100:.0f}%", sel, xn, s["n"], f"{s['n'] / X.nyr:.0f}", P(s.get("mean")),
                             P(s.get("median")), SH(s.get("win")), P(s.get("excess")), TT(s.get("t_exc")),
                             SHP(s, "top5_share")] + [P(s.get(f"exc_{e}")) for e in X.eras])
                sm[f"G{G * 100:.0f} {sel} h{h}"] = slim(s)
    L = ["### EA3b 시가 갭 되돌림 (critique 12a)", "",
         "전날 U200 & 전날 raw≤30만 & 오늘 gap≤−G & 갭 잔차 z (gap−b*·EW갭)/sig≤−3 & EW 갭>−1% & bad20 · 배당락 아님. "
         "EW 갭 = 전날 liquid 종목의 오늘 갭 평균 (시가에 아는 값만). 시가 t 매수 (+0.3% 미끄러짐), "
         "청산 종가 t (h=1) · 종가 t+1 (h=2) · FIX5 (h=5).", ""]
    L += tbl(["G", "선택", "청산", "건수", "연간", "평균", "중앙", "승률", "초과", "t", "상위5일"]
             + [f"{e} 초과" for e in X.eras], rows)
    L += ["해석: 초과 기준은 전날 U200 & 오늘 거래된 종목의 같은 날 EW (같은 시가 t 진입 · 같은 청산). 상위2 는 gz 낮은 순. "
          f"캐시 m_SA8 (그날 liquid · 그날 raw 사용) 과 비교: {' / '.join(diff)}.", ""]
    return L, sm


# ════════════════════════════════════════════════════════════
# EA4 업종 동반 vs 단독
# ════════════════════════════════════════════════════════════
def rel_to_peers(X, v, peer, sec, S):
    """같은 업종 다른 종목(peer) 중앙값 대비 v (자기 자신 제외, 다른 종목 4개↑). 행 t0..t1−1 조각 → 같은 모양."""
    tt, jj = np.where(peer)
    vv = v[tt, jj].astype(float)
    g = tt.astype(np.int64) * (S + 1) + sec[jj]
    o = np.lexsort((vv, g))
    gs, vs = g[o], vv[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    n = np.bincount(np.cumsum(first) - 1)[np.cumsum(first) - 1]
    k = np.arange(len(gs)) - start
    m = n - 1
    lo, hi = (m - 1) // 2, m // 2
    ilo = np.minimum(start + lo + (lo >= k), len(vs) - 1)
    ihi = np.minimum(start + hi + (hi >= k), len(vs) - 1)
    med = np.where(m >= 4, 0.5 * (vs[ilo] + vs[ihi]), np.nan)
    out = np.full(v.shape, np.nan)
    out[tt[o], jj[o]] = vs - med
    return out


def ea4(X):
    F = X.F
    CF, r1, cr1 = X.s("CF"), X.s("ret1"), X.s("cr1")
    secx, sbr = X.s("secx"), X.s("sbreadth")
    sec = np.asarray(F.sector).astype(np.int64)
    S_ = len(F.sectors)
    labd = np.isfinite(secx)
    nolab = (sec < 0)[None, :]
    rows, rows_p, sm, sens = [], [], {}, []

    def row(tag, t, j, U, un, out=rows, store=None):
        s5, s21 = X.st(t, j, "open", 5, U, un), X.st(t, j, "open", 21, U, un)
        w5, nw = X.wk_t(s5.get("_t", []), s5.get("_exc", []), 5)
        w21, _ = X.wk_t(s21.get("_t", []), s21.get("_exc", []), 21)
        w5h, _ = X.wk_t(s5.get("_t", []), s5.get("_exc", []), 5, thu=True)       # 주 경계 민감도 (목요일 시작)
        if np.isfinite(w5) and np.isfinite(w5h):
            sens.append((abs(w5 - w5h), " ".join(map(str, tag)), w5, w5h))
        out.append(tag + [s5["n"], nw, P(s5.get("mean")), P(s5.get("median")), P(s5.get("excess")), TT(w5),
                          P(s21.get("mean")), P(s21.get("excess")), TT(w21)])
        sm[store or " ".join(map(str, tag))] = {"5": slim(s5), "21": slim(s21), "t_wk5": w5, "t_wk21": w21,
                                                "t_wk5_thu": w5h}

    for un in ("U200", "MIDQ"):
        U = X.s(un)
        for R in (0.05, 0.08):
            e0 = CF & U & (r1 <= -R)
            rt = [un, f"{R * 100:.0f}%"]
            row(rt + ["-", "SINGLE"], *X.ev(e0 & (secx >= -0.01)), U, un)
            row(rt + ["-", "라벨 없음"], *X.ev(e0 & nolab), U, un)
            row(rt + ["-", "업종 동료<4"], *X.ev(e0 & ~labd & ~nolab), U, un)
            cand = CF & U & (r1 <= -R / 2) & (sec >= 0)[None, :]
            for S in (0.02, 0.04):
                sect = e0 & (secx <= -S) & (sbr >= 0.5)
                mix = e0 & labd & ~(secx >= -0.01) & ~sect
                t, j = X.ev(sect)
                key = (t - X.t0) * (S_ + 1) + sec[j]
                o = np.lexsort((r1[t - X.t0, j], key))
                f = np.r_[True, key[o][1:] != key[o][:-1]] if len(o) else np.zeros(0, bool)
                deep = (t[o][f], j[o][f])
                tc, jc = np.where(cand)
                kc = tc * (S_ + 1) + sec[jc]
                o2 = np.lexsort((cr1[tc, jc], kc))
                f2 = np.r_[True, kc[o2][1:] != kc[o2][:-1]] if len(o2) else np.zeros(0, bool)
                keep = np.isin(kc[o2][f2], np.unique(key))
                big = (tc[o2][f2][keep] + X.t0, jc[o2][f2][keep])
                st = rt + [f"{S * 100:.0f}%"]
                row(st + ["SECTOR"], t, j, U, un)
                row(st + ["SECTOR 최대낙폭"], *deep, U, un)
                row(st + ["SECTOR 최대시총"], *big, U, un)
                row(st + ["MIXED"], *X.ev(mix), U, un)
    # P1 분해
    tr, raw, at = X.s("tradable"), X.s("raw"), X.s("attn20")
    PU = tr & (raw <= 300_000) & ((cr1 <= 300) | at)
    x1 = X.s("x")[:, None]
    resid = np.clip(r1, -0.3, 0.3) - X.s("bstar") * x1 - (secx - x1)
    for e in (0.08, 0.12):
        ev = PU & (r1 <= -e)
        cl = {"M x≤−2%": ev & (x1 <= -0.02), "M x≤−3%": ev & (x1 <= -0.03),
              "S 업종": ev & (x1 > -0.02) & (secx <= -0.04) & (resid > -0.06),
              "I 단독": ev & (x1 > -0.01) & (np.abs(secx) < 0.02) & (resid <= -0.08),
              "secx 없음 (x>−2%)": ev & (x1 > -0.02) & ~labd}
        for cn, m in cl.items():
            t, j = X.ev(m)
            s5, s21 = X.st(t, j, "open", 5, PU, "P1U"), X.st(t, j, "open", 21, PU, "P1U")
            if cn.startswith("M") and s5["n"]:
                u, dm = C.daily_mean(s5["_t"], s5["_exc"])
                ep = C.ep_table(u, dm, 10)
                ncl, tcl = ep["N"], C.nw_t(np.array(ep.get("ep_means", [])), 0)
            else:
                tcl, ncl = X.wk_t(s5.get("_t", []), s5.get("_exc", []), 5)
            em = X.era_mean(s5.get("_t", np.zeros(0)), s5.get("_exc", np.zeros(0)))
            rows_p.append([f"{e * 100:.0f}%", cn, s5["n"], ncl, P(s5.get("mean")), P(s5.get("median")),
                           SH(s5.get("win")), P(s5.get("excess")), TT(tcl), P(s21.get("mean")), P(s21.get("excess"))]
                          + [P(em[k]) for k in X.eras])
            sm[f"P1 e{e * 100:.0f} {cn}"] = {"5": slim(s5), "21": slim(s21), "clusters": ncl, "t_cl": tcl}
    # HM-rel5
    c, v20p = X.s("c"), X.s("val20", 1)
    peer = np.isfinite(c) & (raw >= 1000) & (v20p >= 1e9) & (sec >= 0)[None, :] & np.isfinite(X.s("ret5"))
    rel5 = rel_to_peers(X, X.s("ret5"), peer, np.maximum(sec, 0), S_)
    U300 = X.s("ok") & (cr1 <= 300)
    hm = CF & U300 & peer & (rel5 <= -0.10)
    rows_h = []
    row(["HM-rel5", "전체"], *X.ev(hm), U300, "C300", rows_h, "HM all")
    row(["HM-rel5", "상위2"], *X.ev(hm, -rel5, 2), U300, "C300", rows_h, "HM top2")
    # 라벨 통제 (critique 2): SA1 상위2 를 라벨 유무로
    t, j, _ = C.events_from_top(F.top_SA1, 2, X.t0, X.t1)
    lb = sec[j] >= 0
    U200 = X.s("U200")
    for nm, mm in (("SA1 상위2 라벨 있음", lb), ("SA1 상위2 라벨 없음", ~lb)):
        s = X.st(t[mm], j[mm], "close", 5, U200, "U200")
        w5, nw = X.wk_t(s["_t"], s.get("_exc", np.zeros(0)), 5)
        rows_h.append([nm, "종가t · 5일", s["n"], nw, P(s.get("mean")), P(s.get("median")), P(s.get("excess")), TT(w5),
                       "-", "-", "-"])
        sm[nm] = dict(slim(s), t_wk5=w5)
    cov = []
    for y in X.years:
        rr = X.yr[X.t0:X.t1] == y
        uu = U200[rr]
        cov.append(f"{y} {SH((uu & (sec >= 0)[None, :]).sum() / max(uu.sum(), 1))}")
    sm["label_coverage_U200"] = cov
    L = ["### EA4 업종 동반 vs 단독 (2018 KSIC 라벨)", "",
         "CF & 유니버스 & ret1≤−R. SINGLE secx≥−1% · SECTOR secx≤−S & 업종 3%↓ 비중≥50% · MIXED 나머지. secx 가 없는 사건은 "
         "따로: 라벨 없음 (2018 스냅샷에 없음) · 업종 동료<4 (라벨은 있으나 같은 업종 다른 종목 4개 미만). 시가 t+1 진입, "
         "t주 = 주 평균 초과의 NW t.", ""]
    L += tbl(["유니버스", "R", "S", "분류", "건수", "주", "5일", "중앙5", "초과5", "t주5", "21일", "초과21", "t주21"], rows)
    L += ["해석: SECTOR 최대낙폭 = 그날 그 업종 SECTOR 사건 중 ret1 최저 1개, 최대시총 = 같은 업종 CF & 유니버스 & "
          "ret1≤−R/2 중 cr1 최소 1개 (SECTOR 사건이 있는 업종-날짜만).", "",
          "주 경계 민감도: t주5 를 목요일 시작 주로 다시 재면 차이가 큰 행 — "
          + " · ".join(f"{nm} {a:.1f}→{b:.1f}" for _, nm, a, b in sorted(sens, reverse=True)[:3])
          + ". 이 정도 흔들리면 t주 는 참고로만 본다.", "",
          "**P1 분해 ((cr1≤300 또는 attn20) & tradable & 30만원↓ & ret1≤−e, 시가 t+1)**", ""]
    L += tbl(["e", "분류", "건수", "군집", "5일", "중앙5", "양수5", "초과5", "t군집", "21일", "초과21"]
             + [f"{k} 초과5" for k in X.eras], rows_p)
    L += ["해석: 잔차 = clip(ret1) − b*·x − (secx − x). 군집 = M 은 에피소드 (간격 10일, t 는 에피소드 평균), S · I 는 주. "
          "평온 필터 없음 (P1 원 정의). 초과는 P1 유니버스 같은 날 EW 대비.", "",
          "**HM-rel5 (CF & cr1≤300, rel5 = ret5 − 같은 업종 다른 종목 ret5 중앙값 ≤ −10%) · 라벨 통제**", ""]
    L += tbl(["규칙", "선택", "건수", "주", "5일", "중앙5", "초과5", "t주5", "21일", "초과21", "t주21"], rows_h)
    L += [f"U200 종목·일 라벨 비율: {' · '.join(cov)}. 라벨은 2018-11 스냅샷이라 2019년 이후 상장은 '라벨 없음' 이 된다 — "
          f"2019+ 부분집합은 VAL · TEST 실행 표 자체다 (IS 에는 없음).", ""]
    return L, sm


# ════════════════════════════════════════════════════════════
# EA5 건강한 추세 속 눌림 · EA6 느린 하락 vs 한 번 충격
# ════════════════════════════════════════════════════════════
def ea5(X):
    F = X.F
    CF, z, r1, cr1, raw = X.s("CF"), X.s("z"), X.s("ret1"), X.s("cr1"), X.s("raw")
    c1, hi1, r60, mx, ma120 = X.s("c", 1), X.s("hi250", 1), X.s("ret60", 1), X.s("max20", 1), X.s("ma120", 1)
    d = X.s("dd_atr")
    U200 = X.s("U200")
    rows, sm = [], {}
    for Q in (0.85, 0.92):
        pre = ((cr1 <= 200) & (raw <= 300_000) & (c1 / hi1 >= Q) & (r60 > 0) & (r60 <= 0.40) & (mx < 0.10)
               & (c1 > ma120))
        for K in (1.5, 2.5, 3.5):
            m = CF & pre & (d <= -K) & (z <= -2) & (r1 > -0.15)
            for sel in ("전체", "상위2"):
                t, j = X.ev(m) if sel == "전체" else X.ev(m, -d, 2)
                S = {(en, h): X.st(t, j, en, h, U200, "U200") for en in ("close", "open") for h in (5, 10, 21)}
                ma = {en: rec_exit(F, t, j, en, 10, "ma5") for en in ("close", "open")}
                s = S[("close", 5)]
                mn, hd = ma["close"]
                rows.append([f"{Q}", f"{K}", sel, s["n"], f"{s['n'] / X.nyr:.0f}"]
                            + [P(S[("close", h)].get("mean")) for h in (5, 10, 21)]
                            + [P(s.get("median")), SH(s.get("win")), P(s.get("excess")), TT(s.get("t_exc")),
                               P(M(mn)), f"{M(hd[np.isfinite(mn)]):.1f}" if np.isfinite(mn).any() else "-",
                               P(S[("open", 5)].get("mean")), P(S[("open", 5)].get("excess")), P(M(ma["open"][0]))])
                for en in ("close", "open"):
                    sm[f"Q{Q} K{K} {sel} {en}"] = {"5": slim(S[(en, 5)]), "10": slim(S[(en, 10)]),
                                                   "21": slim(S[(en, 21)]), "exit_ma": M(ma[en][0])}
    L = ["### EA5 건강한 추세 속 눌림 (SA4 격자)", "",
         "t−1 상태: cr1≤200 · c/hi250≥Q · 0<ret60≤40% · max20<10% · c>ma120. t: (c−ma20)/atr14(t−1)≤−K & z≤−2 & CF & "
         "ret1>−15%. HOLD5 = 5일 열, EXIT-MA = 1..10일차 종가 중 처음 c≥ma5 에 매도 (없으면 10일차).", ""]
    L += tbl(["Q", "K", "선택", "건수", "연간", "종가 5일", "10일", "21일", "중앙5", "승률5", "초과5", "t5",
              "EXIT-MA", "보유일", "시가 5일", "시가 초과5", "시가 EXIT-MA"], rows)
    L += ["해석: 종가 = 종가 t 진입 (중앙 · 승률 · 초과 · EXIT-MA 도 종가 기준), 시가 = 시가 t+1 진입. 상위2 는 날짜별 d 낮은 순. "
          "시가 진입의 EXIT-MA 도 진입일(t+1) 종가부터 본다. 초과는 U200 EW 대비.", ""]
    return L, sm


def ea6(X):
    F = X.F
    CF, U200, r1 = X.s("CF"), X.s("U200"), X.s("ret1")
    nd, icum, wst, bst, c = X.s("down_days"), X.s("icum"), X.s("worst_streak"), X.s("bstar"), X.s("c")
    Mi = np.asarray(F.M)
    tt = np.arange(X.t0, X.t1)
    m5 = (Mi[tt] / Mi[tt - 5] - 1 > -0.04)[:, None]
    rows, sm = [], {}
    for N in (5, 7):
        icN = c / X.s("c", N) - 1 - bst * (Mi[tt] / Mi[tt - N] - 1)[:, None]
        wN = r1.copy()
        for k in range(1, N):
            wN = np.fmin(wN, X.s("ret1", k))
        for Xv in (0.06, 0.10):
            GR = CF & U200 & (nd >= N) & (icum <= -Xv) & (wst >= -0.04) & m5
            SK = CF & U200 & (icN <= -Xv) & (wN <= -0.7 * Xv) & m5
            for typ, m, sc in (("GRIND", GR, -icum), ("SHOCK", SK, -icN)):
                for sel in ("전체", "상위2"):
                    t, j = X.ev(m) if sel == "전체" else X.ev(m, sc, 2)
                    S = {h: X.st(t, j, "close", h, U200, "U200") for h in (5, 21)}
                    s = S[5]
                    mn, hd = rec_exit(F, t, j, "close", 7, "ma5")
                    rows.append([typ, N, f"{Xv * 100:.0f}%", sel, s["n"], f"{s['n'] / X.nyr:.0f}", P(s.get("mean")),
                                 P(S[21].get("mean")), P(s.get("median")), SH(s.get("win")), P(s.get("excess")),
                                 TT(s.get("t_exc")), P(S[21].get("excess")), P(M(mn)),
                                 f"{M(hd[np.isfinite(mn)]):.1f}" if np.isfinite(mn).any() else "-"])
                    sm[f"{typ} N{N} X{Xv * 100:.0f} {sel}"] = {"5": slim(s), "21": slim(S[21]), "exit_ma5": M(mn)}
    L = ["### EA6 느린 하락 (GRIND) vs 한 번 충격 (SHOCK)", "",
         "GRIND: CF & U200 & 연속 하락 ≥N일 & 연속 구간 잔차 누적 icum≤−X & 구간 최악일≥−4% & 시장 5일>−4%. SHOCK: CF & U200 & "
         "N일 잔차 누적≤−X & N일 중 최악일≤−0.7X & 시장 5일>−4%. 종가 t 진입, MA5 청산 = 1..7일차 처음 c≥ma5.", ""]
    L += tbl(["유형", "N", "X", "선택", "건수", "연간", "5일", "21일", "중앙5", "승률5", "초과5", "t5", "초과21",
              "MA5 청산", "보유일"], rows)
    L += ["해석: SHOCK 에도 GRIND 와 같은 CF · 시장 5일 조건을 붙여 비교 가능하게 함. 연속 구간이 이어지면 날마다 사건이 됨 (전체).", ""]
    return L, sm


# ════════════════════════════════════════════════════════════
# EA7 위험 플래그 · 실적 대용 · 배당락 · 상태 대용 · 달력 · 시장
# ════════════════════════════════════════════════════════════
def ea7(X):
    F = X.F
    t0, t1 = X.t0, X.t1
    ok, z, r1 = X.s("ok"), X.s("z"), X.s("ret1")
    x = X.s("x")
    calm = (x > -0.02) & ~X.recentT0[t0:t1]
    U200, MIDQ = X.s("U200"), X.s("MIDQ")
    dv = X.s("divx")
    B = ok & (calm & ~dv)[:, None] & (z <= -3) & (r1 <= -0.04)
    tU, jU = X.ev(B & U200)
    tM, jM = X.ev(B & MIDQ)
    t, j = np.r_[tU, tM].astype(np.int64), np.r_[jU, jM].astype(np.int64)
    isU = np.r_[np.ones(len(tU), bool), np.zeros(len(tM), bool)]
    R = {}
    for en in ("close", "open"):
        for h in (5, 21):
            net = X.fwd(t, j, en, h)[0]
            ewv = np.where(isU, X.ew("U200", U200, en, h)[t], X.ew("MIDQ", MIDQ, en, h)[t])
            R[(en, h)] = (net, net + COST - ewv)

    def g(nm, k=0):
        return np.asarray(getattr(F, nm))[t - k, j]

    # 위험 플래그 F1..F6
    r1e, ld = g("ret1"), g("limdown")
    F1a, F1b = (r1e <= -0.15) | ld, (r1e <= -0.20) | ld
    F2 = (g("ret1", 1) <= -0.10) & (r1e <= -0.10)
    r60p, mx20p = g("ret60", 1), g("max20", 1)
    F3a, F3b = (r60p >= 0.30) | (mx20p >= 0.15), (r60p >= 0.50) | (mx20p >= 0.15)
    md, mo, yy = X.md[t], X.md[t] // 100, X.yr[t]
    F4 = (g("lab") == 2) & (g("cr1") > 300) & (md >= 310) & (md <= 415)
    F5 = (g("raw") < 2000) | (g("val20", 1) < 3e9)
    F6 = g("c", 1) / g("hi250", 1) <= 0.5
    # 금융 배당 급락 (critique 3): 같은 금융 업종 U200 3개↑ −3%↓, 12월 (2023년부터는 2–4월도)
    fin = np.array([any(k in s for k in ("금융", "보험", "은행", "신탁")) for s in F.sectors] + [False])
    sec = np.asarray(F.sector).astype(np.int64)
    secx_ = np.where(sec >= 0, sec, len(F.sectors))
    cnt = np.zeros(len(t))
    for d in np.unique(t):
        dn = (U200[d - t0] & (r1[d - t0] <= -0.03))
        bc = np.bincount(secx_[dn], minlength=len(F.sectors) + 1)
        sel = t == d
        cnt[sel] = bc[secx_[j[sel]]]
    DIVF = fin[secx_[j]] & (cnt >= 3) & ((mo == 12) | ((yy >= 2023) & (mo >= 2) & (mo <= 4)))
    # 실적 대용: P1 달력 · P2 자기 계절성 (w=3 은 여기서 계산, w=5 는 F.earn)
    ym = X.yr * 100 + X.md // 100
    first = np.r_[0, np.where(ym[1:] != ym[:-1])[0] + 1]
    sess = np.arange(F.T) - first[np.searchsorted(first, np.arange(F.T), side="right") - 1] + 1
    win = (((md >= 125) & (md <= 215)) | ((md >= 420) & (md <= 516)) | ((md >= 720) & (md <= 816))
           | ((md >= 1020) & (md <= 1115)))
    P1 = win | ((g("cr1") <= 50) & np.isin(mo, (1, 4, 7, 10)) & (sess[t] <= 10))
    k0 = max(t0 - 270, 0)
    fl = ((np.abs(np.nan_to_num(np.asarray(F.gap[k0:t1]))) / np.asarray(F.sig[k0:t1]) >= 3)
          | (np.nan_to_num(np.asarray(F.vr[k0:t1])) >= 3))

    def p2(w):
        acc = np.zeros(len(t), bool)
        for off in list(range(-252 - w, -252 + w + 1)) + list(range(-63 - w, -63 + w + 1)):
            ix = t + off - k0
            acc |= (ix >= 0) & fl[np.maximum(ix, 0), j]
        return acc
    P2w3, P2w5 = p2(3), g("earn")
    gp = g("gap")
    E3, E5 = P2w3 | (P1 & (gp <= -0.02)), P2w5 | (P1 & (gp <= -0.02))
    # 달력 수급일 (+): 옵션 만기 · MSCI 리밸런싱 대용 (2·5·8·11월 마지막 거래일)
    mo_all = X.md // 100
    msci = np.isin(mo_all, (2, 5, 8, 11)) & np.r_[mo_all[1:] != mo_all[:-1], False]      # 데이터 끝 날은 모름
    OPT, MS = np.asarray(F.optx)[t], msci[t]
    flags = [("F1 ret1≤−15% 또는 하한가", F1a), ("F1 ret1≤−20% 또는 하한가", F1b), ("F2 연속 −10%", F2),
             ("F3 ret60≥30% 또는 max20≥15%", F3a), ("F3 ret60≥50% 또는 max20≥15%", F3b),
             ("F4 감사철 KOSDAQ cr1>300", F4), ("F5 2천원↓ 또는 val20<30억", F5), ("F6 c/hi250≤50%", F6),
             ("F1–F6 합 (15% · 30%)", F1a | F2 | F3a | F4 | F5 | F6), ("F1–F6 합 (20% · 50%)", F1b | F2 | F3b | F4 | F5 | F6),
             ("상태 대용 bad20", g("bad20")), ("금융 배당 급락 (critique 3)", DIVF),
             ("P1 실적 달력", P1), ("P2 자기 계절성 w=3", P2w3), ("P2 자기 계절성 w=5", P2w5), ("P1 & P2(w=5)", P1 & P2w5),
             ("EARN w=3", E3), ("EARN w=5", E5), ("옵션 만기일 (+)", OPT), ("MSCI 리밸런싱 대용 (+)", MS),
             ("만기 · MSCI 합 (+)", OPT | MS), ("KOSDAQ (유지 = 나머지)", g("lab") == 2)]
    n0 = len(t)
    nc5, ec5 = R[("close", 5)]
    rows, sm = [], {}
    allrow = ["기본 전체 (U200 + MIDQ)", n0, "100%", P(M(nc5)), P(M(ec5)), TT(C.nw_t(C.daily_mean(t, ec5)[1], 4)),
              P(M(R[("open", 5)][1])), P(M(R[("close", 21)][1])), P(M(R[("open", 21)][1])), "-", "-", "-", "-"]
    rows.append(allrow)
    for nm, f in flags:
        f = np.asarray(f, bool)
        k = ~f

        def mm(key, which, msk):
            return P(M(R[key][which][msk]))
        rows.append([nm, int(f.sum()), SH(f.mean() if n0 else np.nan), mm(("close", 5), 0, f), mm(("close", 5), 1, f),
                     TT(C.nw_t(C.daily_mean(t[f], ec5[f])[1], 4)), mm(("open", 5), 1, f), mm(("close", 21), 1, f),
                     mm(("open", 21), 1, f), mm(("close", 5), 0, k), mm(("close", 5), 1, k), mm(("open", 5), 1, k),
                     mm(("close", 21), 1, k)])
        sm[nm] = {"n": int(f.sum()), "net5c": M(nc5[f]), "exc5c": M(ec5[f]), "exc5o": M(R[("open", 5)][1][f]),
                  "exc21c": M(R[("close", 21)][1][f]), "kept_exc5c": M(ec5[k]), "kept_exc21c": M(R[("close", 21)][1][k])}
    sm["earn_w5_match_rate"] = float((p2(5) == P2w5).mean()) if n0 else None
    # 필터 효과: 배당락 제외 · 상태 대용 제외
    CFx = ok & calm[:, None] & (r1 > -0.20) & ~X.s("limdown") & ~X.s("bad20")
    sets = [("SA1 정의 · 배당락일 (제외된 것)", CFx & dv[:, None] & U200 & (z <= -3.5) & (r1 <= -0.04), U200, "U200"),
            ("Z−3 R4% U200 · 배당락일", CFx & dv[:, None] & U200 & (z <= -3) & (r1 <= -0.04), U200, "U200"),
            ("Z−3 R4% MIDQ · 배당락일", CFx & dv[:, None] & MIDQ & (z <= -3) & (r1 <= -0.04), MIDQ, "MIDQ"),
            ("SA1 · bad20 로 제외된 것", X.s("m_SA1ns") & ~X.s("m_SA1"), U200, "U200"),
            ("SA1 유지 (m_SA1)", X.s("m_SA1"), U200, "U200")]
    rows2 = []
    for nm, m, U, un in sets:
        tt_, jj_ = X.ev(m)
        a, b, c_ = X.st(tt_, jj_, "close", 5, U, un), X.st(tt_, jj_, "close", 21, U, un), X.st(tt_, jj_, "open", 5, U, un)
        rows2.append([nm, a["n"], len(np.unique(tt_)), P(a.get("mean")), P(a.get("median")), P(a.get("excess")),
                      P(b.get("mean")), P(b.get("excess")), P(c_.get("mean")), P(c_.get("excess"))])
        sm[nm] = {"c5": slim(a), "c21": slim(b), "o5": slim(c_)}
    L = ["### EA7 위험 플래그 · 실적 대용 · 배당락 · 상태 대용 · 달력", "",
         "기본 = ok & x>−2% & t−5..t 에 T0 없음 & 배당락 아님 & (U200 또는 MIDQ) & z≤−3 & ret1≤−4% (플래그를 보려고 CF 의 "
         "ret1>−20% · 하한가 · bad20 제외는 풀었다). 초과는 사건 자기 유니버스 EW 대비, 플래그 쪽 vs 유지(플래그 아님) 쪽.", ""]
    L += tbl(["플래그", "건수", "비중", "5일 종가", "초과5 종가", "t", "초과5 시가", "초과21 종가", "초과21 시가",
              "유지 5일 종가", "유지 초과5 종가", "유지 초과5 시가", "유지 초과21 종가"], rows)
    L += ["해석: P1 첫 10거래일 조건은 cr1≤50. P2 w=5 는 F.earn (w=3 은 같은 식으로 여기서 계산). EARN = P2 | (P1 & gap≤−2%). "
          "KOSPI200 6·12월 정기변경은 그 달 옵션 만기일에 들어 있고, MSCI 는 2·5·8·11월 마지막 거래일로 근사.", "",
          "**필터 효과 (배당락 제외 · 상태 대용 bad20)**", ""]
    L += tbl(["집합", "건수", "날짜", "5일 종가", "중앙", "초과5", "21일 종가", "초과21", "5일 시가", "초과5 시가"], rows2)
    L += ["해석: 배당락일 = 12월 마지막 두 거래일 (2022년까지, F.divx). 배당락일 행은 CF 에서 배당락 조건만 뺀 필터를 쓴다.", ""]
    return L, sm


# ════════════════════════════════════════════════════════════
# EA8 확인 진입 · EA9 청산 규칙
# ════════════════════════════════════════════════════════════
def ea8(X):
    F = X.F
    U200 = X.s("U200")
    rows, sm = [], {}
    for nm in ("SA1", "SA3"):
        t, j, _ = C.events_from_top(getattr(F, f"top_{nm}"), 2, X.t0, X.t1)
        t = t.astype(np.int64)
        n0 = len(t)
        u = np.minimum(t + 1, F.T - 1)
        tr1 = F.traded[u, j] & (t + 1 < F.T)
        l0, l1, c0, c1, o1 = (a[v, j].astype(float) for a, v in ((F.l, t), (F.l, u), (F.c, t), (F.c, u), (F.o, u)))
        C1 = tr1 & (l1 >= l0) & (c1 >= c0)
        vs = [("IMM", np.ones(n0, bool), "close"), ("C1", C1, "close1"),
              ("C1 + x>−2%", C1 & (np.asarray(F.x)[u] > -0.02), "close1"),
              ("C2", tr1 & (c1 > o1) & (F.ibs[u, j] >= 0.6) & (o1 >= 0.99 * c0), "close1")]
        for H in (5, 10):
            for vn, keep, en in vs:
                s = X.st(t[keep], j[keep], en, H, U200, "U200")
                em = X.era_mean(s["_t"], s["_net"])
                rows.append([nm, H, vn, EN[en], s["n"], SH(keep.sum() / n0 if n0 else np.nan), P(s.get("mean")),
                             P(s.get("median")), SH(s.get("win")), P(s.get("p5")), P(s.get("excess")),
                             TT(s.get("t_exc"))] + [P(em[e]) for e in X.eras])
                sm[f"{nm} H{H} {vn}"] = dict(slim(s), kept=keep.sum() / n0 if n0 else None)
    L = ["### EA8 확인 진입 (SA1 · SA3 상위2)", "",
         "IMM 종가 t 매수. C1 = t+1 에 l≥l_t & c≥c_t 일 때만 종가 t+1 매수, C2 = t+1 이 양봉 & ibs≥0.6 & 시가≥0.99·c_t 일 때만. "
         "조건 실패 사건은 버림. 보유 H 일은 진입일부터 (종가 t+1 진입은 t+1+H 종가 청산).", ""]
    L += tbl(["집합", "H", "변형", "진입", "건수", "유지율", "평균", "중앙", "승률", "5퍼센타일", "초과", "t"]
             + [f"{e} 평균" for e in X.eras], rows)
    L += ["해석: 날짜별 z 상위2 를 먼저 고른 뒤 확인 조건을 건다. 'C1 + x>−2%' 는 계약 A04 처럼 진입일 시장 평온을 추가한 것.", ""]
    return L, sm


def ea9(X):
    F = X.F
    rules = ["FIX3", "FIX5", "FIX10", "REC ma5", "REC ma5 +FAIL", "REC 전일종가", "REC 전일종가 +FAIL",
             "REC 반되돌림", "REC 반되돌림 +FAIL", "TPSL +5/−10", "TPSL +8/−12"]
    res, ns, hx = {}, {}, {}
    for k in range(1, 8):
        nm = f"SA{k}"
        t, j, _ = C.events_from_top(getattr(F, f"top_{nm}"), 1, X.t0, X.t1)
        t = t.astype(np.int64)
        # critique 1: 21일 안에 정지 · 상폐(거래 없음)를 만난 비중, 수익을 못 잰 사건 수
        uu = t[:, None] + np.arange(1, 22)[None, :]
        inr = uu < F.T
        stuck = (~F.traded[np.minimum(uu, F.T - 1), j[:, None]] & inr).any(1)
        hx[nm] = (float(stuck.mean()) if len(t) else np.nan, int((~np.isfinite(C.fwd(F, t, j, "close", 5)[0])).sum()))
        out = {}
        for H in (3, 5, 10):
            out[f"FIX{H}"] = (X.fwd(t, j, "close", H)[0], np.full(len(t), float(H)))
        for tg, tn in (("ma5", "ma5"), ("pre", "전일종가"), ("half", "반되돌림")):
            out[f"REC {tn}"] = rec_exit(F, t, j, "close", 10, tg, False)
            out[f"REC {tn} +FAIL"] = rec_exit(F, t, j, "close", 10, tg, True)
        for tp, sl in ((0.05, 0.10), (0.08, 0.12)):
            out[f"TPSL +{tp * 100:.0f}/−{sl * 100:.0f}"] = tpsl_exit(F, t, j, tp, sl, 10)
        for r, (net, hd) in out.items():
            ok = np.isfinite(net)
            n, h = net[ok], hd[ok]
            res[(nm, r)] = (M(n), float(np.median(n)) if len(n) else np.nan, float((n > 0).mean()) if len(n) else np.nan,
                            M(h), len(n))
        ns[nm] = int(np.isfinite(out["FIX5"][0]).sum())
    sets = [f"SA{k}" for k in range(1, 8)]
    r1 = [[r] + [f"{P(res[(s, r)][0])} · {res[(s, r)][3]:.1f}일 · {21 / res[(s, r)][3]:.1f}회"
                 if np.isfinite(res[(s, r)][3]) else "-" for s in sets] for r in rules]
    r1.append(["21일 안 정지 · 상폐 비중 / 수익 못 잰 사건"] + [f"{SH(hx[s][0])} / {hx[s][1]}건" for s in sets])
    r2 = [[r] + [f"{P(res[(s, r)][1])} · {SH(res[(s, r)][2])}" for s in sets] for r in rules]
    L = ["### EA9 청산 규칙 (SA1–SA7 날짜별 1위, 종가 t 진입)", "",
         "거래당 순수익 · 평균 보유일 · 슬롯-월 거래 수 (21/보유일). REC = 1..10일차 종가 중 처음 목표 도달에 매도 "
         "(ma5: c≥ma5, 전일종가: c≥c_{t−1}, 반되돌림: c≥c_t+½(c_{t−1}−c_t)), +FAIL = c<l_t 이면 그날 매도. "
         "TPSL = 장중 익절/손절 (손절 +0.2% 미끄러짐, 같은 날 둘 다면 손절), 최대 10일.", ""]
    L += tbl(["청산 \\ 건수"] + [f"{s} ({ns[s]})" for s in sets], r1)
    L += ["**중앙값 · 승률**", ""]
    L += tbl(["청산"] + sets, r2)
    L += ["해석: 정지일에는 못 팔고 (조건 판정 안 함), 10일차가 정지면 그 뒤 처음 거래된 종가. 시가가 선 밖에서 열리면 시가 체결.", ""]
    sm = {f"{s} {r}": {"mean": res[(s, r)][0], "median": res[(s, r)][1], "win": res[(s, r)][2],
                       "hold": res[(s, r)][3], "trades_per_slot_month": 21 / res[(s, r)][3] if res[(s, r)][3] else None,
                       "n": res[(s, r)][4]} for s in sets for r in rules}
    sm.update({f"{s} halt21_share": hx[s][0] for s in sets})
    sm.update({f"{s} unmeasured_fix5": hx[s][1] for s in sets})
    return L, sm


# ════════════════════════════════════════════════════════════
# EA10 반대매매 시점
# ════════════════════════════════════════════════════════════
def ev10(F, a, z):
    """행 a..z−1 의 EA10 사건 (t, j, 유형 0=(a) 시장 패닉 1=(b) 단독)."""
    ok, cr1, r1 = (np.asarray(getattr(F, n)[a:z]) for n in ("ok", "cr1", "ret1"))
    x = np.asarray(F.x[a:z])[:, None]
    base = ok & (cr1 <= 1000)
    ta, ja = np.where(base & (x <= -0.03) & (r1 <= -0.07))
    tb, jb = np.where(base & (x > -0.03) & (r1 <= -0.10))
    return (np.r_[ta, tb] + a).astype(np.int64), np.r_[ja, jb].astype(np.int64), np.r_[np.zeros(len(ta)), np.ones(len(tb))]


def mx_of(F, t, j):
    """MX(t−1) = 전날 liquid 종목 안 백분위 (ret60 · 회전율 val20/cap · max20) 평균 + KOSDAQ 이면 0.1."""
    out = np.full(len(t), np.nan)
    for d in np.unique(t):
        sel = np.where(t == d)[0]
        jj, p = j[sel], d - 1
        liq = np.asarray(F.liquid[p])
        acc = np.zeros(len(jj))
        for arr in (F.ret60[p], F.turn1[d], F.max20[p]):          # turn1[d] = val20(d−1)/cap(d−1)
            v = np.asarray(arr, float)
            ref = np.sort(v[liq & np.isfinite(v)])
            q = np.searchsorted(ref, v[jj], side="right") / max(len(ref), 1)
            acc += np.where(np.isfinite(v[jj]) & (len(ref) > 0), q, np.nan)
        out[sel] = acc / 3 + 0.1 * (np.asarray(F.lab[p])[jj] == 2)
    return out


def ea10(X):
    F = X.F
    t, j, ty = ev10(F, X.t0, X.t1)
    mx = mx_of(F, t, j)
    # 3분위 경계는 IS 사건으로 고정 (유형별)
    if X.period == "IS":
        ti, ji, tyi, mxi = t, j, ty, mx
    else:
        a, z = C.period_range(F, "IS")
        ti, ji, tyi = ev10(F, a, z)
        mxi = mx_of(F, ti, ji)
    cuts = {k: np.nanquantile(mxi[tyi == k], [1 / 3, 2 / 3]) for k in (0, 1)}
    ENT = {"O1": ("open", 6, 21), "C1": ("close1", 5, 20), "O2": ("open2", 5, 20), "C2": ("close2", 4, 19)}
    R = {}
    for nm, (en, h6, h21) in ENT.items():
        R[(nm, 6)] = X.fwd(t, j, en, h6)[0]
        R[(nm, 21)] = X.fwd(t, j, en, h21)[0]
    nmx = {k: int((~np.isfinite(mx[ty == k])).sum()) for k in (0, 1)}
    tn = np.minimum(t + 1, F.T - 1)
    r1n = np.where(t + 1 < F.T, np.asarray(F.ret1)[tn, j], np.nan)
    neg = r1n < 0
    for hh in (6, 21):
        R[("O2c", hh)] = np.where(neg, R[("O2", hh)], np.nan)
    gp = np.asarray(F.gap)
    gk = {k: np.where(t + k < F.T, gp[np.minimum(t + k, F.T - 1), j], np.nan) for k in (1, 2, 3)}
    i1 = np.where(t + 1 < F.T, np.asarray(F.intra)[tn, j], np.nan)
    kpl, kql = np.asarray(F.kprank1)[t, j] <= 200, np.asarray(F.kqrank1)[t, j] <= 100
    rows_g, rows_r, sm = [], [], {"cuts_IS": {("a", "b")[k]: cuts[k].tolist() for k in (0, 1)},
                                  "mx_missing": {("a", "b")[k]: nmx[k] for k in (0, 1)}}
    for k, tyn in ((0, "(a) 시장 −3%↓ & 종목 −7%↓"), (1, "(b) 종목 −10%↓ & 시장>−3%")):
        q1, q2 = cuts[k]
        base = ty == k
        grp = {"전체": base, "MX 상": base & (mx > q2), "MX 중": base & (mx > q1) & (mx <= q2), "MX 하": base & (mx <= q1),
               "KOSPI-L": base & kpl, "KOSDAQ-L": base & kql}
        for gn, m in grp.items():
            rows_g.append([tyn, gn, int(m.sum()), len(np.unique(t[m])), P(M(gk[1][m])), P(M(gk[2][m])),
                           P(M(gk[2][m & neg])), P(M(gk[3][m])), P(M(i1[m]))])
            d6 = R[("O2", 6)][m] - R[("O1", 6)][m]
            rows_r.append([tyn, gn] + [P(M(R[(e, 6)][m])) for e in ("O1", "C1", "O2", "C2", "O2c")] + [P(M(d6))]
                          + [P(M(R[(e, 21)][m])) for e in ("O1", "C1", "O2", "C2", "O2c")])
            sm[f"{'ab'[k]} {gn}"] = {"n": int(m.sum()), "g1": M(gk[1][m]), "g2": M(gk[2][m]), "g2_neg": M(gk[2][m & neg]),
                                     **{f"{e}_t6": M(R[(e, 6)][m]) for e in ("O1", "C1", "O2", "C2", "O2c")},
                                     **{f"{e}_t21": M(R[(e, 21)][m]) for e in ("O1", "O2")}, "O2mO1_t6": M(d6)}
    L = ["### EA10 반대매매 시점 (D+2 경매)", "",
         "사건: ok & cr1≤1000 & ((a) x≤−3% & ret1≤−7% 또는 (b) ret1≤−10% & x>−3%). MX = 전날 liquid 안 ret60 · 회전율 · max20 "
         "백분위 평균 (+0.1 KOSDAQ), 3분위 경계는 IS 사건으로 유형별 고정. 갭 g_k = o_{t+k}/c_{t+k−1}−1 (비용 전), i1 = t+1 장중.", ""]
    L += tbl(["유형", "묶음", "건수", "날짜", "g1", "g2", "g2 (ret1_{t+1}<0)", "g3", "i1"], rows_g)
    L += ["**공통 종점 순수익: 종가 t+6 · 종가 t+21** (O1 시가 t+1 · C1 종가 t+1 · O2 시가 t+2 · C2 종가 t+2 · "
          "O2c = ret1_{t+1}<0 일 때만 O2)", ""]
    L += tbl(["유형", "묶음", "t+6 O1", "C1", "O2", "C2", "O2c", "O2−O1", "t+21 O1", "C1", "O2", "C2", "O2c"], rows_r)
    L += [f"해석: KOSPI-L = 전날 KOSPI 안 시총 200위, KOSDAQ-L = 전날 KOSDAQ 안 100위. IS 경계 (a) "
          f"{cuts[0][0]:.3f}/{cuts[0][1]:.3f}, (b) {cuts[1][0]:.3f}/{cuts[1][1]:.3f}. 평온 필터 없음 (교차 트랙). "
          f"MX 를 못 잰 사건 (전날 ret60 · max20 없음: 신규 상장 등) 은 '전체' 에만 든다: (a) {nmx[0]}건 · (b) {nmx[1]}건.", ""]
    return L, sm


# ════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════
STUDIES = (("EA1", ea1), ("EA2", ea2), ("EA3", ea3), ("EA3b", ea3b), ("EA4", ea4), ("EA5", ea5), ("EA6", ea6),
           ("EA7", ea7), ("EA8", ea8), ("EA9", ea9), ("EA10", ea10))


def run(F, period: str):
    """트랙 A 사건 조사 → (마크다운 줄, JSON 요약). period 안 신호일만."""
    t_run = time.time()
    X = Ctx(F, period)
    L = [f"## 트랙 A 사건 조사 EA1–EA10 — {period} (신호일 {F.dates[X.t0]} ~ {F.dates[X.t1 - 1]})", "",
         "순수익 = 왕복 0.35% 뺀 FIX-h (진입일 = 1일차, 정지일은 다음 거래 종가). 초과 = 같은 날 같은 유니버스 EW 대비 (비용 전끼리), "
         "t = 날짜 평균 계열 Newey–West (시차 h−1) 라 날짜 가중이다 (사건 가중 평균과 부호가 다를 수 있음). "
         "CF = ok · x>−2% · t−5..t T0 없음 · ret1>−20% · 하한가 · 배당락 · bad20 아님. "
         "청산 가격이 끝내 없는 사건 (상폐 등) 은 C.fwd 규칙대로 통계에서 빠진다 (EA9 에 건수).", ""]
    S = {"period": period, "t0": F.dates[X.t0], "t1": F.dates[X.t1 - 1]}
    tm = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nm, fn in STUDIES:
            a = time.time()
            X.cur = nm
            lines, sm = fn(X)
            L += lines
            S[nm] = sm
            tm[nm] = round(time.time() - a, 1)
    X.cur = None
    # 점검: 청산 가격이 없어 (진입 뒤 데이터 안에서 끝까지 거래 없음: 상폐 등) C.fwd 가 NaN 으로 뺀 사건
    rows, nox = [], {}
    for nm, _ in STUDIES:
        seen = len(np.unique(np.concatenate(X._seen[nm]))) if X._seen.get(nm) else 0
        d = X._nox.get(nm, {})
        v = np.array(list(d.values()), float)
        nox[nm] = {"events": seen, "no_exit": len(d), "stale_mean": M(v)}
        rows.append([nm, seen, len(d), f"{len(d) / seen * 100:.2f}%" if seen else "-", P(M(v))])
    L += ["### 점검: 청산 가격 없음으로 빠진 사건 (상폐 등)", "",
          "C.fwd 규칙상 청산일 뒤로 거래가 한 번도 없으면 수익이 NaN 이라 위 모든 표에서 빠진다 (생존 편향 쪽). "
          "고유 종목·일 기준 건수와, 마지막 거래 종가로 잰 비용 전 수익 (stale mark; 다음 날부터 거래가 없으면 0) 평균. "
          "EA3 밤사이 · REC · TPSL · EXIT-MA 경로는 세지 않음.", ""]
    L += tbl(["조사", "고유 사건", "청산 없음", "비중", "stale mark 평균"], rows)
    S["noexit"] = nox
    S["headline"] = headline(S)
    S["timing_s"] = tm
    S["runtime_s"] = round(time.time() - t_run, 1)
    return L, clean(S)


def headline(S) -> dict:
    """요약 머리: 핵심 칸 몇 개 + EA1 5일 칸 중 트랙 A 관문 모양 (서술용, 관문 판정은 계약 설정에서) 을 넘는 칸."""
    def g(d, *ks):
        for k in ks:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d

    def gate(s):
        f = lambda k, d=np.nan: s.get(k) if s.get(k) is not None else d
        return (f("n", 0) >= 200 and f("weeks", 0) >= 100 and f("median", -1) > 0 and f("excess", -1) > 0
                and f("t_exc", 0) >= 2 and f("top5_share", 9) < 0.40 and f("maxyear_share", 9) < 0.35 and f("plc", -1) > 0)
    e1 = S.get("EA1", {})
    return {"EA1 U200 Z-3.5 R4 top2 close h5 (SA1 격자)": g(e1, "U200 Z-3.5 R4 top2 close", "5"),
            "EA1 U200 Z-3.5 R4 top2 close h21": g(e1, "U200 Z-3.5 R4 top2 close", "21"),
            "EA1 MIDQ Z-3.5 R4 top2 close h5": g(e1, "MIDQ Z-3.5 R4 top2 close", "5"),
            "EA1 cells passing gate-shape h5": [k for k, v in e1.items() if gate(v.get("5", {}))],
            "EA2 U200 1.5/4 SPIKE all open h5": g(S, "EA2", "U200 1.5/4 SPIKE 전체 open", "5"),
            "EA2 U200 1.5/4 MID all open h5": g(S, "EA2", "U200 1.5/4 MID 전체 open", "5"),
            "EA3b G4 all h1": g(S, "EA3b", "G4 전체 h1"),
            "EA7 F1-F6 any (15/30)": g(S, "EA7", "F1–F6 합 (15% · 30%)"),
            "EA7 EARN w=5": g(S, "EA7", "EARN w=5"),
            "EA8 SA1 H5 C2": g(S, "EA8", "SA1 H5 C2"),
            "EA9 SA1 FIX5 / REC ma5": [g(S, "EA9", "SA1 FIX5"), g(S, "EA9", "SA1 REC ma5")],
            "EA10 a all / b all": [g(S, "EA10", "a 전체"), g(S, "EA10", "b 전체")]}


def main() -> int:
    import json
    import resource
    from backtest.lab import panic2_feat as PF
    F = PF.load()
    lines, summ = run(F, "IS")              # 개발 · 확인용: IS 만
    js = json.dumps(summ, ensure_ascii=False, allow_nan=False)
    print("\n".join(lines))
    print(f"\n# {len(lines)} lines · JSON {len(js) / 1e3:.0f} KB · runtime {summ['runtime_s']}s · {summ['timing_s']} · "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
