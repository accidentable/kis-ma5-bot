"""
backtest/lab/panic2.py — 종목 패닉 · 시장 패닉 매수 2차 연구 (사전 등록 격자 · 관문 · 동결 · 검증)

  python -m backtest.lab.panic2 is     P0 점검 → P1 사건 조사(IS) → 관문 → P2 IS 대회 → 동결 파일
  python -m backtest.lab.panic2 val    P3 VAL (동결 파일 필요, 코드가 동결 뒤 바뀌면 거부)
  python -m backtest.lab.panic2 test   P4 TEST (한 번) → backtest/results/lab_panic2.md

설계: backtest/results/lab_panic2_design.json (가설 → 격자 → 검토 의견). 검토 의견이 격자보다 우선한다.
준비: python -m backtest.lab.panic2_feat (지표 캐시), 사건 조사 모듈 panic2_ev_a (종목) · panic2_ev_b (시장).

트랙 B (시장 패닉): 평소엔 현재 봇(R0), 시장 트리거가 오면 보유를 패닉 후보로 바꾼다 (전량 / 1+1 분할).
트랙 A (종목 패닉): 평소엔 R0, 조용한 날 한 종목만 크게 빠지면 평소 보유 중 순위 가장 낮은 것을 그 종목으로 바꾼다.
대조군: R0-REFRESH — 매매는 안 하고, 하이브리드가 평소 전략으로 돌아온 날에 똑같이 재정렬만 한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, replace

import numpy as np

from backtest.lab import engine as E
from backtest.lab import panic2_common as C
from backtest.lab import panic2_feat as PF
from backtest.lab import panic_sim2 as S2

warnings.filterwarnings("ignore")
OUT = os.path.join(E.ROOT, "backtest", "results")
FREEZE = os.path.join(OUT, "lab_panic2_freeze.json")
ISJ = os.path.join(OUT, "lab_panic2_is.json")
VALJ = os.path.join(OUT, "lab_panic2_val.json")
TESTJ = os.path.join(OUT, "lab_panic2_test.json")
REPORT = os.path.join(OUT, "lab_panic2.md")
FILES = ["backtest/lab/panic2.py", "backtest/lab/panic2_common.py", "backtest/lab/panic2_feat.py",
         "backtest/lab/panic_sim2.py", "backtest/lab/panic2_ev_a.py", "backtest/lab/panic2_ev_b.py"]
SEED = 20260928
N_RC = 1000          # IS 현실성 검사 (여러 설정 중 최고를 고른 효과) 추첨 수
N_NOISE = 50         # 종가 결정 잡음 검사 추첨 수
N_PLACEBO = 1000     # VAL (v) 위약 추첨 수
IDX_COST = 0.0013    # 합성 지수 왕복 비용 (수수료 0.015%×2 + 슬리피지 0.05%×2, 세금 없음)


# ════════════════════════════════════════════════════════════
# 공통 준비
# ════════════════════════════════════════════════════════════
class Ctx:
    def __init__(self):
        self.F = F = PF.load()
        self.T, self.N = F.T, F.N
        self.m = PF.market_of(F)
        self._m_idx = None
        self.base = PF.lists_of(F.top_base)
        self.surge = np.asarray(F.ret1 >= 0.10)
        self.ident = np.arange(F.T)
        self._lists: dict = {}
        self.starts = {p: self._starts(*C.PERIODS[p], 2 if p != "TEST" else 1) for p in C.PERIODS}
        self.U200p = np.zeros_like(F.U200)
        self.U200p[1:] = F.U200[:-1]
        self.U200p &= F.traded
        # 평온도(변동성 삼분위: 직전 750일 분포 대비) · 200일선 위
        v = F.mkt_vol20
        self.vol_terc = np.full(F.T, -1)
        for t in range(260, F.T):
            w = v[max(0, t - 750):t - 1]
            w = w[np.isfinite(w)]
            if len(w) > 100 and np.isfinite(v[t - 1]):
                q1, q2 = np.quantile(w, [1 / 3, 2 / 3])
                self.vol_terc[t] = 0 if v[t - 1] <= q1 else (1 if v[t - 1] <= q2 else 2)
        self.ab200 = np.zeros(F.T, bool)
        self.ab200[1:] = F.above200[:-1]
        self.R0 = S2.Hybrid("R0", self.base, self.surge)

    def _starts(self, a, z, step):
        s0, s1 = C.didx(self.F, a), min(C.didx(self.F, z), self.T - E.MONTH)
        return list(range(max(s0, 260), s1, step))

    def L(self, name):
        if name not in self._lists:
            self._lists[name] = PF.lists_of(getattr(self.F, f"top_{name}"))
        return self._lists[name]

    @property
    def m_idx(self):
        if self._m_idx is None:
            self._m_idx = PF.market_of(self.F, "k200")
        return self._m_idx

    def trig(self, name):
        return np.asarray(getattr(self.F, f"trig_{name}"))

    def ep_ids(self, days):
        idx = np.where(days)[0]
        out = np.full(self.T, -1)
        out[idx] = C.episodes_of(idx, 10)
        return out


@dataclass
class Cfg:
    id: str
    track: str                   # R 기준 / B 시장 / A 종목 / AB 조합
    desc: str
    hyb: S2.Hybrid
    trig: str = ""               # B: 트리거 이름, A: 사건 이름 (SA1 …)
    sel: str = ""                # B: 후보 목록 이름
    entry: str = "open"          # 사건 기준 진입 (panic2_common 규칙)
    exit: str = "FIX"            # FIX / XMKT / XMKT_STOP / REC / STAGED_FIX / STAGED_XS / IDX
    hold: int = 5
    ref: bool = False            # 기준선 (선택 대상 아님)
    close_dec: bool = False      # 종가에 결정하고 같은 종가에 체결하는 규칙이 있음
    fam_t: str = ""
    fam_x: str = ""
    comps: tuple = ()            # 조합 설정: 구성 요소가 전부 IS 관문을 통과해야 돌린다
    use_idx: bool = False


def tfam(tn):
    return "T0" if tn in ("T0", "Fa", "Fc") else tn


def afam(sa):
    return {"SA1C1": "SA1", "SA3C1": "SA3"}.get(sa, sa)


def build_configs(X: Ctx) -> list:
    T = X.T
    ident = X.ident
    cf: list = []

    def hyB(name, sig, sel, **kw):
        d = dict(mp_sig=sig, mp_rank_day=ident, mp_anchor_day=ident, mp_top=X.L(sel) if sel else None,
                 mp_exit=S2.Exit(hold=5), mp_episode=X.ep_ids(sig))
        d.update(kw)
        return S2.Hybrid(name, X.base, X.surge, **d)

    T0, T1, T6, Fa = X.trig("T0"), X.trig("T1"), X.trig("T6"), X.trig("Fa")
    cf.append(Cfg("R0", "R", "현재 봇 (패닉 규칙 없음)", X.R0, ref=True))
    cf.append(Cfg("R1", "R", "T0 · K2 · 다음 날 시가 · 5일 (lab_panic IS 1위, 기준)", hyB("R1", T0, "K2"),
                  trig="T0", sel="K2", ref=True, fam_t="T0", fam_x="FIX5"))
    for cid, tn, d in (("B01", "T1", "T1 시장 z≤−4"), ("B02", "T2", "T2 3%↓ 종목 75%↑"), ("B03", "T3", "T3 KOSPI200 대용 −3%"),
                       ("B04", "T4", "T4 3일 −8%"), ("B05", "T5", "T5 변동성 급등"), ("B06", "T7", "T7 망치형"),
                       ("B07", "Fa", "T0 & F-a(상승장 충격·깊은 투매)"), ("B08", "Fc", "T0 & F-c(고점 근처 충격)")):
        cf.append(Cfg(cid, "B", f"{d} · K2 · 시가 · 5일", hyB(cid, X.trig(tn), "K2"), trig=tn, sel="K2",
                      fam_t=tfam(tn), fam_x="FIX5"))
    cf.append(Cfg("B09", "B", "T6 반등 확인(x≥+1%) · K2 목록 · 확인일 종가 · 5일",
                  hyB("B09", T6, "T6K2", mp_entry="close"), trig="T6", sel="T6K2", entry="close", close_dec=True,
                  fam_t="T6", fam_x="FIX5"))
    cf.append(Cfg("B10", "B", "T6 반등 확인 · K2 목록 · 다음 날 시가 · 5일", hyB("B10", T6, "T6K2"), trig="T6",
                  sel="T6K2", fam_t="T6", fam_x="FIX5"))
    for cid, sel, d in (("B11", "SRES", "S-RES 잔차 z 최저"), ("B12", "SLEAD", "S-LEAD 넘어진 주도주"),
                        ("B13", "SLOSER", "S-LOSER 120일 낙오주"), ("B14", "SSECREL", "S-SECREL 업종 대비 낙폭"),
                        ("B15", "SHVOL", "S-HVOL 거래대금 급증 하락주"), ("B16", "SIBS", "S-IBS 저가서 회복한 하락주"),
                        ("B28", "SBETA", "S-BETA 고베타 하락주"), ("B29", "SMEGA", "S-MEGA 시총 30 낙폭순"),
                        ("B30", "K2EP", "K2-EP 10일 고점 대비 낙폭순")):
        cf.append(Cfg(cid, "B", f"T0 · {d} · 시가 · 5일", hyB(cid, T0, sel), trig="T0", sel=sel, fam_t="T0", fam_x="FIX5"))
    sh = np.zeros(T, bool)
    sh[1:] = T0[:-1]
    rd = ident.copy()
    rd[1:] = ident[:-1]
    cf.append(Cfg("B17", "B", "T0 · KOSDAQ 시총 100 낙폭순 · 이틀 뒤(t+2) 시가 · 5일",
                  hyB("B17", sh, "KQL", mp_rank_day=rd, mp_anchor_day=rd), trig="T0", sel="KQL", entry="open2",
                  fam_t="T0", fam_x="FIX5"))
    idx_top = [np.array([X.N])] * T
    cf.append(Cfg("B18", "B", "T0 · KOSPI200 대용 지수 1배 (한 종목, 전액) · 시가 · 5일",
                  S2.Hybrid("B18", X.base, X.surge, mp_sig=T0, mp_rank_day=ident, mp_anchor_day=ident, mp_top=idx_top,
                            mp_exit=S2.Exit(hold=5), mp_episode=X.ep_ids(T0), idx_col=X.N),
                  trig="T0", sel="IDX", exit="IDX", use_idx=True, fam_t="T0", fam_x="FIX5"))
    cf.append(Cfg("B19", "B", "T0 · K2 · 당일 종가 · 5일", hyB("B19", T0, "K2", mp_entry="close"), trig="T0", sel="K2",
                  entry="close", close_dec=True, fam_t="T0", fam_x="FIX5"))
    cf.append(Cfg("B20", "B", "T0 · K2 · 이틀 뒤(t+2) 시가 · 5일", hyB("B20", sh, "K2", mp_rank_day=rd, mp_anchor_day=rd),
                  trig="T0", sel="K2", entry="open2", fam_t="T0", fam_x="FIX5"))
    cf.append(Cfg("B21", "B", "T0 · K2 · 시가 · 10일", hyB("B21", T0, "K2", mp_exit=S2.Exit(hold=10)), trig="T0",
                  sel="K2", hold=10, fam_t="T0", fam_x="LONG"))
    cf.append(Cfg("B22", "B", "T0 · K2 · 시장 +6% 회복 시 매도 (최대 10일)",
                  hyB("B22", T0, "K2", mp_exit=S2.Exit(hold=10, mkt_tp=0.06)), trig="T0", sel="K2", exit="XMKT",
                  hold=10, close_dec=True, fam_t="T0", fam_x="LONG"))
    cf.append(Cfg("B23", "B", "T0 · K2 · 시장 +6% 익절 · −6% 손절 · 사건당 2회",
                  hyB("B23", T0, "K2", mp_exit=S2.Exit(hold=10, mkt_tp=0.06, mkt_stop=0.06), mp_max_entries=2),
                  trig="T0", sel="K2", exit="XMKT_STOP", hold=10, close_dec=True, fam_t="T0", fam_x="LONG"))
    cf.append(Cfg("B24", "B", "T0 · K2 · 1+1 분할 (추가 T0 또는 시장 −6%) · 5일",
                  hyB("B24", T0, "K2", mp_mode="staged", stage2_sig=T0, stage2_drop=0.06), trig="T0", sel="K2",
                  exit="STAGED_FIX", fam_t="T0", fam_x="STAGED"))
    cf.append(Cfg("B25", "B", "F-a · S-RES · 시장 +6% 회복 시 매도",
                  hyB("B25", Fa, "SRES", mp_exit=S2.Exit(hold=10, mkt_tp=0.06)), trig="Fa", sel="SRES", exit="XMKT",
                  hold=10, close_dec=True, fam_t="T0", fam_x="LONG", comps=("B07", "B11", "B22")))
    cf.append(Cfg("B26", "B", "T1 · S-RES · 1+1 분할 · 시장 +6%/−6%",
                  hyB("B26", T1, "SRES", mp_mode="staged", stage2_sig=T1, stage2_drop=0.06,
                      mp_exit=S2.Exit(hold=10, mkt_tp=0.06, mkt_stop=0.06)), trig="T1", sel="SRES", exit="STAGED_XS",
                  hold=10, close_dec=True, fam_t="T1", fam_x="STAGED", comps=("B01", "B11", "B23", "B24")))
    cf.append(Cfg("B27", "B", "T6 반등 확인 · S-RES 목록 · 종가 · 10일",
                  hyB("B27", T6, "T6RES", mp_entry="close", mp_exit=S2.Exit(hold=10)), trig="T6", sel="T6RES",
                  entry="close", hold=10, close_dec=True, fam_t="T6", fam_x="LONG", comps=("B09", "B11", "B21")))

    fix5 = S2.Exit(hold=5, surge=True)
    rec = S2.Exit(hold=10, rec=True, fail_low=True)
    kor_e = {"close": "당일 종가", "open": "다음 날 시가", "open_same": "당일 시가"}
    kor_x = {"FIX": "5일(+급등 매도)", "REC": "5일선 회복·저가 이탈 (최대 10일)"}
    sa_desc = {"SA1": "SA1 잔차 z≤−3.5 & −4%↓", "SA1C1": "SA1 다음 날 확인(저가·종가 안 깨짐)",
               "SA2": "SA2 조용한 급락(거래대금 2배↓·저회전·저변동)", "SA3": "SA3 깨끗한 급락(위험 신호 제외)",
               "SA3C1": "SA3 다음 날 확인", "SA4": "SA4 건강한 추세 속 눌림", "SA5": "SA5 중형 조용한 급락",
               "SA6": "SA6 5일 연속 잔차 하락", "SA7": "SA7 업종 동반 하락", "SA8": "SA8 시가 갭 −4%↓ 되돌림"}
    for cid, sa, en, xk in (("A01", "SA1", "close", "FIX"), ("A02", "SA1", "close", "REC"), ("A03", "SA1", "open", "FIX"),
                            ("A04", "SA1C1", "close", "FIX"), ("A05", "SA2", "close", "FIX"), ("A06", "SA2", "close", "REC"),
                            ("A07", "SA3", "close", "FIX"), ("A08", "SA3", "close", "REC"), ("A09", "SA3", "open", "FIX"),
                            ("A10", "SA3C1", "close", "FIX"), ("A11", "SA4", "close", "FIX"), ("A12", "SA4", "close", "REC"),
                            ("A13", "SA5", "close", "FIX"), ("A14", "SA5", "close", "REC"), ("A15", "SA6", "close", "FIX"),
                            ("A16", "SA6", "close", "REC"), ("A17", "SA7", "close", "FIX"), ("A19", "SA8", "open_same", "FIX")):
        hy = S2.Hybrid(cid, X.base, X.surge, sp_top=X.L(sa), sp_entry=en, sp_exit=fix5 if xk == "FIX" else rec,
                       sp_extra_slip=0.003 if en == "open_same" else 0.0)
        cf.append(Cfg(cid, "A", f"{sa_desc[sa]} · {kor_e[en]} · {kor_x[xk]}", hy, trig=sa, entry=en, exit=xk,
                      hold=5 if xk == "FIX" else 10, close_dec=(en == "close" or xk == "REC"), fam_t=afam(sa), fam_x=xk))
    empty = [np.zeros(0, np.int64)] * T
    cf.append(Cfg("A18", "A", "SA3 단독 (평소 전략 없이 사건만 2종목, 쉴 땐 현금) · 종가 · 5일",
                  S2.Hybrid("A18", empty, X.surge, sp_top=X.L("SA3"), sp_entry="close", sp_exit=fix5, sp_max=2),
                  trig="SA3", entry="close", ref=True, close_dec=True, fam_t="SA3", fam_x="FIX"))
    return cf


# ════════════════════════════════════════════════════════════
# 사건 단위: 트랙 B Delta_switch
# ════════════════════════════════════════════════════════════
def mkt_exit_h(X, t, entry, H, tp, stop, anchor=None):
    """시장 기준 청산: 진입일 뒤 첫 종가 u 에 M_u ≥ M_a(1+tp) (또는 ≤ M_a(1−stop)) 면 그날, 아니면 H 일차. h 반환."""
    M = X.F.M
    a = t if anchor is None else anchor
    te = t + C.ENTRY_DAY[entry]
    last = te + H - 1 if entry not in ("close", "close1", "close2") else te + H
    for u in range(te + 1, min(last, X.T - 1) + 1):
        if (tp and M[u] >= M[a] * (1 + tp)) or (stop and M[u] <= M[a] * (1 - stop)):
            return u - t - C.EXIT_OFF[entry]
    return last - t - C.EXIT_OFF[entry]


def slot_delta(X, t, j, entry, H, exit_kind, which=(0, 1), tp=0.06, stop=0.06):
    """한 슬롯(들) 의 Delta: 바구니 순수익 − 평소 which 종목 비용 전 − 0.35%. j 는 종목 배열."""
    F = X.F
    if exit_kind in ("XMKT", "XMKT_STOP", "STAGED_XS"):
        h = mkt_exit_h(X, t, entry, H, tp, stop if exit_kind != "XMKT" else 0.0)
    else:
        h = H
    if len(j) == 0:
        return np.nan
    net = C.fwd(F, np.full(len(j), t), np.asarray(j), entry, h)[0]
    if not np.isfinite(net).any():
        return np.nan
    b = C.base_gross(F, np.array([t]), entry, h, which)[0]
    if not np.isfinite(b):
        return np.nan
    return float(np.nanmean(net) - b - C.COST)


def b_day_deltas(X, cfg: Cfg, days, H=None, list_override=None):
    """트리거일 각각의 Delta_switch. 분할(STAGED) 은 사건(첫 트리거일) 단위. 반환: (날짜 배열, Delta 배열)."""
    F = X.F
    H = H or cfg.hold
    days = np.asarray(sorted(days))
    if cfg.exit == "IDX":
        out = []
        for t in days:
            if t + H >= X.T:
                out.append(np.nan)
                continue
            g = F.k200_nav[t + H] / F.k200_open[t + 1] - 1 - IDX_COST
            b = C.base_gross(F, np.array([t]), "open", H, (0, 1))[0]
            out.append(g - b - C.COST)
        return days, np.array(out, float)
    L = list_override or X.L(cfg.sel)
    if cfg.exit.startswith("STAGED"):
        xk = "FIX" if cfg.exit == "STAGED_FIX" else "STAGED_XS"
        s2 = np.asarray(cfg.hyb.stage2_sig) if cfg.hyb.stage2_sig is not None else np.zeros(X.T, bool)
        firsts, out = [], []
        until = -1
        for t0 in days:
            if t0 <= until:
                continue
            until = t0 + 10
            c1 = [int(x) for x in L[t0][:1]]
            d1 = slot_delta(X, t0, c1, "open", H, xk, which=(1,))
            d2 = 0.0
            for u in range(t0 + 1, min(t0 + 10, X.T - 2) + 1):
                if s2[u] or F.M[u] <= F.M[t0] * (1 - cfg.hyb.stage2_drop):
                    c2 = [int(x) for x in L[u] if int(x) not in c1][:1]
                    v = slot_delta(X, u, c2, "open", H, xk, which=(1,))
                    d2 = v if np.isfinite(v) else 0.0
                    break
            firsts.append(t0)
            out.append(0.5 * d1 + 0.5 * d2 if np.isfinite(d1) else np.nan)
        return np.array(firsts, int), np.array(out, float)
    out = []
    for t in days:
        jj = [int(x) for x in L[t][:2]]
        out.append(slot_delta(X, t, jj, cfg.entry, H, cfg.exit))
    return days, np.array(out, float)


def trig_days(X, cfg: Cfg, t0, t1):
    d = np.where(X.trig(cfg.trig))[0]
    return d[(d >= t0) & (d < t1)]


def jaccard_T0(X, name, t0, t1):
    a = X.trig(name)[t0:t1]
    b = X.trig("T0")[t0:t1]
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def gate_B(X, cfg: Cfg) -> dict:
    F = X.F
    t0, t1 = C.period_range(F, "IS")
    v0, v1 = C.period_range(F, "VAL")
    days = trig_days(X, cfg, t0, t1)
    vdays = trig_days(X, cfg, v0, v1)
    g = {"n_days_IS": int(len(days)), "N_ep_VAL": int(C.episodes_of(vdays, 10).max() + 1) if len(vdays) else 0}
    dd, dl = b_day_deltas(X, cfg, days)
    ep = C.ep_table(dd, dl, 10)
    g.update({"N_ep_IS": ep.get("N", 0), "ep_mean": ep.get("mean", np.nan), "ep_pos": ep.get("pos", np.nan),
              "ep_median": ep.get("median", np.nan), "ep_worst": ep.get("worst", np.nan), "day_mean": ep.get("day_mean", np.nan)})
    reasons = []
    if cfg.trig in ("T1", "T2", "T3", "T4", "T5", "T7"):
        jac = jaccard_T0(X, cfg.trig, t0, t1)
        g["jaccard_T0"] = jac
        if jac >= 0.6:
            reasons.append(f"T0 과 겹침 {jac:.2f} ≥ 0.6 → T0 에 합침")
    if g["N_ep_IS"] < 6:
        reasons.append(f"IS 사건 {g['N_ep_IS']} < 6 (검증 불가)")
    if g["N_ep_VAL"] < 5:
        reasons.append(f"VAL 사건 {g['N_ep_VAL']} < 5 (검증 불가)")
    if not (g["ep_mean"] > 0):
        reasons.append("IS 사건 평균 Delta ≤ 0")
    if not (g["ep_pos"] >= 0.5):
        reasons.append("IS 사건 중 Delta>0 비율 < 50%")
    if cfg.exit in ("FIX", "IDX"):
        Hs = (3, 7) if cfg.hold == 5 else (7, 15)
        pl = {}
        for H in Hs:
            d2, l2 = b_day_deltas(X, cfg, days, H=H)
            pl[H] = C.ep_table(d2, l2, 10).get("mean", np.nan)
        g["plateau"] = pl
        base = g["ep_mean"]
        if np.isfinite(base) and base > 0 and not all(np.isfinite(v) and v > 0 and v >= 0.5 * base for v in pl.values()):
            reasons.append(f"고원 조건 실패 (H={Hs}: " + ", ".join(C.pct(pl[h], 2) for h in Hs) + ")")
    g["pass"] = not reasons
    g["reasons"] = reasons
    return g


# ════════════════════════════════════════════════════════════
# 사건 단위: 트랙 A
# ════════════════════════════════════════════════════════════
def a_universe(X, sa):
    return X.F.MIDQ if sa == "SA5" else (X.U200p if sa.startswith("SA8") else X.F.U200)


def a_events(X, sa, t0, t1, n=2):
    t, j, _ = C.events_from_top(getattr(X.F, f"top_{sa}"), n, t0, t1)
    return t, j


def matched_placebo_diff(X, t, j, gross, entry, h, uni):
    """같은 날 · 같은 유니버스 · 시총 5분위(전날) · 변동성 3분위(전날) 같고 |z|<1 인 종목 평균과의 차 (비용 전)."""
    F = X.F
    out = np.full(len(t), np.nan)
    for d in np.unique(t):
        mem = np.where(uni[d])[0]
        if len(mem) < 15:
            continue
        cr = F.cr1[d, mem].astype(float)
        vv = F.vol60[d - 1, mem].astype(float)
        q5 = np.floor(np.argsort(np.argsort(cr)) / len(mem) * 5)
        vr = np.argsort(np.argsort(np.where(np.isfinite(vv), vv, np.inf)))
        v3 = np.floor(vr / len(mem) * 3)
        calm = np.abs(np.nan_to_num(F.z[d, mem], nan=9)) < 1
        idx = np.where(t == d)[0]
        evj = set(int(x) for x in j[idx])
        pos = {int(x): i for i, x in enumerate(mem)}
        g_all = None
        for i in idx:
            p = pos.get(int(j[i]))
            if p is None or not np.isfinite(gross[i]):
                continue
            cand = calm & (q5 == q5[p]) & (v3 == v3[p])
            cand &= ~np.isin(mem, list(evj))
            if not cand.any():
                continue
            if g_all is None:
                g_all = C.fwd(F, np.full(len(mem), d), mem, entry, h)[1]
            gc = g_all[cand]
            if np.isfinite(gc).any():
                out[i] = gross[i] - np.nanmean(gc)
    return out


def gate_A(X, sa, entry) -> dict:
    F = X.F
    t0, t1 = C.period_range(F, "IS")
    t, j = a_events(X, sa, t0, t1)
    net, gross = C.fwd(F, t, j, entry, 5)
    uni = a_universe(X, sa)
    st = C.ev_stats(F, t, j, net, uni, entry, 5)
    st.pop("_exc", None)
    b1 = C.base_gross(F, t, entry, 5, (1,))
    dA = net - b1 - C.COST
    u, dm = C.daily_mean(t, dA)
    st["dA_mean"] = float(np.nanmean(dA)) if np.isfinite(dA).any() else np.nan
    st["dA_t"] = C.nw_t(dm, 4)
    pl = matched_placebo_diff(X, t, j, gross, entry, 5, uni)
    st["placebo_diff"] = float(np.nanmean(pl)) if np.isfinite(pl).any() else np.nan
    st["placebo_n"] = int(np.isfinite(pl).sum())
    r = []
    if st.get("n", 0) < 200 or st.get("weeks", 0) < 100:
        r.append(f"사건 {st.get('n', 0)}건 · {st.get('weeks', 0)}주 (< 200건 · 100주)")
    if not (st.get("median", -1) > 0):
        r.append(f"중앙값 {C.pct(st.get('median'), 2)} ≤ 0")
    if not (st.get("excess", -1) > 0 and st.get("t_exc", 0) >= 2.0):
        r.append(f"초과 {C.pct(st.get('excess'), 2)} (t {st.get('t_exc', float('nan')):.1f}) — 양수·t≥2 아님")
    if not (st.get("top5_share", 9) < 0.40):
        r.append("상위 5일 손익 비중 ≥ 40% (또는 총손익 ≤ 0)")
    if not (st.get("maxyear_share", 9) < 0.35):
        r.append("한 해 손익 비중 ≥ 35% (또는 총손익 ≤ 0)")
    if not (st["placebo_diff"] > 0):
        r.append(f"같은 날 대조 종목 대비 {C.pct(st['placebo_diff'], 2)} ≤ 0")
    if not (st["dA_mean"] > 0 and st["dA_t"] >= 2.0):
        r.append(f"평소 보유 대비 {C.pct(st['dA_mean'], 2)} (t {st['dA_t']:.1f}) — 양수·t≥2 아님")
    for era in ("E1", "E2"):
        if not (st.get(f"exc_{era}", -1) > 0):
            r.append(f"{era} 초과 {C.pct(st.get(f'exc_{era}'), 2)} ≤ 0")
    st["pass"] = not r
    st["reasons"] = r
    st["by_year"] = st.get("by_year", {})
    return st


# ════════════════════════════════════════════════════════════
# 대회 시뮬레이션
# ════════════════════════════════════════════════════════════
def run_cfg(X: Ctx, cfg: Cfg, starts, r0=None, control=True):
    """반환 dict: ret, ctrl(대조군), flag(패닉 진입 있었던 창), mp/sp 신호일, 거래."""
    m = X.m_idx if cfg.use_idx else X.m
    n = len(starts)
    ret = np.zeros(n)
    ctrl = np.full(n, np.nan)
    flag = np.zeros(n, bool)
    mp, sp, trades = [], [], []
    halt = 0
    for i, s in enumerate(starts):
        r, info = S2.run_month(m, cfg.hyb, s, 2)
        ret[i] = r
        halt += info["halt_end"]
        f = bool(info["mp_entries"] or info["sp_entries"])
        flag[i] = f
        mp.append(info["mp_entries"])
        sp.append(info["sp_entries"])
        trades.append(info["trades"])
        if control and cfg.track != "R":
            if f and (info["rebal_days"] or info["refill_days"]):
                ctrl[i] = S2.run_month(X.m, X.R0, s, 2, force_rebal=set(info["rebal_days"]),
                                       force_swap=set(info["refill_days"]))[0]
            elif r0 is not None:
                ctrl[i] = r0[i]
    return {"ret": ret, "ctrl": ctrl, "flag": flag, "mp": mp, "sp": sp, "trades": trades, "halt_end": halt}


def cvar10(a):
    a = np.sort(np.asarray(a, float))
    if not len(a):
        return float("nan")
    k = max(1, int(np.ceil(0.1 * len(a))))
    return float(a[:k].mean())


def cstats(res, r0, starts, F) -> dict:
    a = res["ret"]
    f = res["flag"]
    d = a - r0
    out = {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)), "p10d": float((a <= -0.1).mean()),
           "p10u": float((a >= 0.1).mean()), "cvar10": cvar10(a), "worst": float(a.min()), "p_pos": float((a > 0).mean()),
           "share_entry": float(f.mean()), "delta": float(d.mean()),
           "trades_pm": float(np.mean([len(x) for x in res["trades"]])),
           "mean_entry": float(a[f].mean()) if f.any() else float("nan"),
           "mean_noentry": float(a[~f].mean()) if (~f).any() else float("nan"),
           "delta_entry": float(d[f].mean()) if f.any() else float("nan"),
           "cvar10_entry": cvar10(a[f]) if f.sum() >= 5 else float("nan"),
           "cvar10_entry_r0": cvar10(r0[f]) if f.sum() >= 5 else float("nan"),
           "p10d_entry": float((a[f] <= -0.1).mean()) if f.any() else float("nan"),
           "halt_end": int(res["halt_end"])}
    out["J"] = out["mean"] - 0.2 * out["p10d"]
    c = res["ctrl"]
    if np.isfinite(c).all():
        out["delta_ctrl"] = float((a - c).mean())
    last = C.didx(F, "20250901")
    sel = np.array(starts) >= last
    out["last12"] = float(a[sel].mean()) if sel.any() else float("nan")
    return out


# ════════════════════════════════════════════════════════════
# 종가 잡음 · 다음 날 시가 강건성
# ════════════════════════════════════════════════════════════
def perturbed_A_lists(X, sa, rng, t0, t1):
    """종가를 c(1+ε) 로 흔들어 (ε = 종목 N(0, 0.4%) + 공통 N(0, 0.2%)) 사건 조건을 다시 계산한 후보 목록."""
    F = X.F
    a, b = max(t0 - 1, 1), min(t1 + 25, X.T)
    R = b - a
    sl = slice(a, b)
    em = rng.normal(0, 0.002, R)
    e = rng.normal(0, 0.004, (R, X.N)) + em[:, None]
    ret1 = F.ret1[sl].astype(np.float64)
    r1 = (1 + ret1) * (1 + e) - 1
    x = F.x[sl] + em
    r = np.clip(r1, -0.3, 0.3)
    z = (r - F.bstar[sl] * x[:, None]) / F.sig[sl]
    vr = F.vr[sl] * 0.9
    c = F.c[sl].astype(np.float64)
    prev = F.c[a - 1:b - 1].astype(np.float64)
    lim = F.lim[sl][:, None]
    limdown = c * (1 + e) <= prev * (1 - lim + 0.005)
    T0 = X.trig("T0")
    t0r = np.array([T0[max(0, t - 5):t].any() for t in range(a, b)]) | (x <= -0.04)
    ok, U = F.ok[sl], F.U200[sl]
    CF = ok & (x > -0.02)[:, None] & ~t0r[:, None] & (r1 > -0.20) & ~limdown & ~F.divx[sl][:, None] & ~F.bad20[sl]

    if sa in ("SA1", "SA3", "SA1C1", "SA3C1"):
        m = CF & U & (z <= -3.5) & (r1 <= -0.04)
        sc = -z
        if sa in ("SA3", "SA3C1"):
            ret1p = F.ret1[a - 1:b - 1]
            hi1 = (F.c[a - 1:b - 1] / F.hi250[a - 1:b - 1])
            flags = ((ret1p <= -0.10) | (F.ret60[a - 1:b - 1] >= 0.30) | (F.max20[a - 1:b - 1] >= 0.15) | (hi1 <= 0.5)
                     | F.earn[sl])
            intra = (1 + F.intra[sl]) * (1 + e) - 1
            news = (F.gap[sl] <= -0.03) & (intra <= 0.005)
            m = m & ~(r1 <= -0.15) & ~flags & ~news
        if sa.endswith("C1"):
            base_m = np.asarray(getattr(F, f"m_{sa[:3]}")[a - 1:b - 1])
            m = base_m & ok & (F.l[sl] >= F.l[a - 1:b - 1]) & (c * (1 + e) >= prev) & (x > -0.02)[:, None]
            sc = -F.z[a - 1:b - 1].astype(np.float64)
    elif sa == "SA2":
        med_t = np.nanmedian(np.where(U, F.turn1[sl], np.nan), axis=1)[:, None]
        v60p = F.vol60[a - 1:b - 1]
        med_v = np.nanmedian(np.where(U, v60p, np.nan), axis=1)[:, None]
        m = CF & U & (z <= -3) & (r1 <= -0.04) & (vr <= 2) & (F.turn1[sl] <= med_t) & (v60p <= med_v)
        sc = -z
    elif sa == "SA4":
        pre = ((F.cr1[sl] <= 200) & (F.c[a - 1:b - 1] / F.hi250[a - 1:b - 1] >= 0.85) & (F.ret60[a - 1:b - 1] > 0)
               & (F.ret60[a - 1:b - 1] <= 0.40) & (F.max20[a - 1:b - 1] < 0.10) & (F.c[a - 1:b - 1] > F.ma120[a - 1:b - 1]))
        atrp = F.atr14[a - 1:b - 1].astype(np.float64)
        dd = (c * (1 + e) - (F.ma20[sl] + c * e / 20)) / atrp
        m = CF & pre & (dd <= -2.5) & (z <= -2) & (r1 > -0.15)
        sc = -dd
    elif sa == "SA5":
        m = CF & F.MIDQ[sl] & (z <= -3.5) & (r1 <= -0.05) & (vr <= 2)
        sc = -z
    elif sa == "SA6":
        ndp = F.down_days[a - 1:b - 1].astype(np.int32)
        dn = r1 < 0
        nd = np.where(dn, ndp + 1, 0)
        wp = F.worst_streak[a - 1:b - 1].astype(np.float64)
        worst = np.where(dn, np.fmin(np.where(ndp > 0, wp, np.inf), r1), np.nan)
        icum = F.icum[sl].astype(np.float64) + e
        M5 = F.M[sl] * (1 + em) / F.M[a - 5:b - 5] - 1
        m = CF & U & (nd >= 5) & (icum <= -0.06) & (worst >= -0.04) & (M5 > -0.04)[:, None]
        sc = -icum
    elif sa == "SA7":
        secx = F.secx[sl] + em[:, None]
        m = (CF & U & (F.sector >= 0)[None, :] & (r1 <= -0.05) & (secx <= -0.03) & (F.sbreadth[sl] >= 0.5)
             & ((r1 - secx) > -0.05))
        sc = -r1
    else:
        raise ValueError(sa)
    top = PF.top_lists(np.where(m, sc, np.nan))
    full = [np.zeros(0, np.int64)] * X.T
    for i, t in enumerate(range(a, b)):
        row = top[i]
        full[t] = row[row >= 0]
    rec_arr = None
    if True:
        rec_arr = np.zeros((X.T, X.N), bool)
        ee = rng.normal(0, 0.004, (R, X.N)) + rng.normal(0, 0.002, R)[:, None]
        rec_arr[sl] = c * (1 + 0.8 * ee) >= F.ma5[sl]
    return full, rec_arr


def noise_check(X, cfg: Cfg, r0_is, base_delta):
    """종가 결정 설정: 흔든 입력으로 IS Delta 를 50번 다시 재서 평균이 원래의 75% 이상인지."""
    F = X.F
    rng = np.random.default_rng(SEED + sum(ord(ch) for ch in cfg.id))
    t0, t1 = C.period_range(F, "IS")
    st = X.starts["IS"]
    vals = []
    for _ in range(N_NOISE):
        if cfg.track == "A":
            lst, rec_arr = perturbed_A_lists(X, cfg.trig, rng, t0, t1)
            ex = replace(cfg.hyb.sp_exit, rec_arr=rec_arr if cfg.hyb.sp_exit.rec else None)
            hy = replace(cfg.hyb, sp_top=lst, sp_exit=ex)
        else:
            em = rng.normal(0, 0.002, X.T)
            x2 = F.x + em
            hy = cfg.hyb
            if cfg.id in ("B19",):
                sig = x2 <= -0.04
                e = rng.normal(0, 0.004, (t1 - t0 + 25, X.N)) + em[t0:t1 + 25, None]
                r1 = np.full((X.T, X.N), np.nan)
                r1[t0:t1 + 25] = (1 + F.ret1[t0:t1 + 25]) * (1 + e) - 1
                sc = np.where(F.U200, -r1, np.nan)
                top = PF.lists_of(PF.top_lists(sc))
                hy = replace(hy, mp_sig=sig, mp_top=top, mp_episode=X.ep_ids(sig))
            if cfg.trig == "T6":
                T0 = X.trig("T0")
                conf = np.zeros(X.T, bool)
                tt0 = -1
                for t in range(X.T):
                    if T0[t]:
                        tt0 = t
                        continue
                    if tt0 >= 0 and t - tt0 > 10:
                        tt0 = -1
                    if tt0 >= 0 and x2[t] >= 0.01:
                        conf[t] = True
                        tt0 = -1
                # 목록은 원래 T6 목록 (고정 목록 순서는 u* 종가 기준이라 그대로 둔다); 새 확인일엔 K2 기준 재구성
                base_l = X.L(cfg.sel)
                K2l = X.L("K2")
                lst = [base_l[t] if len(base_l[t]) else K2l[t] for t in range(X.T)]
                hy = replace(hy, mp_sig=conf, mp_top=lst, mp_episode=X.ep_ids(conf))
            if cfg.exit in ("XMKT", "XMKT_STOP", "STAGED_XS"):
                mk = replace(X.m, ind={**X.m.ind, "mkt": F.M * (1 + em)})
                vals.append(np.mean([S2.run_month(mk, hy, s, 2)[0] for s in st]) - np.mean(r0_is))
                continue
        m = X.m_idx if cfg.use_idx else X.m
        vals.append(np.mean([S2.run_month(m, hy, s, 2)[0] for s in st]) - np.mean(r0_is))
    v = np.array(vals)
    return {"mean": float(v.mean()), "min": float(v.min()), "ratio": float(v.mean() / base_delta) if base_delta else float("nan"),
            "pass": bool(base_delta > 0 and v.mean() >= 0.75 * base_delta)}


def next_open_variant(cfg: Cfg) -> Cfg:
    """종가 결정 → 다음 날 시가로 옮긴 설정 (진입 · 종가 청산 모두)."""
    hy = cfg.hyb
    if cfg.track == "A":
        hy = replace(hy, sp_entry="open" if hy.sp_entry == "close" else hy.sp_entry,
                     sp_exit=replace(hy.sp_exit, defer_all=True))
    else:
        hy = replace(hy, mp_entry="open", mp_exit=replace(hy.mp_exit, defer_all=hy.mp_exit.mkt_tp > 0 or hy.mp_exit.mkt_stop > 0))
    return replace(cfg, id=cfg.id + "o", hyb=hy)


# ════════════════════════════════════════════════════════════
# 위약 (placebo)
# ════════════════════════════════════════════════════════════
def placebo_pool(X, t0, t1):
    """시장 위약일 후보: x > −2%, 어떤 T0 일과도 21 거래일 이상 떨어짐, 범위 [t0, t1)."""
    F = X.F
    T0 = np.where(X.trig("T0"))[0]
    days = []
    for u in range(max(t0, 260), t1):
        if F.x[u] <= -0.02 or X.vol_terc[u] < 0:
            continue
        if len(T0) and np.min(np.abs(T0 - u)) < 21:
            continue
        days.append(u)
    return np.array(days, int)


def rc_B(X, cfgs, real, t0, t1, n_draw=N_RC):
    """IS 현실성 검사: 각 설정의 IS 사건 첫날마다 (변동성 삼분위 · 200일선) 가 같은 위약일 하나를 뽑아
    같은 규칙으로 Delta 를 재고, 추첨마다 설정들 중 최고값을 모은다. 반환: 최고값 분포의 90% 분위."""
    pool = placebo_pool(X, t0, t1)
    cls = X.vol_terc[pool] * 2 + X.ab200[pool]
    rng = np.random.default_rng(SEED)
    per_cfg = []
    for cfg in cfgs:
        days = trig_days(X, cfg, t0, t1)
        dd, _ = b_day_deltas(X, cfg, days)
        ep = C.episodes_of(dd, 10) if len(dd) else np.zeros(0, int)
        firsts = [dd[ep == e][0] for e in range(ep.max() + 1)] if len(dd) else []
        lo = X.L("K2") if cfg.sel in ("T6K2", "T6RES") else None
        if cfg.exit.startswith("STAGED"):          # 분할은 사건 단위로 묶이므로 위약일마다 따로 잰다
            dpool = np.array([b_day_deltas(X, cfg, [u], list_override=lo)[1][0] for u in pool])
        else:
            _, dpool = b_day_deltas(X, cfg, pool, list_override=lo)
        per_cfg.append((firsts, dpool))
    best = np.full(n_draw, -np.inf)
    for i in range(n_draw):
        for firsts, dpool in per_cfg:
            vals = []
            for f in firsts:
                c = X.vol_terc[f] * 2 + X.ab200[f]
                cand = np.where((cls == c) & np.isfinite(dpool))[0]
                if not len(cand):
                    cand = np.where(np.isfinite(dpool))[0]
                vals.append(dpool[rng.choice(cand)])
            if vals:
                best[i] = max(best[i], float(np.mean(vals)))
    return float(np.quantile(best, 0.9)), best


def rc_A(X, keys, t0, t1, n_draw=N_RC):
    """트랙 A 현실성 검사: 같은 사건 날짜에 U200 중 ret1 ≤ 0 & |z| < 1 인 무작위 종목으로 바꿔 평소 보유 대비 Delta 평균,
    추첨마다 설정 중 최고값. 반환: 90% 분위."""
    F = X.F
    rng = np.random.default_rng(SEED + 1)
    cache = {}
    per = []
    for sa, entry in keys:
        t, j = a_events(X, sa, t0, t1)
        net, _ = C.fwd(F, t, j, entry, 5)
        b1 = C.base_gross(F, t, entry, 5, (1,))
        ok = np.isfinite(net) & np.isfinite(b1)
        t, b1 = t[ok], b1[ok]
        cands = []
        for d in t:
            key = (int(d), entry)
            if key not in cache:
                mem = np.where(F.U200[d] & (np.nan_to_num(F.ret1[d], nan=1) <= 0) & (np.abs(np.nan_to_num(F.z[d], nan=9)) < 1))[0]
                g = C.fwd(F, np.full(len(mem), d), mem, entry, 5)[1] if len(mem) else np.zeros(0)
                cache[key] = g[np.isfinite(g)]
            cands.append(cache[key])
        per.append((b1, cands))
    best = np.full(n_draw, -np.inf)
    for i in range(n_draw):
        for b1, cands in per:
            g = np.array([c[rng.integers(len(c))] if len(c) else np.nan for c in cands])
            v = g - C.COST - b1 - C.COST
            if np.isfinite(v).any():
                best[i] = max(best[i], float(np.nanmean(v)))
    return float(np.quantile(best, 0.9)), best


# ════════════════════════════════════════════════════════════
# 동결
# ════════════════════════════════════════════════════════════
def sha_files():
    h = {}
    for f in FILES:
        p = os.path.join(E.ROOT, f)
        h[f] = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    h["data/lab/panic2/meta.json"] = hashlib.sha256(open(os.path.join(PF.DIR, "meta.json"), "rb").read()).hexdigest()
    return h


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=E.ROOT, text=True).strip()
    except Exception:
        return "?"


def check_freeze():
    if not os.path.exists(FREEZE):
        sys.exit("동결 파일이 없다: 먼저 `python -m backtest.lab.panic2 is`")
    fz = json.load(open(FREEZE, encoding="utf-8"))
    now = sha_files()
    diff = [k for k in now if fz["sha256"].get(k) != now[k]]
    if diff:
        sys.exit(f"동결 이후 바뀐 파일: {diff} — VAL/TEST 는 거부 (새 사전 등록이 필요)")
    return fz


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()]
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(conv(obj), fh, ensure_ascii=False, indent=1)


# ════════════════════════════════════════════════════════════
# P0 · P1 · P2 (IS)
# ════════════════════════════════════════════════════════════
def phase_is(dry=False):
    t_start = time.time()
    X = Ctx()
    F = X.F
    cfgs = build_configs(X)
    byid = {c.id: c for c in cfgs}
    out = {"phase": "IS", "configs": {}, "gates_A": {}, "gates_B": {}}
    # P0 점검
    yrs = np.array([d[:4] for d in F.dates])
    out["trigger_days_per_year"] = {n: {y: int(X.trig(n)[yrs == y].sum()) for y in sorted(set(yrs))}
                                    for n in ("T0", "T1", "T2", "T3", "T4", "T5", "T7", "Fa", "Fc", "T6")}
    t0, t1 = C.period_range(F, "IS")
    out["jaccard_T0_IS"] = {n: jaccard_T0(X, n, t0, t1) for n in ("T1", "T2", "T3", "T4", "T5", "T7")}
    lab_cov = {}
    for y in ("2011", "2014", "2018", "2021", "2024", "2026"):
        i = C.didx(F, f"{y}0601")
        u = F.U200[i]
        lab_cov[y] = float((F.sector[u] >= 0).mean()) if u.any() else float("nan")
    out["sector_coverage_U200"] = lab_cov
    # 신호 없는 창 Delta = 0 점검 (하이브리드 전부)
    st = X.starts["IS"]
    r0res = run_cfg(X, byid["R0"], st, control=False)
    r0 = r0res["ret"]
    r0res["ctrl"] = r0.copy()
    bad = {}
    for c in cfgs:
        if c.track == "R" and c.id == "R0":
            continue
        if c.id == "A18":
            continue
        sigs = np.zeros(X.T, bool)
        if c.hyb.mp_sig is not None:
            sigs |= np.asarray(c.hyb.mp_sig)
        if c.hyb.sp_top is not None:
            sigs |= np.array([len(x) > 0 for x in c.hyb.sp_top])
        n_bad = 0
        for i, s in enumerate(st[::7]):
            e = min(s + E.MONTH - 1, X.T - 1)
            if sigs[s - 1:e].any() or (c.hyb.sp_entry == "open_same" and sigs[s:e + 1].any()):
                continue
            m = X.m_idx if c.use_idx else X.m
            if abs(S2.run_month(m, c.hyb, s, 2)[0] - r0[st.index(s)]) > 1e-12:
                n_bad += 1
        bad[c.id] = n_bad
    out["zero_delta_violations"] = bad
    if any(bad.values()):
        print("P0 실패: 신호 없는 창에서 Delta ≠ 0", {k: v for k, v in bad.items() if v})
        if not dry:
            sys.exit(1)
    print(f"P0 {time.time() - t_start:.0f}s", flush=True)

    # P1 사건 조사 (IS)
    if not dry:
        try:
            from backtest.lab import panic2_ev_a, panic2_ev_b
            la, sa_ = panic2_ev_a.run(F, "IS")
            lb, sb_ = panic2_ev_b.run(F, "IS")
            out["events_IS"] = {"A_lines": la, "B_lines": lb, "A_summary": sa_, "B_summary": sb_}
        except Exception as ex:  # 사건 조사 실패는 기록만 (관문 판정은 아래에서 따로 한다)
            out["events_IS"] = {"error": repr(ex)}
        print(f"P1 {time.time() - t_start:.0f}s", flush=True)

    # 관문
    akeys = sorted({(c.trig, c.entry) for c in cfgs if c.track == "A"})
    for sa, en in akeys:
        out["gates_A"][f"{sa}|{en}"] = gate_A(X, sa, en)
        print("gate A", sa, en, out["gates_A"][f"{sa}|{en}"]["pass"], flush=True)
    for c in cfgs:
        if c.track == "B" or c.id == "R1":
            out["gates_B"][c.id] = gate_B(X, c)
            print("gate B", c.id, out["gates_B"][c.id]["pass"], out["gates_B"][c.id]["reasons"], flush=True)
    print(f"관문 {time.time() - t_start:.0f}s", flush=True)

    # P2 IS 대회
    res = {}
    for c in cfgs:
        if c.id == "R0":
            res[c.id] = r0res
        else:
            res[c.id] = run_cfg(X, c, st, r0)
        out["configs"][c.id] = {"track": c.track, "desc": c.desc, "IS": cstats(res[c.id], r0, st, F)}
    print(f"P2 대회 {time.time() - t_start:.0f}s", flush=True)
    R0s = out["configs"]["R0"]["IS"]

    # 적격 판정
    elig = {}
    for c in cfgs:
        if c.track not in ("A", "B") or c.ref:
            continue
        s_ = out["configs"][c.id]["IS"]
        r = []
        if c.track == "A":
            g = out["gates_A"][f"{c.trig}|{c.entry}"]
            if not g["pass"]:
                r.append("사건 관문 실패")
        else:
            g = out["gates_B"][c.id]
            if not g["pass"]:
                r.append("사건 관문 실패")
            for comp in c.comps:
                if not out["gates_B"][comp]["pass"]:
                    r.append(f"구성 요소 {comp} 관문 실패 → 조합 안 돌림")
        if not (s_["J"] > R0s["J"]):
            r.append("J_IS ≤ R0")
        if not (s_["p10d"] <= R0s["p10d"] + 0.02):
            r.append("P10d_IS > R0 + 2pp")
        if np.isfinite(s_["cvar10_entry"]) and not (s_["cvar10_entry"] >= s_["cvar10_entry_r0"] - 0.02):
            r.append("진입 창 CVaR10 < R0 − 2pp")
        elig[c.id] = r
    # 종가 결정 설정: 잡음 · 다음 날 시가 강건성 (다른 조건을 통과한 것만)
    rob = {}
    for c in cfgs:
        if c.id not in elig or elig[c.id] or not c.close_dec:
            continue
        base_delta = out["configs"][c.id]["IS"]["delta"]
        nz = noise_check(X, c, r0, base_delta)
        v = next_open_variant(c)
        rv = run_cfg(X, v, st, r0, control=False)["ret"]
        dv = float((rv - r0).mean())
        rob[c.id] = {"noise": nz, "next_open_delta": dv, "next_open_pass": dv > 0}
        if not nz["pass"]:
            elig[c.id].append(f"종가 잡음 뒤 Delta {nz['ratio'] * 100:.0f}% < 75%")
        if not dv > 0:
            elig[c.id].append(f"다음 날 시가로 옮기면 Delta {C.pct(dv, 2)} (부호 바뀜)")
        print("robust", c.id, rob[c.id], flush=True)
    out["robustness"] = rob
    # 현실성 검사 (적격 설정들 중 최고값의 위약 분포)
    for trk in ("B", "A"):
        ok_ids = [cid for cid, r in elig.items() if not r and byid[cid].track == trk]
        if not ok_ids:
            out[f"rc_{trk}"] = {"n_cfg": 0}
            continue
        if trk == "B":
            q90, _ = rc_B(X, [byid[c] for c in ok_ids], None, t0, t1)
            real = {cid: out["gates_B"][cid]["ep_mean"] for cid in ok_ids}
        else:
            keys = sorted({(byid[c].trig, byid[c].entry) for c in ok_ids})
            q90, _ = rc_A(X, keys, t0, t1)
            real = {cid: out["gates_A"][f"{byid[cid].trig}|{byid[cid].entry}"]["dA_mean"] for cid in ok_ids}
        out[f"rc_{trk}"] = {"n_cfg": len(ok_ids), "q90": q90, "real": real}
        for cid in ok_ids:
            if not real[cid] > q90:
                elig[cid].append(f"현실성 검사: {C.pct(real[cid], 2)} ≤ 위약 최고값 90% {C.pct(q90, 2)}")
        print("rc", trk, out[f"rc_{trk}"], flush=True)
    out["eligibility"] = elig
    # 결선 (트랙마다 J_IS 상위 3; B 는 트리거 계열 · 청산 계열마다 하나)
    fin = {}
    for trk in ("B", "A"):
        cand = sorted([cid for cid, r in elig.items() if not r and byid[cid].track == trk],
                      key=lambda c: -out["configs"][c]["IS"]["J"])
        pick, ft, fx = [], set(), set()
        for cid in cand:
            c = byid[cid]
            if trk == "B" and (c.fam_t in ft or c.fam_x in fx):
                continue
            if trk == "A" and c.fam_t in ft:
                continue
            pick.append(cid)
            ft.add(c.fam_t)
            fx.add(c.fam_x)
            if len(pick) == 3:
                break
        fin[trk] = pick
    ab = None
    if fin["B"] and fin["A"]:
        ab = {"B": fin["B"][0], "A": fin["A"][0]}
    out["finalists"] = fin
    out["AB"] = ab
    out["R0_IS"] = R0s
    out["elapsed_s"] = time.time() - t_start
    if dry:
        return out, X, cfgs, res
    jdump(out, ISJ)
    freeze = {"git_head": git_head(), "sha256": sha_files(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "finalists": fin, "AB": ab, "eligibility": elig, "rc": {k: out[k] for k in ("rc_A", "rc_B")},
              "R0_IS": R0s, "IS_stats": {k: v["IS"] for k, v in out["configs"].items()},
              "gates_A": {k: {kk: vv for kk, vv in v.items() if kk in ("pass", "reasons", "n", "mean", "median", "excess",
                                                                       "t_exc", "dA_mean", "dA_t", "placebo_diff")}
                          for k, v in out["gates_A"].items()},
              "gates_B": {k: {kk: vv for kk, vv in v.items() if kk in ("pass", "reasons", "N_ep_IS", "N_ep_VAL", "ep_mean",
                                                                       "ep_pos", "plateau", "jaccard_T0")}
                          for k, v in out["gates_B"].items()}}
    jdump(freeze, FREEZE)
    print("동결:", FREEZE, "결선:", fin, "AB:", ab, f"{time.time() - t_start:.0f}s")
    return out, X, cfgs, res


# ════════════════════════════════════════════════════════════
# P3 (VAL) · P4 (TEST)
# ════════════════════════════════════════════════════════════
def make_ab(X, byid, ab):
    b, a = byid[ab["B"]], byid[ab["A"]]
    hy = replace(b.hyb, name="AB", sp_top=a.hyb.sp_top, sp_entry=a.hyb.sp_entry, sp_exit=a.hyb.sp_exit,
                 sp_extra_slip=a.hyb.sp_extra_slip)
    return Cfg("AB", "AB", f"조합: {b.id} + {a.id}", hy, trig=b.trig, sel=b.sel, entry=b.entry, exit=b.exit, hold=b.hold,
               use_idx=b.use_idx)


def val_criteria_B(X, cfg, res, r0, starts, is_res, is_r0, is_starts) -> dict:
    F = X.F
    d = res["ret"] - r0
    dc = res["ret"] - res["ctrl"]
    ep0 = X.ep_ids(X.trig(cfg.trig))
    epi = np.full(X.T, -1)                      # 신호일 → 사건 번호 (t+2 진입 설정은 신호일이 트리거 다음 날)
    if cfg.entry == "open2":
        epi[1:] = ep0[:-1]
    else:
        epi[:] = ep0
    out = {"delta": float(d.mean()), "delta_ctrl": float(dc.mean())}
    # (iii) 사건 하나씩 빼기 (VAL)
    eps = sorted({int(epi[x]) for L in res["mp"] for x in L if epi[x] >= 0})
    loo, looc = [], []
    for e_ in eps:
        keep = np.array([not any(epi[x] == e_ for x in L) for L in res["mp"]])
        if keep.any():
            loo.append(float(d[keep].mean()))
            looc.append(float(dc[keep].mean()))
    out["loo_min"] = min(loo) if loo else float("nan")
    out["loo_min_ctrl"] = min(looc) if looc else float("nan")
    # (iv) IS+VAL 합쳐 사건별 Delta
    allp = [(is_res, is_r0), (res, r0)]
    ep_d = {}
    for rr, rz in allp:
        dd = rr["ret"] - rz
        for i, L in enumerate(rr["mp"]):
            es = [int(epi[x]) for x in L if epi[x] >= 0]
            if es:
                ep_d.setdefault(es[0], []).append(dd[i])
    em = np.array([np.mean(v) for v in ep_d.values()]) if ep_d else np.zeros(0)
    out["pooled_ep_n"] = int(len(em))
    out["pooled_ep_pos"] = float((em > 0).mean()) if len(em) else float("nan")
    out["pooled_no_top2"] = float(np.sort(em)[:-2].mean()) if len(em) > 2 else float("nan")
    out["pooled_ep_means"] = em.tolist()
    lo, hi = C.boot_ci(em, 5000, (0.05, 0.95), SEED) if len(em) > 1 else (float("nan"), float("nan"))
    out["pooled_ep_ci90"] = [lo, hi]
    # (iii) 대체 청산 간격 G=5 · 21 (진단)
    for G in (5, 21):
        idx = np.where(X.trig(cfg.trig))[0]
        e2 = np.full(X.T, -1)
        e2[idx + (1 if cfg.entry == "open2" else 0)] = C.episodes_of(idx, G)
        vals = []
        for e_ in sorted({int(e2[x]) for L in res["mp"] for x in L if e2[x] >= 0}):
            keep = np.array([not any(e2[x] == e_ for x in L) for L in res["mp"]])
            if keep.any():
                vals.append(float(d[keep].mean()))
        out[f"loo_min_G{G}"] = min(vals) if vals else float("nan")
    return out


def val_criteria_A(X, cfg, res, r0, starts, is_res, is_r0, is_starts) -> dict:
    F = X.F
    d = res["ret"] - r0
    dc = res["ret"] - res["ctrl"]
    out = {"delta": float(d.mean()), "delta_ctrl": float(dc.mean())}
    yrs = np.array([F.dates[s][:4] for s in starts])
    out["loyo_min"] = min(float(d[yrs != y].mean()) for y in sorted(set(yrs)))
    out["loyo_min_ctrl"] = min(float(dc[yrs != y].mean()) for y in sorted(set(yrs)))
    tr = [(t0, r) for L in res["trades"] for (k, t0, t1, r) in L if k == "sp"]
    top5 = set(t for t, _ in sorted(tr, key=lambda x: -x[1])[:5])
    keep = np.array([not any(t0 in top5 for (k, t0, t1, r) in L if k == "sp") for L in res["trades"]])
    out["drop_top5_delta"] = float(d[keep].mean()) if keep.any() else float("nan")
    q = np.array([F.dates[s][:4] + "Q" + str((int(F.dates[s][4:6]) - 1) // 3 + 1) for s in starts])
    qm = [d[q == x].mean() for x in sorted(set(q))]
    out["quarters_pos"] = float(np.mean(np.array(qm) > 0))
    return out


def placebo_B_contest(X, cfg, r0_pool, starts_pool, n_draw=N_PLACEBO):
    """VAL (v): 같은 해 · 같은 (변동성 삼분위, 200일선) 인 위약일에 같은 규칙을 걸어 IS+VAL 전체 평균 Delta 분포."""
    F = X.F
    a, _ = C.period_range(F, "IS")
    _, b = C.period_range(F, "VAL")
    pool = placebo_pool(X, a, b)
    cls = X.vol_terc[pool] * 2 + X.ab200[pool]
    pyr = np.array([F.dates[u][:4] for u in pool])
    real_days = [t for t in np.where(X.trig(cfg.trig))[0] if a <= t < b]
    rng = np.random.default_rng(SEED + 7)
    st = np.array(starts_pool)
    out = np.zeros(n_draw)
    lo = X.L("K2") if cfg.sel in ("T6K2", "T6RES") else X.L(cfg.sel) if cfg.sel not in ("IDX",) else None
    for i in range(n_draw):
        pdays = []
        for t in real_days:
            c = X.vol_terc[t] * 2 + X.ab200[t]
            cand = np.where((cls == c) & (pyr == F.dates[t][:4]))[0]
            if not len(cand):
                cand = np.where(cls == c)[0]
            if not len(cand):
                cand = np.arange(len(pool))
            pdays.append(int(pool[rng.choice(cand)]))
        sig = np.zeros(X.T, bool)
        sig[pdays] = True
        hy = cfg.hyb
        if cfg.entry == "open2":
            s2 = np.zeros(X.T, bool)
            s2[1:] = sig[:-1]
            sig = s2
        kw = dict(mp_sig=sig, mp_episode=X.ep_ids(sig))
        if lo is not None:
            kw["mp_top"] = lo
        if hy.mp_mode == "staged":
            kw["stage2_sig"] = sig
        hy = replace(hy, **kw)
        m = X.m_idx if cfg.use_idx else X.m
        dsum = 0.0
        for k_, s in enumerate(st):
            e = min(s + E.MONTH - 1, X.T - 1)
            if sig[s - 1:e].any():
                dsum += S2.run_month(m, hy, int(s), 2)[0] - r0_pool[k_]
        out[i] = dsum / len(st)
    return out


def placebo_A_contest(X, cfg, r0_pool, starts_pool, n_draw=N_PLACEBO):
    """VAL (v) 트랙 A: 같은 사건 날짜에 U200 중 ret1 ≤ 0 & |z| < 1 무작위 한 종목."""
    F = X.F
    a, _ = C.period_range(F, "IS")
    _, b = C.period_range(F, "VAL")
    L = X.L(cfg.trig)
    days = [t for t in range(a, b) if len(L[t])]
    cands = {}
    for d in days:
        dd = d if cfg.entry == "open_same" else d
        cands[d] = np.where(F.U200[dd] & (np.nan_to_num(F.ret1[dd], nan=1) <= 0) & (np.abs(np.nan_to_num(F.z[dd], nan=9)) < 1))[0]
    rng = np.random.default_rng(SEED + 11)
    out = np.zeros(n_draw)
    for i in range(n_draw):
        lst = [np.zeros(0, np.int64)] * X.T
        for d in days:
            c = cands[d]
            if len(c):
                lst[d] = np.array([c[rng.integers(len(c))]])
        hy = replace(cfg.hyb, sp_top=lst)
        rr = np.array([S2.run_month(X.m, hy, int(s), 2)[0] for s in starts_pool])
        out[i] = float((rr - r0_pool).mean())
    return out


def phase_val():
    fz = check_freeze()
    t_start = time.time()
    X = Ctx()
    F = X.F
    cfgs = build_configs(X)
    byid = {c.id: c for c in cfgs}
    if fz["AB"]:
        byid["AB"] = make_ab(X, byid, fz["AB"])
        cfgs.append(byid["AB"])
    out = {"phase": "VAL", "configs": {}, "criteria": {}}
    st = X.starts["VAL"]
    sti = X.starts["IS"]
    r0vres = run_cfg(X, byid["R0"], st, control=False)
    r0v = r0vres["ret"]
    r0vres["ctrl"] = r0v.copy()
    r0i = run_cfg(X, byid["R0"], sti, control=False)["ret"]
    resv, resi = {}, {}
    for c in cfgs:
        if c.id == "R0":
            continue
        resv[c.id] = run_cfg(X, c, st, r0v)
        out["configs"][c.id] = {"VAL": cstats(resv[c.id], r0v, st, F)}
    out["configs"]["R0"] = {"VAL": cstats(r0vres, r0v, st, F)}
    R0v = out["configs"]["R0"]["VAL"]
    print(f"VAL 대회 {time.time() - t_start:.0f}s", flush=True)
    fins = fz["finalists"]["B"] + fz["finalists"]["A"] + (["AB"] if fz["AB"] else [])
    for cid in fins:
        c = byid[cid]
        resi[cid] = run_cfg(X, c, sti, r0i)
        s_ = out["configs"][cid]["VAL"]
        crit = {}
        crit["i"] = s_["delta"] >= 0.003 and s_.get("delta_ctrl", -1) >= 0.003
        crit["ii"] = s_["p10d"] <= R0v["p10d"] + 0.01
        crit["cvar"] = (not np.isfinite(s_["cvar10_entry"])) or s_["cvar10_entry"] >= s_["cvar10_entry_r0"] - 0.02
        pool_st = sti + st
        r0p = np.concatenate([r0i, r0v])
        real_pool = float(np.concatenate([resi[cid]["ret"], resv[cid]["ret"]]).mean() - r0p.mean())
        if c.track in ("B", "AB"):
            vb = val_criteria_B(X, c, resv[cid], r0v, st, resi[cid], r0i, sti)
            crit.update(vb)
            crit["iii"] = vb["loo_min"] > 0 and vb["loo_min_ctrl"] > 0
            crit["iv"] = vb["pooled_no_top2"] > 0 and vb["pooled_ep_pos"] >= 0.5
            pb = placebo_B_contest(X, c, r0p, pool_st)
        else:
            va = val_criteria_A(X, c, resv[cid], r0v, st, resi[cid], r0i, sti)
            crit.update(va)
            crit["iii"] = va["loyo_min"] > 0 and va["loyo_min_ctrl"] > 0 and va["drop_top5_delta"] > 0
            crit["iv"] = va["quarters_pos"] >= 0.55
            pb = placebo_A_contest(X, c, r0p, pool_st)
        crit["placebo_q90"] = float(np.quantile(pb, 0.9))
        crit["pooled_delta"] = real_pool
        crit["v"] = real_pool >= crit["placebo_q90"]
        if cid == "AB":
            b, a = fz["AB"]["B"], fz["AB"]["A"]
            best = max(out["configs"][b]["VAL"]["mean"], out["configs"][a]["VAL"]["mean"])
            bp = min(out["configs"][b]["VAL"]["p10d"], out["configs"][a]["VAL"]["p10d"])
            crit["vi"] = s_["mean"] >= best - 0.002 and s_["p10d"] <= bp + 0.01
        pooled_J = float(np.concatenate([resi[cid]["ret"], resv[cid]["ret"]]).mean()
                         - 0.2 * (np.concatenate([resi[cid]["ret"], resv[cid]["ret"]]) <= -0.1).mean())
        crit["pooled_J"] = pooled_J
        keys = ["i", "ii", "iii", "iv", "v", "cvar"] + (["vi"] if cid == "AB" else [])
        crit["pass"] = all(bool(crit[k]) for k in keys)
        out["criteria"][cid] = crit
        print("VAL", cid, {k: crit[k] for k in keys}, f"{time.time() - t_start:.0f}s", flush=True)
    adopt = {}
    for trk, ids in (("B", fz["finalists"]["B"]), ("A", fz["finalists"]["A"])):
        ok_ = [c for c in ids if out["criteria"][c]["pass"]]
        adopt[trk] = max(ok_, key=lambda c: out["criteria"][c]["pooled_J"]) if ok_ else None
    if fz["AB"] and out["criteria"].get("AB", {}).get("pass"):
        adopt["AB"] = "AB"
    out["adopt_after_VAL"] = adopt
    out["R0_VAL"] = R0v
    # 사건 조사 VAL 열
    try:
        from backtest.lab import panic2_ev_a, panic2_ev_b
        la, sa_ = panic2_ev_a.run(F, "VAL")
        lb, sb_ = panic2_ev_b.run(F, "VAL")
        out["events_VAL"] = {"A_lines": la, "B_lines": lb, "A_summary": sa_, "B_summary": sb_}
    except Exception as ex:
        out["events_VAL"] = {"error": repr(ex)}
    out["elapsed_s"] = time.time() - t_start
    jdump(out, VALJ)
    print("VAL 끝:", adopt, f"{time.time() - t_start:.0f}s")


def nested_loeo(X, cfgs, t_end):
    """중첩 시간순 사건 하나 빼기: 각 사건 e 에서 그 전 사건들(5개 이상) 로 최고 설정을 골라 e 에서의 Delta 를 기록."""
    F = X.F
    a = C.didx(F, C.PERIODS["IS"][0])
    union = np.zeros(X.T, bool)
    per = {}
    for c in cfgs:
        d = np.where(X.trig(c.trig))[0]
        d = d[(d >= a) & (d < t_end)]
        union[d] = True
        dd, dl = b_day_deltas(X, c, d)
        per[c.id] = dict(zip(dd.tolist(), dl.tolist()))
    days = np.where(union)[0]
    ep = C.episodes_of(days, 10)
    E_ = ep.max() + 1 if len(days) else 0
    D = np.zeros((len(cfgs), E_))
    for k, c in enumerate(cfgs):
        for e in range(E_):
            v = [per[c.id][t] for t in days[ep == e] if t in per[c.id] and np.isfinite(per[c.id][t])]
            D[k, e] = np.mean(v) if v else 0.0
    ooe, ins, picks, when = [], [], [], []
    for e in range(5, E_):
        tr = D[:, :e].mean(1)
        k = int(np.argmax(tr))
        ooe.append(D[k, e])
        ins.append(tr[k])
        picks.append(cfgs[k].id)
        when.append(F.dates[days[ep == e][0]])
    ooe = np.array(ooe)
    from collections import Counter
    cc = Counter(picks)
    return {"n_folds": len(ooe), "ooe_mean": float(ooe.mean()) if len(ooe) else float("nan"),
            "ooe_median": float(np.median(ooe)) if len(ooe) else float("nan"),
            "optimism_gap": float(np.mean(ins) - ooe.mean()) if len(ooe) else float("nan"),
            "stability": float(cc.most_common(1)[0][1] / len(picks)) if picks else float("nan"),
            "picks": list(zip(when, picks, ooe.tolist()))}


def module_value(X, cfg, ooe_mean):
    """P(시작 전날 변동성 삼분위가 지금과 같을 때 한 달 안에 트리거) × 사건 밖 평균 Delta."""
    F = X.F
    trig = X.trig(cfg.trig)
    a = C.didx(F, C.PERIODS["IS"][0])
    v = F.mkt_vol20
    w = v[X.T - 751:X.T - 1]
    q1, q2 = np.quantile(w[np.isfinite(w)], [1 / 3, 2 / 3])
    now = 0 if v[X.T - 1] <= q1 else (1 if v[X.T - 1] <= q2 else 2)       # 2026-09-23 종가 기준
    hits, n = 0, 0
    for s in range(a, X.T - E.MONTH):
        if X.vol_terc[s] != now:
            continue
        n += 1
        hits += bool(trig[s - 1:s + E.MONTH - 1].any())
    p = hits / n if n else float("nan")
    return {"tercile_now": now, "p_trigger": p, "ooe_mean": ooe_mean, "value": p * ooe_mean if np.isfinite(ooe_mean) else float("nan")}


def phase_test():
    fz = check_freeze()
    if not os.path.exists(VALJ):
        sys.exit("VAL 결과가 없다: 먼저 `python -m backtest.lab.panic2 val`")
    vj = json.load(open(VALJ, encoding="utf-8"))
    t_start = time.time()
    X = Ctx()
    F = X.F
    cfgs = build_configs(X)
    byid = {c.id: c for c in cfgs}
    if fz["AB"]:
        byid["AB"] = make_ab(X, byid, fz["AB"])
        cfgs.append(byid["AB"])
    st = X.starts["TEST"]
    r0res = run_cfg(X, byid["R0"], st, control=False)
    r0 = r0res["ret"]
    r0res["ctrl"] = r0.copy()
    out = {"phase": "TEST", "configs": {}}
    res = {}
    for c in cfgs:
        if c.id == "R0":
            continue
        res[c.id] = run_cfg(X, c, st, r0)
        out["configs"][c.id] = {"TEST": cstats(res[c.id], r0, st, F)}
    out["configs"]["R0"] = {"TEST": cstats(r0res, r0, st, F)}
    R0t = out["configs"]["R0"]["TEST"]
    # 2026-03 사건 뺀 TEST
    a26, b26 = C.didx(F, "20260225"), C.didx(F, "20260415")
    for cid, rr in res.items():
        keep = np.array([not any(a26 <= x < b26 for x in (rr["mp"][i] + rr["sp"][i])) for i in range(len(st))])
        out["configs"][cid]["TEST"]["delta_ex_2026_03"] = float((rr["ret"] - r0)[keep].mean()) if keep.any() else float("nan")
    veto = {}
    for cid in fz["finalists"]["B"] + fz["finalists"]["A"] + (["AB"] if fz["AB"] else []):
        s_ = out["configs"][cid]["TEST"]
        veto[cid] = {"pass": s_["mean"] >= R0t["mean"] and s_["p10d"] <= R0t["p10d"] + 0.02,
                     "mean": s_["mean"], "p10d": s_["p10d"]}
    out["veto"] = veto
    # 중첩 LOEO (트랙 B: IS 관문 통과한 설정 전부, 2011 ~ 2026.9)
    isj = json.load(open(ISJ, encoding="utf-8"))
    gB = [byid[c] for c, g in isj["gates_B"].items() if g["pass"] and c in byid and byid[c].track == "B"]
    out["nested_loeo"] = nested_loeo(X, gB, X.T) if gB else {"n_folds": 0}
    adopt = {}
    for trk in ("B", "A", "AB"):
        cid = vj["adopt_after_VAL"].get(trk)
        if not cid:
            adopt[trk] = None
            continue
        ok_ = veto[cid]["pass"]
        info = {"id": cid, "veto_pass": ok_}
        if trk in ("B", "AB"):
            nl = out["nested_loeo"]
            mv = module_value(X, byid[cid], nl.get("ooe_mean", float("nan")))
            info.update({"nested_ok": nl.get("ooe_mean", -1) > 0 and nl.get("ooe_median", -1) >= 0, "module_value": mv,
                         "value_ok": mv["value"] >= 0.003})
            ok_ = ok_ and info["nested_ok"] and info["value_ok"]
        ci = vj["criteria"][cid].get("pooled_ep_ci90")
        info["label"] = "strong" if ci and ci[0] is not None and ci[0] > 0 else "weak / positively skewed"
        info["adopt"] = bool(ok_)
        adopt[trk] = info
    out["adopt"] = adopt
    try:
        from backtest.lab import panic2_ev_a, panic2_ev_b
        la, sa_ = panic2_ev_a.run(F, "TEST")
        lb, sb_ = panic2_ev_b.run(F, "TEST")
        out["events_TEST"] = {"A_lines": la, "B_lines": lb, "A_summary": sa_, "B_summary": sb_}
    except Exception as ex:
        out["events_TEST"] = {"error": repr(ex)}
    out["elapsed_s"] = time.time() - t_start
    jdump(out, TESTJ)
    print("TEST 끝:", adopt, f"{time.time() - t_start:.0f}s")
    write_report()


# ════════════════════════════════════════════════════════════
# 보고서
# ════════════════════════════════════════════════════════════
def write_report():
    isj = json.load(open(ISJ, encoding="utf-8"))
    vj = json.load(open(VALJ, encoding="utf-8"))
    tj = json.load(open(TESTJ, encoding="utf-8"))
    fz = json.load(open(FREEZE, encoding="utf-8"))
    P = C.pct

    def g(d, k, dig=1):
        v = (d or {}).get(k)
        return P(v, dig) if isinstance(v, (int, float)) else "-"
    L = ["# 종목 패닉 · 시장 패닉 매수 — 2차 연구 (사전 등록 · 동결 · 검증)", "",
         (__doc__ or "").split("\n\n", 1)[1].strip(), "",
         f"동결: git {fz['git_head'][:10]} · {fz['created']} · 파일 해시는 lab_panic2_freeze.json", "",
         "## 결론", ""]
    ad = tj["adopt"]
    for trk, nm in (("B", "시장 패닉 (트랙 B)"), ("A", "종목 패닉 (트랙 A)"), ("AB", "조합")):
        a = ad.get(trk)
        if trk == "AB" and not a:
            continue
        if not a:
            L.append(f"- {nm}: 채택 없음 — VAL 까지 통과한 결선 설정이 없다.")
        else:
            L.append(f"- {nm}: {a['id']} — TEST 거부권 {'통과' if a['veto_pass'] else '실패'}"
                     + (f", 중첩 LOEO {'통과' if a.get('nested_ok') else '실패'}, 모듈 가치 {P(a['module_value']['value'], 2)}"
                        if trk in ("B", "AB") else "") + f" → {'채택' if a['adopt'] else '채택 안 함'} ({a['label']})")
    L += ["", "## 결선 · 적격 판정 (IS)", "", f"결선 B: {fz['finalists']['B'] or '없음'} · A: {fz['finalists']['A'] or '없음'} · 조합: {fz['AB']}", "",
          f"현실성 검사 (위약 최고값 90% 분위): B {g(isj.get('rc_B'), 'q90', 2)} · A {g(isj.get('rc_A'), 'q90', 2)}", "",
          "| 설정 | 트랙 | 설명 | IS 평균 | VAL 평균 | TEST 평균 | IS Δ | VAL Δ | TEST Δ | VAL Δ(대조군) | IS P10d | 진입 창 비율 | 탈락 사유 (IS) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cid, c in isj["configs"].items():
        v = vj["configs"].get(cid, {}).get("VAL", {})
        t = tj["configs"].get(cid, {}).get("TEST", {})
        s = c["IS"]
        why = "; ".join(isj["eligibility"].get(cid, [])) if cid in isj["eligibility"] else ("기준선" if c["track"] == "R" else "")
        L.append(f"| {cid} | {c['track']} | {c['desc']} | {g(s, 'mean')} | {g(v, 'mean')} | {g(t, 'mean')} | {g(s, 'delta', 2)} | "
                 f"{g(v, 'delta', 2)} | {g(t, 'delta', 2)} | {g(v, 'delta_ctrl', 2)} | {s['p10d'] * 100:.0f}% | "
                 f"{s['share_entry'] * 100:.0f}% | {why or ('결선' if cid in fz['finalists']['A'] + fz['finalists']['B'] else '적격 (결선 밖)')} |")
    if "AB" in vj["configs"]:
        v, t = vj["configs"]["AB"]["VAL"], tj["configs"]["AB"]["TEST"]
        L.append(f"| AB | AB | 조합 {fz['AB']} | - | {g(v, 'mean')} | {g(t, 'mean')} | - | {g(v, 'delta', 2)} | {g(t, 'delta', 2)} | "
                 f"{g(v, 'delta_ctrl', 2)} | - | - | |")
    L += ["", "## VAL 판정 (결선만)", "", "| 설정 | (i) Δ≥0.3pp (R0·대조군) | (ii) P10d | (iii) 하나 빼기 | (iv) 합친 사건 | (v) 위약 90% | CVaR | 합계 |",
          "|---|---|---|---|---|---|---|---|"]
    for cid, cr in vj["criteria"].items():
        L.append(f"| {cid} | {'O' if cr['i'] else 'X'} ({P(cr['delta'], 2)} / {P(cr['delta_ctrl'], 2)}) | {'O' if cr['ii'] else 'X'} | "
                 f"{'O' if cr['iii'] else 'X'} | {'O' if cr['iv'] else 'X'} | {'O' if cr['v'] else 'X'} ({P(cr['pooled_delta'], 2)} vs {P(cr['placebo_q90'], 2)}) | "
                 f"{'O' if cr['cvar'] else 'X'} | {'통과' if cr['pass'] else '탈락'} |")
    if not vj["criteria"]:
        L.append("| (결선 없음) | | | | | | | |")
    L += ["", "## TEST 거부권 · 중첩 LOEO", "", "TEST 는 이전 연구(lab_panic)에서 이미 본 구간이라 오염 — 거부만 할 수 있다.", ""]
    for cid, v in tj["veto"].items():
        L.append(f"- {cid}: TEST {P(v['mean'])} (R0 {P(tj['configs']['R0']['TEST']['mean'])}), P10d {v['p10d'] * 100:.0f}% → "
                 f"{'통과' if v['pass'] else '거부'}")
    nl = tj.get("nested_loeo", {})
    if nl.get("n_folds"):
        L.append(f"- 중첩 LOEO (관문 통과 B 설정들, {nl['n_folds']}개 사건): 사건 밖 평균 {P(nl['ooe_mean'], 2)}, 중앙값 "
                 f"{P(nl['ooe_median'], 2)}, 낙관 편향 {P(nl['optimism_gap'], 2)}, 선택 안정성 {nl['stability'] * 100:.0f}%")
    L += ["", "## 사건 조사 관문 — 트랙 A (IS 2011 ~ 2018, 상위 2/일, 5일)", "",
          "| 사건 · 진입 | 건수 | 중앙값 | 초과 (NW t) | 평소 보유 대비 (t) | 대조 종목 대비 | E1 / E2 초과 | 통과 | 사유 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, gA in isj["gates_A"].items():
        L.append(f"| {k} | {gA.get('n', 0)} | {g(gA, 'median', 2)} | {g(gA, 'excess', 2)} ({gA.get('t_exc', float('nan')):.1f}) | "
                 f"{g(gA, 'dA_mean', 2)} ({gA.get('dA_t', float('nan')):.1f}) | {g(gA, 'placebo_diff', 2)} | "
                 f"{g(gA, 'exc_E1', 2)} / {g(gA, 'exc_E2', 2)} | {'O' if gA['pass'] else 'X'} | {'; '.join(gA['reasons'])} |")
    L += ["", "## 사건 조사 관문 — 트랙 B (IS, 사건=10거래일 안 묶음, Delta = 패닉 바구니 − 평소 상위 2 − 0.35%)", "",
          "| 설정 | IS 트리거일 | IS 사건 | VAL 사건 | 사건 평균 Δ | Δ>0 비율 | 최악 사건 | 고원 | 통과 | 사유 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, gB in isj["gates_B"].items():
        pl = gB.get("plateau") or {}
        L.append(f"| {k} | {gB['n_days_IS']} | {gB['N_ep_IS']} | {gB['N_ep_VAL']} | {g(gB, 'ep_mean', 2)} | "
                 f"{(gB.get('ep_pos') or 0) * 100:.0f}% | {g(gB, 'ep_worst', 2)} | "
                 + " / ".join(f"H{h}:{P(v, 2)}" for h, v in pl.items()) + f" | {'O' if gB['pass'] else 'X'} | {'; '.join(gB['reasons'])} |")
    L += ["", "## 기준선", ""]
    for per, j in (("IS", isj["configs"]["R0"]["IS"]), ("VAL", vj["configs"]["R0"]["VAL"]), ("TEST", tj["configs"]["R0"]["TEST"])):
        L.append(f"- R0 {per}: 평균 {P(j['mean'])}, 중앙값 {P(j['median'])}, P10d {j['p10d'] * 100:.0f}%, CVaR10 {P(j['cvar10'])}, "
                 f"최근 12개월 {P(j.get('last12'))}")
    L += ["", "## 트리거 일수 (연도별)", "", "| 연도 | " + " | ".join(isj["trigger_days_per_year"]) + " |",
          "|---|" + "---|" * len(isj["trigger_days_per_year"])]
    yrs = list(next(iter(isj["trigger_days_per_year"].values())).keys())
    for y in yrs:
        L.append(f"| {y} | " + " | ".join(str(isj["trigger_days_per_year"][n][y]) for n in isj["trigger_days_per_year"]) + " |")
    L += ["", f"IS 에서 T0 과의 겹침 (Jaccard): " + ", ".join(f"{k} {v:.2f}" for k, v in isj["jaccard_T0_IS"].items()), "",
          "U200 업종 라벨 비율: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in isj["sector_coverage_U200"].items()), ""]
    for per, j in (("IS", isj.get("events_IS")), ("VAL", vj.get("events_VAL")), ("TEST", tj.get("events_TEST"))):
        L += ["", f"## 사건 조사 — {per}", ""]
        if not j or "error" in j:
            L.append(f"(실패: {(j or {}).get('error')})")
            continue
        L += ["### 트랙 A (종목 패닉)", ""] + j["A_lines"] + ["", "### 트랙 B (시장 패닉)", ""] + j["B_lines"]
    L += ["", "## 한계", "",
          "- 업종 라벨은 2018 스냅샷, 시장 구분은 시점별(point-in-time). 관리 · 투자경고 · 정리매매 이력은 없어 '최근 20일 정지 · 하한가' 로 대신했다.",
          "- 트랙 B 의 VAL 은 깨끗하지 않다: F-a · F-c · 손절 · 분할 · T6 는 lab_panic 에서 2020 · 2022 가 VAL 을 망친 걸 본 뒤 설계했다. "
          "TEST(2024.10 ~) 도 이미 봤다. 그래서 중첩 LOEO 를 채택 조건에 넣었다.",
          "- 트랙 A 의 조용한 급락 · 투매 아이디어는 2015 ~ 2026 칼날 연구(VAL · TEST 포함)를 본 뒤 나왔다.",
          "- 일봉만 있다: 15:19 판단은 종가로 근사했고 (잡음 검사 · 다음 날 시가 검사로 보완), 장중 체결 순서는 모른다."]
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("보고서:", REPORT)


def main() -> int:
    ph = sys.argv[1] if len(sys.argv) > 1 else "is"
    if ph == "is":
        phase_is()
    elif ph == "val":
        phase_val()
    elif ph == "test":
        phase_test()
    elif ph == "report":
        write_report()
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
