"""
backtest/lab/panic2_ev_b.py — 패닉 2차 연구 트랙 B (시장 전체 패닉) 서술 사건 조사 EB1 ~ EB7

  from backtest.lab import panic2_ev_b as EVB; lines, summary = EVB.run(F, "IS")
  PYTHONPATH=. python -m backtest.lab.panic2_ev_b          → IS 만 계산해서 출력 (개발 · 검사용)

정의는 backtest/results/lab_panic2_design.json 의 grid.event_studies (EB1..EB7) 와 critique 를 따른다.
공통 규칙 (panic2_common)
  · 신호일 t 의 선택 · 트리거는 t 종가까지의 정보만 쓴다 (정규화 창은 t−1 에서 끝난다).
    트리거 격자는 기간 끝(t1) 앞까지의 배열만 잘라 인과적으로 다시 계산한다.
  · 수익률은 C.fwd / C.fwd_path 로만 (진입일 = 1일차, 청산일이 거래정지면 다음 거래 종가, 왕복 0.35%).
  · Δswitch(h) = 바구니 net − 같은 창 평소 전략 top_base[t][:2] 평균 gross − 0.35% (갈아타는 비용).
  · 에피소드: 신호일을 G=10 거래일 간격으로 묶은 평균 (집계용일 뿐, 매매 신호가 아니다).
  · 모든 통계는 C.period_range(F, period) 안의 신호일만 쓴다 (EB7 위약 후보일도 같은 기간 안).

  EB1 강도-반응   x · zm · K200p z 구간별 시장(베타) · 선택(깊은 2·5 − EW U200) · Δswitch, 스피어만 · 기울기 CI · LOEO
  EB2 트리거 목록 T0 T1 T2 T3 T4 T5 T7 T6 격자: 빈도 · 에피소드 통계 · 시대별 · Δswitch(K2 top-2 · top-10) · Jaccard
  EB3 선택 경주   T0 · x ≤ −3% 에서 선택 규칙 36개 (지수 대용 1x · 2x 포함): 베타/잔차 분해 · K2 대비 · 무작위 쌍 · 셔플
  EB4 진입 시점   종가 t · 시가 t+1 · 시가 t+2 · 종가 t+2 · 확인(u*) 진입 → 공통 종점 종가 t+10 / t+21, g1 i1 g2 진단
  EB5 국면 나누기 F-a · mkt_on · deepcap · F-c · 변동성 · 연쇄 · 200일선 · CLV · 갭/장중 (+ 월요일 · 연휴 다음 날 보고만)
  EB6 청산 · 슬롯 고정 H 곡선 · 시장 익절 X-MKT · 시장 손절 · 회복 청산 · 전량 2 / 분할 1+1 / 3슬롯(20만원 상한)
  EB7 위약       짝지은 평온일 (2000회) · 시간 이동 +21/+42 · 무작위 U200 종목
"""
from __future__ import annotations

import datetime as _dt
import sys
import time
import warnings

import numpy as np

from backtest.lab import panic2_common as C
from backtest.lab import panic2_feat as PF

warnings.filterwarnings("ignore", category=RuntimeWarning)

COST = C.COST
IDX_COST = 0.00065          # 지수 대용 왕복 (수수료 · 슬리피지, 매도세 없음)
DRAG2 = 0.0064 / 250        # 2배 상품 연 0.64% 보수 (하루치)
G = 10
NB = 2000                   # 무작위 반복 수
SEED = 20260928
CAP3 = 200_000              # 3슬롯 가격 상한
nan = np.nan


# ════════════════════════════════════════════════════════════
# 도구
# ════════════════════════════════════════════════════════════
def fp(v, d=1):
    return C.pct(v, d)


def fs(v):
    """비율 (0~1) → 정수 %."""
    return "-" if v is None or not np.isfinite(v) else f"{v * 100:.0f}%"


def fn(v, d=2):
    return "-" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def head(*cells):
    return [row(*cells), "|" + "|".join(["---"] + ["---:"] * (len(cells) - 1)) + "|"]


def clean(o):
    """JSON 으로 쓸 수 있게 (numpy → 파이썬, NaN → None, 키는 문자열)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), 6) if np.isfinite(o) else None
    return o


def eras_of(F, t0, t1):
    """기간과 겹치는 시대 (이름, 시작, 끝)."""
    out = []
    for nm, (a, z) in C.ERAS.items():
        a_, z_ = C.didx(F, a), C.didx(F, z)
        if a_ < t1 and z_ > t0:
            out.append((nm, max(a_, t0), min(z_, t1)))
    return out


def ymd(s):
    return _dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def years_of(F, t0, t1):
    return max((ymd(F.dates[t1 - 1]) - ymd(F.dates[t0])).days / 365.25, 1e-9)


def bask(F, days, J, entry, h):
    """바구니 (날짜 × 종목, −1 없음) → 날짜별 평균 gross. 진입 못 한 종목은 빼고, 다 빠지면 NaN. h 는 날짜별 배열도 된다."""
    days = np.asarray(days, np.int64)
    if not len(days):
        return np.zeros(0)
    J = np.asarray(J, np.int64).reshape(len(days), -1)
    g = np.full(J.shape, nan)
    v = J >= 0
    if v.any():
        tt = np.broadcast_to(days[:, None], J.shape)
        hh = np.broadcast_to(np.asarray(h).reshape(-1, 1) if np.ndim(h) else np.asarray(h), J.shape)
        g[v] = C.fwd(F, tt[v], J[v], entry, hh[v])[1]
    return np.nanmean(g, 1)


def dsw(F, days, J, entry, h):
    """Δswitch = 바구니 net − top_base[t][:2] 평균 gross − 0.35% (h 는 정수)."""
    days = np.asarray(days, np.int64)
    if not len(days):
        return np.zeros(0)
    return bask(F, days, J, entry, h) - COST - C.base_gross(F, days, entry, h) - COST


def ept(days, vals, gap=G):
    """C.ep_table + 빈 경우 기본값."""
    r = C.ep_table(days, vals, gap)
    if r.get("N", 0) == 0:
        r = {k: nan for k in ("mean", "median", "pos", "worst", "best", "loo_min", "loo_max", "no_best1",
                              "no_best2", "day_mean")}
        r.update(N=0, ep_means=[], ep_first=[])
    return r


def nep(days, gap=G):
    return int(C.episodes_of(np.sort(days), gap).max()) + 1 if len(days) else 0


def epm(ep, vals):
    """에피소드 번호별 평균 (NaN 무시, 값이 없으면 NaN)."""
    vals = np.asarray(vals, float)
    n = int(ep.max()) + 1 if len(ep) else 0
    ok = np.isfinite(vals)
    s = np.bincount(ep[ok], weights=vals[ok], minlength=n)
    c = np.bincount(ep[ok], minlength=n)
    return np.where(c > 0, s / np.maximum(c, 1), nan)


def em_stats(em):
    em = np.asarray(em, float)
    em = em[np.isfinite(em)]
    if not len(em):
        return {"N": 0, "mean": nan, "median": nan, "worst3": nan, "pos": nan}
    return {"N": int(len(em)), "mean": float(em.mean()), "median": float(np.median(em)),
            "worst3": float(np.sort(em)[:3].mean()), "pos": float((em > 0).mean())}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return nan
    ra, rb = a[ok].argsort().argsort(), b[ok].argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def pctile(actual, dist):
    """분포 안에서 실제 값의 백분위 (동점 반)."""
    d = np.asarray(dist, float)
    d = d[np.isfinite(d)]
    if not len(d) or not np.isfinite(actual):
        return nan
    return float((d < actual).mean() + 0.5 * (d == actual).mean())


def topn(score, n=10):
    """점수 (NaN = 후보 아님, 클수록 먼저) → 상위 n 인덱스 (−1 채움). 동점은 인덱스 작은 쪽."""
    s = np.where(np.isfinite(score), score, -np.inf)
    idx = np.argsort(-s, kind="stable")[:n]
    idx = idx[np.isfinite(s[idx])]
    out = np.full(n, -1, np.int64)
    out[:len(idx)] = idx
    return out


def pad(mask2d):
    """날짜 × 종목 bool → 날짜별 종목 목록 (−1 채움)."""
    rows = [np.where(r)[0] for r in mask2d]
    w = max([len(r) for r in rows] + [1])
    out = np.full((len(rows), w), -1, np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def tops(F, name, days, n):
    if name == "KPLU":
        return kospi_l(F, days, n)
    return np.asarray(getattr(F, "top_" + name)[np.asarray(days, np.int64), :n]).astype(np.int64)


def kospi_l(F, days, n=10):
    """KOSPI-L (설계 EB3: KOSPI & caprank ≤ 200 = U200 & 그날 KOSPI), ret1 깊은 순 (캐시 top_KPL 과 같은 정의)."""
    days = np.asarray(days, np.int64)
    out = np.full((len(days), n), -1, np.int64)
    for i, t in enumerate(days):
        m = np.asarray(F.U200[t]) & (np.asarray(F.lab[t]) == 1)
        out[i] = topn(np.where(m, -np.asarray(F.ret1[t], float), nan), n)
    return out


def first_days(days, gap=G):
    """정렬된 날짜 → 각 에피소드 첫날 bool."""
    if not len(days):
        return np.zeros(0, bool)
    ep = C.episodes_of(days, gap)
    return np.r_[True, np.diff(ep) > 0]


# ════════════════════════════════════════════════════════════
# EB1 강도-반응
# ════════════════════════════════════════════════════════════
BK = {"x": [(-2, -3), (-3, -4), (-4, -5.5), (-5.5, -np.inf)],
      "zm": [(-2, -3), (-3, -4.5), (-4.5, -np.inf)],
      "zk": [(-2, -3), (-3, -4.5), (-4.5, -np.inf)]}
DEFN = {"x": "x (%)", "zm": "zm", "zk": "K200p z"}
H1 = (1, 5, 10, 21)


def bname(hi, lo):
    return f"≤ {hi:g}" if not np.isfinite(lo) else f"({hi:g}, {lo:g}]"


def wslope(s, y, W):
    """가중 OLS 기울기 (W: 반복 × 날짜 가중치)."""
    sw = W.sum(1)
    ms, my = (W @ s) / sw, (W @ y) / sw
    return ((W @ (s * y)) / sw - ms * my) / ((W @ (s * s)) / sw - ms ** 2)


def eb1(F, t0, t1, X):
    x = np.asarray(F.x, float)
    days = np.where(x[t0:t1] <= -0.02)[0] + t0
    L = ["### EB1 강도-반응 (x · zm · K200p z 구간)",
         "x ≤ −2% 인 날을 세 강도 정의로 나누고 시가 t+1 진입 뒤 (a) 유동 시장 동일가중 gross, (b) U200 가장 깊은 2·5 − EW U200 "
         "(gross), (c) K2 top-2 Δswitch 를 구간 안 에피소드(G=10) 평균으로 잰다.", ""]
    out = {"n_days": int(len(days))}
    if len(days) < 3:
        return L + ["(표본 부족)"], out
    kr = np.asarray(F.k200_ret1, float)
    zk = np.array([kr[t] / max(np.sqrt(np.mean(kr[t - 20:t] ** 2)), 1e-9) for t in days])     # RMS 창 t−20..t−1
    sev = {"x": x[days] * 100, "zm": np.asarray(F.zm, float)[days], "zk": zk}
    top = tops(F, "K2", days, 5)
    a, b2, b5, c = {}, {}, {}, {}
    for h in H1:
        ewl = C.same_day_ew(F, days, F.liquid, "open", h)
        ewu = C.same_day_ew(F, days, F.U200, "open", h)
        a[h] = np.array([ewl.get(int(t), nan) for t in days])
        eu = np.array([ewu.get(int(t), nan) for t in days])
        g2 = bask(F, days, top[:, :2], "open", h)
        b2[h], b5[h] = g2 - eu, bask(F, days, top, "open", h) - eu
        c[h] = g2 - COST - C.base_gross(F, days, "open", h) - COST
    k2n5 = bask(F, days, top[:, :2], "open", 5) - COST

    T1 = head("정의", "구간", "일수", "에피소드", "시장 1d", "5d", "10d", "21d", "깊은2−EW 5d", "21d", "깊은5−EW 5d", "21d")
    T2 = head("정의", "구간", "일수", "에피소드", "Δ 1d", "Δ 5d", "Δ 10d", "Δ 21d", "Δ5 중앙", "Δ21 중앙", "Δ5>0")
    bm, out["buckets"] = {}, {}
    for dn, bks in BK.items():
        v = sev[dn]
        grp = [(bname(hi, lo), (v <= hi) & (v > lo)) for hi, lo in bks]
        if dn != "x":
            grp.append(("> −2 (기타)", np.isfinite(v) & (v > -2)))
        out["buckets"][dn] = []
        for k, (lab, m) in enumerate(grp):
            dd = days[m]
            st = {q: {h: ept(dd, arr[h][m]) for h in H1} for q, arr in (("a", a), ("b2", b2), ("b5", b5), ("c", c))}
            ne = nep(dd)
            T1.append(row(DEFN[dn], lab, int(m.sum()), ne, *[fp(st["a"][h]["mean"], 2) for h in H1],
                          fp(st["b2"][5]["mean"]), fp(st["b2"][21]["mean"]), fp(st["b5"][5]["mean"]), fp(st["b5"][21]["mean"])))
            T2.append(row(DEFN[dn], lab, int(m.sum()), ne, *[fp(st["c"][h]["mean"], 2) for h in H1],
                          fp(st["c"][5]["median"]), fp(st["c"][21]["median"]), fs(st["c"][5]["pos"])))
            out["buckets"][dn].append({"bucket": lab, "n_days": int(m.sum()), "n_ep": ne,
                                       **{q: {h: st[q][h]["mean"] for h in H1} for q in st},
                                       "c_median": {h: st["c"][h]["median"] for h in H1}})
            if k < len(bks):
                for q in st:
                    for h in H1:
                        bm.setdefault((dn, h, q), []).append(st[q][h]["mean"])
    L += ["(a)(b) 시장 · 선택 부분 (에피소드 평균, gross)", ""] + T1 + ["", "(c) K2 top-2 Δswitch (에피소드 평균)", ""] + T2
    L.append("구간 (a, b] 는 b < 값 ≤ a (T0 = x ≤ −4% 가 x 의 3·4 구간 합과 같게). K200p z = k200_ret1 / RMS(t−20..t−1). "
             "'기타' 는 x ≤ −2% 인데 z 가 −2 보다 큰 날 (순위상관에서 뺌).")

    # 순위상관 · 기울기 (에피소드 블록 부트스트랩 90%)
    ep_all = C.episodes_of(days, G)
    E = int(ep_all.max()) + 1
    rng = np.random.default_rng(SEED)
    cnt = np.zeros((NB, E))
    np.add.at(cnt, (np.arange(NB)[:, None], rng.integers(0, E, size=(NB, E))), 1)
    T3 = head("정의", "h", "ρ 시장", "ρ 깊은2", "ρ 깊은5", "ρ Δ", "Δ 기울기 (|강도| 1단위)", "90% CI")
    out["spearman"], out["slope"] = {}, {}
    for dn, bks in BK.items():
        for h in H1:
            rho = {q: spearman(np.arange(len(bks)), bm[(dn, h, q)]) for q in ("a", "b2", "b5", "c")}
            s, y = np.abs(sev[dn]), c[h]
            ok = np.isfinite(s) & np.isfinite(y)
            sl = float(wslope(s[ok], y[ok], np.ones((1, ok.sum())))[0]) if ok.sum() > 2 else nan
            bs = wslope(s[ok], y[ok], cnt[:, ep_all[ok]]) if ok.sum() > 2 else np.array([nan])
            lo, hi = np.nanquantile(bs, [0.05, 0.95]) if np.isfinite(bs).any() else (nan, nan)
            T3.append(row(DEFN[dn], h, *[fn(rho[q]) for q in ("a", "b2", "b5", "c")], fp(sl, 2),
                          f"[{fp(lo, 2)}, {fp(hi, 2)}]"))
            out["spearman"].setdefault(dn, {})[h] = rho
            out["slope"].setdefault(dn, {})[h] = [sl, float(lo), float(hi)]
    L += ["", "구간 평균의 순위상관 (강도 순, + = 셀수록 큼) · 날짜 단위 Δ 회귀 기울기", ""] + T3
    L.append("기울기 단위: x 는 1%p, z 는 1 당 Δ 변화. CI 는 x ≤ −2% 전체 날의 에피소드(G=10)를 2000번 복원 추출.")

    # LOEO: 에피소드 최대 강도 → 에피소드 5d 수익 예측
    Y = {"K2 net 5d": epm(ep_all, k2n5), "Δ 5d": epm(ep_all, c[5])}
    T4 = head("정의", "에피소드", "K2 net 5d RMSE", "R² (LOEO)", "Δ5 RMSE", "R² (LOEO)")
    out["loeo"] = {}
    for dn in BK:
        fe = np.array([np.nanmax(np.abs(sev[dn][ep_all == e])) if np.isfinite(sev[dn][ep_all == e]).any() else nan
                       for e in range(E)])
        cells, res = [], {}
        for yn, ye in Y.items():
            ok = np.isfinite(fe) & np.isfinite(ye)
            f_, y_ = fe[ok], ye[ok]
            n = len(y_)
            if n < 5:
                cells += ["-", "-"]
                continue
            pr, b0 = np.zeros(n), np.zeros(n)
            for i in range(n):
                m = np.arange(n) != i
                k1, k0 = np.polyfit(f_[m], y_[m], 1)
                pr[i], b0[i] = k0 + k1 * f_[i], y_[m].mean()
            sse, sse0 = ((y_ - pr) ** 2).sum(), ((y_ - b0) ** 2).sum()
            res[yn] = {"n": n, "rmse": float(np.sqrt(sse / n)), "r2_oos": float(1 - sse / sse0)}
            cells += [fp(res[yn]["rmse"], 2), fn(res[yn]["r2_oos"])]
        T4.append(row(DEFN[dn], E, *cells))
        out["loeo"][dn] = res
    L += ["", "어느 정의가 에피소드 5d 수익을 더 잘 맞히나 (에피소드 최대 |강도| 로 선형회귀, 한 에피소드씩 빼고 예측)", ""] + T4
    L.append("R² (LOEO) = 1 − 오차제곱합 / (빼고 구한 평균으로 맞힐 때 오차제곱합). 0 이하면 평균보다 못 맞힌다.")
    return L, out


# ════════════════════════════════════════════════════════════
# EB2 트리거 목록
# ════════════════════════════════════════════════════════════
MID = ["T0 θ=4%", "T1 Z=4", "T2 b=0.75", "T3 k=3%", "T4 c=8%", "T5 (2.5, 10%)", "T7 δ=8%", "T6 X=+1%"]


def trig_grid(F, t1):
    """트리거 격자를 [0, t1) 배열로만 다시 계산 (인과적, 불응기는 처음부터 이어 센다)."""
    x = np.asarray(F.x[:t1], float)
    zm, br, kr = (np.asarray(getattr(F, k)[:t1], float) for k in ("zm", "breadth3", "k200_ret1"))
    M, S5, dd60, ibs, dd20 = (np.asarray(getattr(F, k)[:t1], float) for k in ("M", "S5", "mkt_dd60", "ew_ibs", "dd20m"))
    M3 = np.full(t1, nan)
    M3[3:] = M[3:] / M[:-3] - 1
    g = {}
    for th in (3, 4, 5):
        g[f"T0 θ={th}%"] = x <= -th / 100
    for Z in (3, 4, 5):
        g[f"T1 Z={Z}"] = (zm <= -Z) & (x <= -0.025)
    for b in (0.6, 0.75, 0.9):
        g[f"T2 b={b:g}"] = br >= b
    for k in (2.5, 3, 4):
        g[f"T3 k={k:g}%"] = kr <= -k / 100
    for cc in (6, 8, 10):
        g[f"T4 c={cc}%"] = PF.refractory((M3 <= -cc / 100) & (x < 0), 5)
    for th, dl in ((2, 8), (2.5, 10), (3, 15)):
        g[f"T5 ({th:g}, {dl}%)"] = PF.refractory((S5 >= th) & (dd60 <= -dl / 100), 5)
    for dl in (8, 12):
        g[f"T7 δ={dl}%"] = (x <= -0.03) & (ibs >= 0.5) & (dd20 <= -dl / 100)
    return g


def t6(x, T0, X, W=10):
    """확인일 u*: T0 날 t0 뒤 W 일 안 첫 x ≥ X (그 전에 새 T0 면 t0 갱신). (확인 bool, 기준일)."""
    n = len(x)
    conf, anc = np.zeros(n, bool), np.full(n, -1, np.int64)
    a = -1
    for t in range(n):
        if T0[t]:
            a = t
            continue
        if a >= 0 and t - a > W:
            a = -1
        if a >= 0 and x[t] >= X:
            conf[t], anc[t], a = True, a, -1
    return conf, anc


def t6_lists(F, days, anc, n=10, key="top_K2"):
    """기준일 t0 의 상위 12 를 고정, u* 에 아직 ok 인 종목을 c[u*]/c[t0−1]−1 오름차순."""
    out = np.full((len(days), n), -1, np.int64)
    for i, u in enumerate(days):
        a = int(anc[u])
        c0, cu, oku = F.c[a - 1], F.c[u], F.ok[u]
        Lj = [int(j) for j in getattr(F, key)[a] if j >= 0 and oku[j] and np.isfinite(c0[j])]
        Lj.sort(key=lambda j: cu[j] / c0[j] - 1)
        out[i, :min(n, len(Lj))] = Lj[:n]
    return out


def eb2(F, t0, t1, X):
    L = ["### EB2 트리거 목록 (T0 · T1 · T2 · T3 · T4 · T5 · T7 · T6 격자)",
         "트리거마다 빈도 · 에피소드 수(G=5/10/21) · K2 top-2 Δswitch (시가 t+1, T6 은 종가 u* / 시가 u*+1) 의 에피소드 통계, "
         "top-10 Δswitch · 시대별, 그리고 중간값 트리거끼리의 날짜 Jaccard.", ""]
    g = trig_grid(F, t1)
    x = np.asarray(F.x[:t1], float)
    T0m = g["T0 θ=4%"]
    rows = []                    # (이름, 날짜, 목록, 진입)
    for nm, m in g.items():
        d = np.where(m[t0:t1])[0] + t0
        rows.append((nm, d, tops(F, "K2", d, 10), "open"))
    t6m = {}
    for X_ in (0.0, 0.01, 0.02):
        conf, anc = t6(x, T0m, X_)
        d = np.where(conf[t0:t1])[0] + t0
        Jl = t6_lists(F, d, anc)
        t6m[f"T6 X={X_:+.0%}"] = conf
        for ent, lab in (("close", "종가 u*"), ("open", "시가 u*+1")):
            rows.append((f"T6 X={X_:+.0%} {lab}", d, Jl, ent))
    yrs = years_of(F, t0, t1)
    eras = eras_of(F, t0, t1)
    HS = (1, 3, 5, 10, 21)
    T0d = set(np.where(T0m[t0:t1])[0] + t0)
    Ta = head("트리거", "일수", "연", "에피 G5", "G10", "G21", "J(T0)", "일평균", "에피 평균", "중앙", ">0", "최악",
              "LOO 최소", "LOO 최대", "−최고1", "−최고2")
    Tb = head("트리거", "Δ2 1d", "3d", "5d", "10d", "21d", "Δ10 5d", "Δ10 21d", "G5 5d", "G21 5d",
              *[f"{e[0]} 5d" for e in eras], *[f"{e[0]} 21d" for e in eras])
    out = {}
    for nm, d, Jl, ent in rows:
        s = {"days": int(len(d)), "per_yr": len(d) / yrs, "n_ep": {gg: nep(d, gg) for gg in (5, 10, 21)}}
        ds = set(d.tolist())
        s["jac_T0"] = len(ds & T0d) / max(len(ds | T0d), 1)
        d2 = {h: dsw(F, d, Jl[:, :2], ent, h) for h in HS}
        d10 = {h: dsw(F, d, Jl[:, :10], ent, h) for h in HS}
        e5 = ept(d, d2[5])
        s["h5"] = {k: e5[k] for k in ("day_mean", "mean", "median", "pos", "worst", "loo_min", "loo_max", "no_best1", "no_best2")}
        s["d2"] = {h: ept(d, d2[h])["mean"] for h in HS}
        s["d2_day"] = {h: float(np.nanmean(d2[h])) if np.isfinite(d2[h]).any() else nan for h in HS}
        s["d2_median"] = {h: ept(d, d2[h])["median"] for h in HS}
        s["d10"] = {h: ept(d, d10[h])["mean"] for h in HS}
        s["g5_h5"], s["g21_h5"] = ept(d, d2[5], 5)["mean"], ept(d, d2[5], 21)["mean"]
        s["era"] = {}
        for en, a_, z_ in eras:
            m = (d >= a_) & (d < z_)
            s["era"][en] = {h: ept(d[m], d2[h][m])["mean"] for h in (5, 21)}
        out[nm] = s
        Ta.append(row(nm, s["days"], f"{s['per_yr']:.1f}", *s["n_ep"].values(), f"{s['jac_T0']:.2f}",
                      fp(e5["day_mean"]), fp(e5["mean"]), fp(e5["median"]), fs(e5["pos"]), fp(e5["worst"]),
                      fp(e5["loo_min"]), fp(e5["loo_max"]), fp(e5["no_best1"]), fp(e5["no_best2"])))
        Tb.append(row(nm, *[fp(s["d2"][h], 2 if h == 1 else 1) for h in HS], fp(s["d10"][5]), fp(s["d10"][21]),
                      fp(s["g5_h5"]), fp(s["g21_h5"]), *[fp(s["era"][e[0]][5]) for e in eras],
                      *[fp(s["era"][e[0]][21]) for e in eras]))
    L += ["K2 top-2 Δswitch 5d (에피소드 G=10)", ""] + Ta
    L += ["", "지평별 · top-10 · 묶음 간격 · 시대별 Δswitch 에피소드 평균", ""] + Tb
    L.append("T4 · T5 는 5일 불응기 (t−5..t−1 에 발동했으면 쉼). T6 은 T0(4%) 뒤 10일 안 첫 x ≥ X 날 (그 전 새 T0 면 기준 갱신), "
             "후보는 기준일 K2 상위 12 를 c[u*]/c[t0−1]−1 오름차순으로 다시 세운 것. J(T0) = T0(4%) 와의 날짜 Jaccard; "
             "0.6 이상 (critique 7a 병합 대상): "
             + (", ".join(k for k, v in out.items() if v["jac_T0"] >= 0.6 and not k.startswith("T0 θ=4")) or "없음") + ".")
    # 중간값 트리거 Jaccard
    mid = {k: (g[k] if k in g else t6m["T6 X=+1%"])[t0:t1] for k in MID}
    Tj = head("", *MID)
    jac = {}
    for a_ in MID:
        cells = []
        for b_ in MID:
            u = (mid[a_] | mid[b_]).sum()
            v = (mid[a_] & mid[b_]).sum() / u if u else nan
            jac.setdefault(a_, {})[b_] = float(v)
            cells.append(f"{v:.2f}" if np.isfinite(v) else "-")
        Tj.append(row(a_, *cells))
    L += ["", "중간값 트리거 날짜 Jaccard (T6 은 확인일이라 T0 와 겹치지 않는다)", ""] + Tj
    return L, {"triggers": out, "jaccard": jac}


# ════════════════════════════════════════════════════════════
# EB3 선택 경주
# ════════════════════════════════════════════════════════════
CACHED3 = [("K2", "K2"), ("S-RES", "SRES"), ("S-RES3", "SRES3"), ("S-BETA", "SBETA"), ("S-LEAD k5 q.90", "SLEAD"),
           ("S-LOSER L120", "SLOSER"), ("LVOL", "LVOL"), ("HMOM", "HMOM"), ("S-SECREL m5", "SSECREL"),
           ("S-HVOL", "SHVOL"), ("S-IBS", "SIBS"), ("KOSPI-L", "KPLU"),
           ("KOSDAQ-L", "KQL"), ("K1 관심", "K1"), ("S-MEGA", "SMEGA"), ("K2-EP", "K2EP")]
ORDER3 = ["K2", "K2 업종≤1", "S-RES", "S-RES 업종≤1", "S-RES3", "S-BETA", "S-LEAD k5 q.90", "S-LEAD k5 q.85",
          "S-LEAD k5 q.95", "S-LEAD k20 q.85", "S-LEAD k20 q.95", "S-LOSER L60", "S-LOSER L120", "S-LOSER L250", "LVOL",
          "HMOM", "S-SECREL m5", "S-SECREL m10", "약한 업종 대장", "강한 업종 대장", "S-HVOL", "S-LVOL", "S-HVOL 관심",
          "S-LVOL 관심", "S-IBS", "KOSPI-L", "KOSDAQ-L", "K1 관심", "S-MEGA", "K2-EP",
          "EW U200", "STAY (평소 top-2)",
          "K200p 1x", "K200p 2x", "KQ150p 1x", "KQ150p 2x"]


def eb3_lists(F, D):
    """날짜별 규칙 목록 (날짜 × 10). 저장된 목록 + 이 자리에서 만드는 격자 변형 (모두 t 종가까지 정보)."""
    nD = len(D)
    R = {nm: tops(F, key, D, 10) for nm, key in CACHED3}
    loc = ["K2 업종≤1", "S-RES 업종≤1", "S-LEAD k5 q.85", "S-LEAD k5 q.95", "S-LEAD k20 q.85", "S-LEAD k20 q.95",
           "S-LOSER L60", "S-LOSER L250", "S-SECREL m10", "약한 업종 대장", "강한 업종 대장", "S-LVOL", "S-HVOL 관심", "S-LVOL 관심"]
    for k in loc:
        R[k] = np.full((nD, 10), -1, np.int64)
    sec = np.asarray(F.sector)
    S_ = len(F.sectors)
    colx = np.where(sec >= 0, sec, S_)                   # 미분류 = 한 묶음

    def seccap(score, n=10):
        s = np.where(np.isfinite(score), score, -np.inf)
        seen, o = set(), []
        for j in np.argsort(-s, kind="stable"):
            if not np.isfinite(s[j]) or len(o) >= n:
                break
            if colx[j] not in seen:
                seen.add(colx[j])
                o.append(j)
        r = np.full(n, -1, np.int64)
        r[:len(o)] = o
        return r

    def volsplit(univ, r1, vr):
        H_ = Lw = np.full(10, -1, np.int64)
        if univ.sum() < 3:
            return H_, Lw
        dec = univ & (r1 <= np.nanmedian(np.where(univ, r1, nan)))
        rrv = vr / np.nanmedian(np.where(univ, vr, nan))
        if dec.sum() < 3:
            return H_, Lw
        rd = np.where(dec, rrv, nan)
        hi, lo = np.nanquantile(rd, 2 / 3), np.nanquantile(rd, 1 / 3)
        return topn(np.where(dec & (rrv >= hi), -r1, nan)), topn(np.where(dec & (rrv <= lo), -r1, nan))

    for i, t in enumerate(D):
        U, ok = np.asarray(F.U200[t]), np.asarray(F.ok[t])
        r1 = np.asarray(F.ret1[t], float)
        R["K2 업종≤1"][i] = seccap(np.asarray(F.sc_K2[t], float))
        R["S-RES 업종≤1"][i] = seccap(np.asarray(F.sc_SRES[t], float))
        ret3 = np.asarray(F.ret3[t], float)
        for k in (5, 20):                                  # 떨어진 주도주: p = t−k 에 봇 후보 풀
            p = t - k
            base = (F.caprank[p] <= 200) & (F.ret60[p] > 0) & (F.max20[p] < 0.10)
            hi = F.c[p] / F.hi250[p]
            for q in (0.85, 0.95):
                R[f"S-LEAD k{k} q{str(q)[1:]}"][i] = topn(np.where(U & base & (hi >= q), -ret3, nan))
        R["S-LOSER L60"][i] = topn(np.where(U, -np.asarray(F.ret60[t - 1], float), nan))
        R["S-LOSER L250"][i] = topn(np.where(U, -np.asarray(F.ret250[t - 1], float), nan))
        # 업종 평균 (ok & val20 ≥ 30억, 미분류 한 묶음) — S-SECREL m10 · 약한/강한 업종 대장
        mem = ok & (np.asarray(F.val20[t], float) >= 3e9)
        rr = np.clip(r1, -0.3, 0.3)
        ssum = np.bincount(colx[mem], weights=np.nan_to_num(rr[mem]), minlength=S_ + 1)
        scnt = np.bincount(colx[mem], minlength=S_ + 1)
        SR10 = np.where(scnt >= 10, ssum / np.maximum(scnt, 1), nan)
        R["S-SECREL m10"][i] = topn(np.where(U, -(rr - SR10[colx]), nan))
        SR5 = np.where(scnt >= 5, ssum / np.maximum(scnt, 1), nan)[:S_]          # 이름 있는 업종만
        good = np.where(np.isfinite(SR5))[0]
        capp = np.asarray(F.cap[t - 1], float)
        for nm, pick in (("약한 업종 대장", good[np.argsort(SR5[good], kind="stable")][:2]),
                         ("강한 업종 대장", good[np.argsort(-SR5[good], kind="stable")][:2])):
            names = []
            for s_ in pick:
                cand = np.where(ok & (sec == s_) & np.isfinite(capp))[0]
                if len(cand):
                    names.append(cand[np.argmax(capp[cand])])
            R[nm][i, :len(names)] = names
        vr = np.asarray(F.vr[t], float)
        _, R["S-LVOL"][i] = volsplit(U, r1, vr)
        R["S-HVOL 관심"][i], R["S-LVOL 관심"][i] = volsplit(ok & np.asarray(F.attn20[t]), r1, vr)
    return R


def idx_leg(F, D, key, lev, h):
    """지수 대용 시가 t+1 → 종가 t+h gross. 2배: 매일 2·r − 보수, 시가 = 전날 NAV·(1 + 2·밤사이)."""
    nav, navo, r = (np.asarray(getattr(F, f"{key}_{k}"), float) for k in ("nav", "open", "ret1"))
    out = np.full(len(D), nan)
    for i, t in enumerate(D):
        if t + h >= F.T:
            continue
        if lev == 1:
            out[i] = nav[t + h] / navo[t + 1] - 1
        else:
            ov = navo[t + 1] / nav[t] - 1
            out[i] = np.prod(1 + 2 * r[t + 1:t + h + 1] - DRAG2) / (1 + 2 * ov) - 1
    return out


def lg(Gh, J):
    """날짜 × 종목 gross 행렬 → 목록 J (−1 없음) 의 날짜별 평균."""
    J = np.asarray(J, np.int64)
    g = np.take_along_axis(Gh, np.maximum(J, 0), 1)
    return np.nanmean(np.where(J >= 0, g, nan), 1)


def lbeta(Gh, B, ew, J):
    """바구니 gross = b*·(같은 창 EW 시장) + 잔차, 종목별로 나눠 평균."""
    J = np.asarray(J, np.int64)
    g, b = np.take_along_axis(Gh, np.maximum(J, 0), 1), np.take_along_axis(B, np.maximum(J, 0), 1)
    ok = (J >= 0) & np.isfinite(g) & np.isfinite(b)
    bp = b * ew[:, None]
    return np.nanmean(np.where(ok, bp, nan), 1), np.nanmean(np.where(ok, g - bp, nan), 1)


def rand_em(Gh, rows, pools, ep, k, rng):
    """날짜마다 후보 pools 에서 k 개 무작위 (NB 번) → 에피소드 × NB 평균 gross."""
    Rm = np.full((len(rows), NB), nan)
    for r_, i in enumerate(rows):
        g = Gh[i, pools[i]]
        g = g[np.isfinite(g)]
        if len(g) < k:
            continue
        if k == 2:
            a = rng.integers(0, len(g), NB)
            b = (a + rng.integers(1, len(g), NB)) % len(g)
            Rm[r_] = (g[a] + g[b]) / 2
        else:
            Rm[r_] = g[rng.random((NB, len(g))).argsort(1)[:, :k]].mean(1)
    E = int(ep.max()) + 1 if len(ep) else 0
    return np.array([np.nanmean(Rm[ep == e], 0) for e in range(E)]).reshape(E, NB)


def eb3(F, t0, t1, X):
    x = np.asarray(F.x, float)
    D = np.where(x[t0:t1] <= -0.03)[0] + t0
    L = ["### EB3 선택 경주 (T0 · x ≤ −3%)",
         "U200 등 후보에서 규칙별 top-2 · top-10 을 시가 t+1 에 사서 5 · 10 · 21 일 (net 0.35%, 지수 0.065%, STAY 는 비용 없는 gross). "
         "에피소드(G=10) 평균 · K2 대비 · b*·x (베타) 와 잔차 분해 · 무작위 쌍 / 셔플 백분위.", ""]
    if len(D) < 2:
        return L + ["(표본 부족)"], {}
    N, nD = F.N, len(D)
    R = eb3_lists(F, D)
    Hs = (5, 10, 21)
    Gm = {h: C.fwd(F, np.repeat(D, N), np.tile(np.arange(N), nD), "open", h)[1].reshape(nD, N) for h in Hs}
    B = np.asarray(F.bstar[D], float)
    liq, U = np.asarray(F.liquid[D]), np.asarray(F.U200[D])
    ew = {h: np.nanmean(np.where(liq, Gm[h], nan), 1) for h in Hs}
    JU, JB = pad(U), tops(F, "base", D, 2)
    V = {}
    for nm, J in list(R.items()) + [("EW U200", JU), ("STAY (평소 top-2)", JB)]:
        cst = 0.0 if nm.startswith("STAY") else COST
        J2 = J if nm in ("EW U200",) or nm.startswith("STAY") else J[:, :2]
        V[nm] = {"cost": cst, "J": J if nm in R else None,
                 "n2": {h: lg(Gm[h], J2) - cst for h in Hs}, "n10": {h: lg(Gm[h], J) - cst for h in Hs},
                 "bp": {}, "rs": {}}
        for h in (5, 21):
            V[nm]["bp"][h], V[nm]["rs"][h] = lbeta(Gm[h], B, ew[h], J2)
    for key, lab in (("k200", "K200p"), ("kq150", "KQ150p")):
        for lev in (1, 2):
            n2 = {h: idx_leg(F, D, key, lev, h) - IDX_COST for h in Hs}
            V[f"{lab} {lev}x"] = {"cost": IDX_COST, "J": None, "n2": n2, "n10": dict(n2),
                                  "bp": {h: np.full(nD, nan) for h in (5, 21)}, "rs": {h: np.full(nD, nan) for h in (5, 21)}}
    Fa = np.asarray(F.mkt_on[D - 1]) | np.asarray(F.deepcap[D])
    eras = eras_of(F, t0, t1)
    rng = np.random.default_rng(SEED)
    out = {}
    for sk, lab, rows in (("T0", "T0 (x ≤ −4%)", np.where(x[D] <= -0.04)[0]), ("x3", "x ≤ −3%", np.arange(nD))):
        if len(rows) < 2:
            L += [f"{lab}: 표본 부족", ""]
            continue
        dd = D[rows]
        ep = C.episodes_of(dd, G)
        E = int(ep.max()) + 1
        eyr = np.array([F.dates[dd[ep == e][0]][:4] for e in range(E)])
        # 셔플 분포 (U200 무작위 2)
        sh = {h: rand_em(Gm[h], rows, [np.where(U[i])[0] for i in range(nD)], ep, 2, rng) for h in (5, 21)}
        res = {}
        emK = {h: epm(ep, V["K2"]["n2"][h][rows]) for h in (5, 21)}
        for nm in ORDER3:
            v = V[nm]
            st = {h: em_stats(epm(ep, v["n2"][h][rows])) for h in Hs}
            s = {"h": st, "top10": {h: em_stats(epm(ep, v["n10"][h][rows]))["mean"] for h in Hs},
                 "beta": {h: em_stats(epm(ep, v["bp"][h][rows]))["mean"] for h in (5, 21)},
                 "resid": {h: em_stats(epm(ep, v["rs"][h][rows]))["mean"] for h in (5, 21)}}
            for h in (5, 21):
                em = epm(ep, v["n2"][h][rows])
                ok = np.isfinite(em) & np.isfinite(emK[h])
                d_ = em[ok] - emK[h][ok]
                s[f"beatK2_{h}"] = float((d_ > 0).mean()) if ok.any() and nm != "K2" else nan
                s[f"ci_{h}"] = C.boot_ci(d_, n=5000, q=(0.025, 0.975), seed=SEED) if nm != "K2" else (nan, nan)
                okE = np.isfinite(em)                                        # 규칙 값이 있는 에피소드끼리만 비교
                gm = np.mean(em[okE] + v["cost"]) if okE.any() else nan      # gross 로 비교
                s[f"shuf_{h}"] = pctile(gm, np.nanmean(sh[h][okE], 0))
                if v["J"] is not None:
                    rp = rand_em(Gm[h], rows, [v["J"][i][v["J"][i] >= 0] for i in range(nD)], ep, 2, rng)
                    s[f"rpair_{h}"] = pctile(gm, np.nanmean(rp[okE], 0))
                else:
                    s[f"rpair_{h}"] = nan
            em5 = epm(ep, v["n2"][5][rows])
            yl = [np.nanmean(em5[eyr != y]) for y in np.unique(eyr)]
            s["loyo_min"] = float(np.nanmin(yl)) if len(yl) > 1 else nan
            for side, m in (("Fa_pass", Fa[rows]), ("Fa_fail", ~Fa[rows])):
                s[side] = ept(dd[m], v["n2"][5][rows][m])["mean"]
            s["era"] = {en: ept(dd[(dd >= a_) & (dd < z_)], v["n2"][5][rows][(dd >= a_) & (dd < z_)])["mean"]
                        for en, a_, z_ in eras}
            res[nm] = s
        shm = {h: np.nanmean(sh[h], 0) for h in (5, 21)}
        res["_shuffle"] = {h: {"median": float(np.nanmedian(shm[h])), "p90": float(np.nanquantile(shm[h], 0.9))}
                           for h in (5, 21)}
        out[sk] = {"n_days": int(len(rows)), "n_ep": E, "rules": res}
        cs = lambda s, h: f"[{fp(s[f'ci_{h}'][0])}, {fp(s[f'ci_{h}'][1])}]"
        if sk == "T0":
            Ta = head("규칙", "5d 평균", "5d 중앙", "5d 최악3", "10d 평균", "21d 평균", "21d 중앙", "21d 최악3",
                      "top-10 5d", "top-10 21d")
            Tb = head("규칙", "β·x 5d", "잔차 5d", ">K2", "LOYO 최소", "K2 대비 95% CI", "무작위쌍 %ile", "셔플 %ile",
                      "F-a 통과", "F-a 실패", *[e[0] for e in eras])
            for nm in ORDER3:
                s = res[nm]
                h = s["h"]
                Ta.append(row(nm, fp(h[5]["mean"]), fp(h[5]["median"]), fp(h[5]["worst3"]), fp(h[10]["mean"]),
                              fp(h[21]["mean"]), fp(h[21]["median"]), fp(h[21]["worst3"]), fp(s["top10"][5]), fp(s["top10"][21])))
                Tb.append(row(nm, fp(s["beta"][5]), fp(s["resid"][5]), fs(s["beatK2_5"]), fp(s["loyo_min"]), cs(s, 5),
                              fs(s["rpair_5"]), fs(s["shuf_5"]), fp(s["Fa_pass"]), fp(s["Fa_fail"]),
                              *[fp(s["era"][e[0]]) for e in eras]))
            Ta.append(row("무작위 2 (U200, 2000회 중앙)", fp(res["_shuffle"][5]["median"] - COST), "-", "-", "-",
                          fp(res["_shuffle"][21]["median"] - COST), "-", "-", "-", "-"))
            L += [f"{lab}: {len(rows)}일 · {E} 에피소드 — top-2 net (에피소드 평균)", ""] + Ta
            L += ["", f"{lab} 진단 (5d): 베타/잔차 (gross), K2 를 이긴 에피소드 비율, 한 해 빼기 최소, K2 와 차이의 95% 에피소드 "
                      "부트스트랩 CI, 자기 top-10 에서 무작위 쌍 대비 · U200 무작위 2 대비 백분위, F-a 국면 · 시대별 평균", ""] + Tb
        else:
            Tc = head("규칙", "5d 평균", "5d 중앙", "21d 평균", "21d 중앙", "top-10 5d", ">K2 5d", "K2 대비 95% CI",
                      "β·x 5d", "잔차 5d", "무작위쌍 %ile", "셔플 %ile")
            for nm in ORDER3:
                s = res[nm]
                h = s["h"]
                Tc.append(row(nm, fp(h[5]["mean"]), fp(h[5]["median"]), fp(h[21]["mean"]), fp(h[21]["median"]),
                              fp(s["top10"][5]), fs(s["beatK2_5"]), cs(s, 5), fp(s["beta"][5]), fp(s["resid"][5]),
                              fs(s["rpair_5"]), fs(s["shuf_5"])))
            L += ["", f"{lab}: {len(rows)}일 · {E} 에피소드 — top-2 net (에피소드 평균)", ""] + Tc
    L.append("S-LEAD 격자는 p=t−k 에 봇 후보 풀(시총 200 · c/hi250 ≥ q · ret60>0 · max20<10%) 안 ret3 오름차순. 업종 규칙은 "
             "미분류를 한 묶음으로 (critique). 업종 대장 = SR 최저/최고 2개 업종(이름 있는, 5개↑)의 전날 시총 1위. S-LVOL 은 "
             "하락 종목 중 rrv 하위 1/3. 관심 = attn20 (전날까지 20일 거래대금 50위). KOSPI-L = U200(전날 시총 200위) & KOSPI (설계). "
             "지수 top-10 = top-2 와 같은 한 종목. "
             "β·x = 종목별 b*(t−1 까지 창) × 같은 창 유동 시장 EW gross, 잔차 = gross − β·x. LOYO = 에피소드 첫날 연도 하나씩 뺀 "
             "평균의 최소. 무작위쌍 · 셔플 백분위는 gross 에피소드 평균끼리 비교 (seed 고정).")
    return L, out


# ════════════════════════════════════════════════════════════
# EB4 진입 시점
# ════════════════════════════════════════════════════════════
BASK4 = [("K2 top-2", "K2", 2), ("K2 top-5", "K2", 5), ("S-RES top-2", "SRES", 2), ("KOSPI-L top-2", "KPLU", 2),
         ("KOSDAQ-L top-2", "KQL", 2)]


def ustar(x, t, X, W=10):
    seg = x[t + 1:t + 1 + W]
    k = np.where(seg >= X)[0]
    return t + 1 + int(k[0]) if len(k) else -1


def mx_of(F, t, J):
    """신용 노출 대용 MX (t−1 유동 종목 안 백분위: ret60 · 회전율 · max20 평균, KOSDAQ 이면 +0.1)."""
    liq = np.asarray(F.liquid[t - 1])
    comps = []
    for arr in (F.ret60[t - 1], F.turn1[t], F.max20[t - 1]):
        a = np.asarray(arr, float)
        ref = np.sort(a[liq & np.isfinite(a)])
        v = a[J]
        comps.append(np.where(np.isfinite(v), np.searchsorted(ref, v, side="right") / max(len(ref), 1), nan))
    return np.nanmean(np.array(comps), 0) + 0.1 * (np.asarray(F.lab[t - 1])[J] == 2)


def eb4_events(F, days):
    """다섯 바구니 종목 합집합 (날짜, 종목)."""
    ts, js = [], []
    for t in days:
        jj = set()
        for _, key, n in BASK4:
            jj |= {int(j) for j in tops(F, key, [t], n)[0] if j >= 0}
        ts += [t] * len(jj)
        js += sorted(jj)
    return np.array(ts, np.int64), np.array(js, np.int64)


def eb4(F, t0, t1, X):
    x = np.asarray(F.x, float)
    days = np.where(x[t0:t1] <= -0.04)[0] + t0
    L = ["### EB4 진입 시점 (T0, 바구니는 t 에 고정)",
         "같은 종목을 여러 시점에 사서 공통 종점 종가 t+10 · t+21 까지 net (0.35%). 확인 진입 E3 = t+1..t+10 중 첫 x_u ≥ X 인 u* "
         "(없으면 거래 없음). 전체 T0 날과 에피소드 첫날만.", ""]
    if len(days) < 2:
        return L + ["(표본 부족)"], {}
    n = len(days)
    first = first_days(days)
    us = {X_: np.array([ustar(x, t, X_) for t in days]) for X_ in (0.0, 0.01, 0.02)}

    def spec(kind, X_, E):
        if kind == "E0":
            return days, "close", np.full(n, E), np.ones(n, bool)
        if kind == "E1":
            return days, "open", np.full(n, E), np.ones(n, bool)
        if kind == "E2":
            return days, "open2", np.full(n, E - 1), np.ones(n, bool)
        if kind == "E2c":
            return days, "close2", np.full(n, E - 2), np.ones(n, bool)
        u = us[X_]
        h = days + E - u
        return np.maximum(u, 0), ("close" if kind == "E3c" else "open"), h, (u >= 0) & (h >= 1)

    ents = [("E0 종가 t", "E0", None), ("E1 시가 t+1", "E1", None), ("E2 시가 t+2", "E2", None), ("E2c 종가 t+2", "E2c", None)]
    for X_ in (0.0, 0.01, 0.02):
        ents += [(f"E3 종가 u* X={X_:+.0%}", "E3c", X_), (f"E3 시가 u*+1 X={X_:+.0%}", "E3o", X_)]
    T = head("바구니", "진입", "채움", "t+10 평균", "중앙", "최악", "t+21 평균", "중앙", "최악", "첫날 t+10", "첫날 t+21",
             "−E1 t+10 (같은 날)")
    out = {}
    for bn, key, nn in BASK4:
        J = tops(F, key, days, nn)
        VV, trig = {}, {}
        for lab, kind, X_ in ents:
            VV[lab] = {}
            for E in (10, 21):
                sig, ent, h, valid = spec(kind, X_, E)
                v = np.full(n, nan)
                if valid.any():
                    v[valid] = bask(F, sig[valid], J[valid], ent, h[valid]) - COST
                VV[lab][E] = v
                if E == 10:
                    trig[lab] = float(np.isfinite(v).mean())       # 채움 = 실제로 산 날 비율 (확인 · 진입 가능, t+10 종점)
        base = VV["E1 시가 t+1"]
        for lab, kind, X_ in ents:
            vals = VV[lab]
            st = {E: ept(days, vals[E]) for E in (10, 21)}
            fd = {E: float(np.nanmean(vals[E][first])) if np.isfinite(vals[E][first]).any() else nan for E in (10, 21)}
            fmed = {E: float(np.nanmedian(vals[E][first])) if np.isfinite(vals[E][first]).any() else nan for E in (10, 21)}
            fwst = {E: float(np.nanmin(vals[E][first])) if np.isfinite(vals[E][first]).any() else nan for E in (10, 21)}
            dv = ept(days, vals[10] - base[10])["mean"] if kind not in ("E1",) else nan
            fill = trig[lab]
            out.setdefault(bn, {})[lab] = {"fill": fill, **{f"t{E}": {k: st[E][k] for k in ("N", "mean", "median", "worst")}
                                                            for E in (10, 21)},
                                           "first10": fd[10], "first21": fd[21], "first_median": fmed, "first_worst": fwst,
                                           "vsE1_10": dv}
            T.append(row(bn, lab, fs(fill), fp(st[10]["mean"]), fp(st[10]["median"]), fp(st[10]["worst"]),
                         fp(st[21]["mean"]), fp(st[21]["median"]), fp(st[21]["worst"]), fp(fd[10]), fp(fd[21]), fp(dv)))
    L += T
    L.append("E2 = open2 (t+2 시가, h 를 줄여 종점을 맞춤), E2c = t+2 종가. 확인 진입은 종점 전 하루 이상 보유할 때만 (종가 u* = 종점이면 거래 없음). "
             "채움 = t+10 종점 값이 있는 날 비율 (확인 없음 · 진입 불가 제외). "
             "'−E1' 은 같은 날들에서 E1 대비 차이의 에피소드 평균.")
    # 진단 g1 i1 g2 (시장 구분 t−1 · MX 3분위, 분위 경계는 IS 사건으로 고정)
    et, ej = eb4_events(F, days)
    if not len(et):
        return L, {"timing": out}
    mx = np.concatenate([mx_of(F, t, ej[et == t]) for t in np.unique(et)])
    ia, iz = C.period_range(F, "IS")
    if (ia, iz) == (t0, t1):
        mx_is = mx
    else:
        dis = np.where(x[ia:iz] <= -0.04)[0] + ia
        it, ij = eb4_events(F, dis)
        mx_is = np.concatenate([mx_of(F, t, ij[it == t]) for t in np.unique(it)]) if len(it) else np.array([nan])
    cut = np.nanquantile(mx_is, [1 / 3, 2 / 3]) if np.isfinite(mx_is).any() else np.array([nan, nan])
    e1_, e2_ = np.minimum(et + 1, F.T - 1), np.minimum(et + 2, F.T - 1)
    tr1 = np.asarray(F.traded[e1_, ej]) & (et + 1 < F.T)
    tr2 = np.asarray(F.traded[e2_, ej]) & (et + 2 < F.T)
    o1, c1 = np.asarray(F.o[e1_, ej], float), np.asarray(F.c[e1_, ej], float)
    o2, c0 = np.asarray(F.o[e2_, ej], float), np.asarray(F.c[et, ej], float)
    g1 = np.where(tr1, o1 / c0 - 1, nan)
    i1 = np.where(tr1, c1 / o1 - 1, nan)
    g2 = np.where(tr1 & tr2, o2 / c1 - 1, nan)
    e1 = C.fwd(F, et, ej, "open", 10)[0]
    e2 = C.fwd(F, et, ej, "open2", 9)[0]
    lab_ = np.asarray(F.lab[et - 1, ej])
    grp = [("전체", np.ones(len(et), bool)), ("KOSPI (t−1)", lab_ == 1), ("KOSDAQ (t−1)", lab_ == 2),
           ("MX 하", mx <= cut[0]), ("MX 중", (mx > cut[0]) & (mx <= cut[1])), ("MX 상", mx > cut[1])]
    T2 = head("구분", "사건", "에피소드", "g1 (o₁/c₀)", "i1 (c₁/o₁)", "g2 (o₂/c₁)", "E1 → t+10", "E2 − E1 (t+10)")
    diag = {}
    for nm, m in grp:
        dm = {k: float(np.nanmean(v[m])) if np.isfinite(v[m]).any() else nan
              for k, v in (("g1", g1), ("i1", i1), ("g2", g2), ("e1", e1), ("e2_e1", e2 - e1))}
        diag[nm] = {"n": int(m.sum()), **dm}
        T2.append(row(nm, int(m.sum()), nep(np.unique(et[m])), fp(dm["g1"], 2), fp(dm["i1"], 2), fp(dm["g2"], 2),
                      fp(dm["e1"]), fp(dm["e2_e1"], 2)))
    L += ["", "진단: 다섯 바구니 종목 합집합, 사건 단순 평균 (MX 3분위 경계는 IS T0 사건으로 고정: "
              f"{fn(cut[0])} / {fn(cut[1])})", ""] + T2
    return L, {"timing": out, "diag": diag, "mx_cut": cut.tolist()}


# ════════════════════════════════════════════════════════════
# EB5 국면 나누기
# ════════════════════════════════════════════════════════════
def classes(F, days, own):
    """t−1 까지 정보 (CLV · 갭 은 t) 로 국면 구분. own = 연쇄 계산에 쓰는 트리거 bool (전 기간)."""
    t = np.asarray(days, np.int64)
    on1 = np.asarray(F.mkt_on[t - 1])
    deep = np.asarray(F.deepcap[t])
    dd60 = np.asarray(F.mkt_dd60[t - 1], float)
    v20 = np.asarray(F.mkt_vol20, float)
    calm = np.array([v20[u - 1] <= np.nanmedian(v20[max(0, u - 250):u]) for u in t])
    n10 = np.array([own[max(0, u - 10):u].sum() for u in t])
    ab = np.asarray(F.above200[t - 1])
    clv = np.asarray(F.ew_ibs[t], float)
    ewi = np.array([np.nanmean(np.where(np.asarray(F.liquid[u]), np.clip(np.asarray(F.intra[u], float), -0.3, 0.3), nan))
                    for u in t])
    gapd = np.abs(np.asarray(F.ew_gap[t], float)) > np.abs(ewi)
    dts = [ymd(F.dates[u]) for u in t]
    prv = [ymd(F.dates[u - 1]) for u in t]
    mon = np.array([d.weekday() == 0 for d in dts])
    post = np.array([np.busday_count(p + _dt.timedelta(days=1), d) >= 1 for p, d in zip(prv, dts)])
    S = [("F-a (mkt_on[t−1] 또는 deepcap)", on1 | deep, "통과", "실패", False),
         ("mkt_on[t−1]", on1, "켜짐", "꺼짐", False), ("deepcap", deep, "예", "아니오", False)]
    for d in (5, 8, 12):
        S.append((f"F-c dd60[t−1] ≥ −{d}%", dd60 >= -d / 100, "고점 근처", "하락 중", False))
    S += [("변동성 vol20[t−1] ≤ 250일 중앙", calm, "평온", "긴장", False),
          ("연쇄 n10 (t−10..t−1)", n10 == 0, "첫 패닉", "≥1", False),
          ("200일선 위 [t−1]", ab, "위", "아래", False)]
    for q in (0.3, 0.4, 0.5):
        S.append((f"CLV (EW ibs_t) ≥ {q:g}", clv >= q, "흡수", "저가 마감", False))
    S += [("갭 주도 |EW 갭| > |EW 장중|", gapd, "갭", "장중", False),
          ("월요일 (보고만)", mon, "예", "아니오", True), ("연휴 다음 날 (보고만)", post, "예", "아니오", True)]
    return S


def usable(dA, vA, dB, vB):
    a, b = ept(dA, vA), ept(dB, vB)
    if a["N"] < 4 or b["N"] < 4:
        return False
    gm, gd = a["mean"] - b["mean"], a["median"] - b["median"]
    return bool(np.sign(gm) == np.sign(gd) and abs(gd) >= abs(gm))


CELLS5 = [("K2", 5), ("K2", 21), ("S-RES", 5), ("S-RES", 21)]


def eb5_core(F, days, T0full):
    """T0 날들의 Δswitch (시가 t+1 · 종가 t) · 국면 구분 · '사용' 표시 {구분: (시가 표시, 종가 표시)}."""
    J = {"K2": tops(F, "K2", days, 2), "S-RES": tops(F, "SRES", days, 2)}
    Do = {(s, h): dsw(F, days, J[s], "open", h) for s in J for h in (1, 5, 21)}
    Dc = {(s, h): dsw(F, days, J[s], "close", h) for s in J for h in (1, 5, 21)}
    S = classes(F, days, T0full)
    fl = {}
    for nm, m, la, lb, rep in S:
        f_ = "-" if rep else "·".join("Y" if usable(days[m], Do[c][m], days[~m], Do[c][~m]) else "N" for c in CELLS5)
        fc = None
        if nm.startswith("CLV") or nm.startswith("갭"):
            hs = (1, 5, 21) if nm.startswith("CLV") else (5, 21)
            fc = "·".join("Y" if usable(days[m], Dc[c][m], days[~m], Dc[c][~m]) else "N"
                          for c in [(s, h) for s in ("K2", "S-RES") for h in hs])
        fl[nm] = (f_, fc)
    return Do, Dc, S, fl


def eb5(F, t0, t1, X):
    x = np.asarray(F.x, float)
    T0full = x <= -0.04
    days = np.where(T0full[t0:t1])[0] + t0
    L = ["### EB5 국면 나누기 (T0, K2 · S-RES top-2 Δswitch)",
         "T0 날을 t−1 까지 정보로 (CLV · 갭 은 t) 둘로 나눠 쪽마다 에피소드(G=10, 쪽 안) Δswitch. '사용' = IS 에서 양쪽 에피소드 ≥4 "
         "이고 중앙값 차이가 평균 차이와 같은 부호 · 크기 이상 (K2 5d / K2 21d / S-RES 5d / S-RES 21d 순).", ""]
    if len(days) < 2:
        return L + ["(표본 부족)"], {}
    Do, Dc, S, fl_cur = eb5_core(F, days, T0full)
    ia, iz = C.period_range(F, "IS")
    if (ia, iz) == (t0, t1):
        fl_is = fl_cur
    else:                                                    # '사용' 은 IS 로만 정한다 (VAL · TEST 는 IS 표시를 그대로 보여 줌)
        dis = np.where(T0full[ia:iz])[0] + ia
        fl_is = eb5_core(F, dis, T0full)[3] if len(dis) >= 2 else {k: ("N", "N") for k in fl_cur}
    Tm = head("구분", "쪽", "일수", "에피소드", "K2 Δ5 평균", "중앙", "최악", "K2 Δ21 평균", "중앙", "S-RES Δ5 평균", "중앙",
              "S-RES Δ21 평균", "중앙", "사용")
    Tc = head("구분", "쪽", "에피소드", "K2 E0 1d", "K2 E1 1d", "K2 E0 5d", "K2 E0 21d", "S-RES E0 1d", "S-RES E1 1d",
              "S-RES E0 5d", "S-RES E0 21d", "사용 (E0)")
    out = {}
    cells = CELLS5
    for nm, m, la, lb, rep in S:
        fl = fl_is[nm][0]
        o = {"usable": fl, "usable_this_period": fl_cur[nm][0]}
        for side, mm, sl in (("A", m, la), ("B", ~m, lb)):
            dd = days[mm]
            st = {c: ept(dd, Do[c][mm]) for c in cells}
            o[side] = {"label": sl, "n_days": int(mm.sum()), "n_ep": nep(dd),
                       **{f"{c[0]}_{c[1]}": {k: st[c][k] for k in ("mean", "median", "worst")} for c in cells}}
            Tm.append(row(nm if side == "A" else "", sl, int(mm.sum()), nep(dd), fp(st[("K2", 5)]["mean"]),
                          fp(st[("K2", 5)]["median"]), fp(st[("K2", 5)]["worst"]), fp(st[("K2", 21)]["mean"]),
                          fp(st[("K2", 21)]["median"]), fp(st[("S-RES", 5)]["mean"]), fp(st[("S-RES", 5)]["median"]),
                          fp(st[("S-RES", 21)]["mean"]), fp(st[("S-RES", 21)]["median"]), fl if side == "A" else ""))
        if nm.startswith("CLV") or nm.startswith("갭"):
            flc = fl_is[nm][1]
            o["close"] = {"usable": flc, "usable_this_period": fl_cur[nm][1]}
            for side, mm, sl in (("A", m, la), ("B", ~m, lb)):
                dd = days[mm]
                e = lambda D_, s, h: ept(dd, D_[(s, h)][mm])["mean"]
                o["close"][side] = {f"{s}_{h}": e(Dc, s, h) for s in ("K2", "S-RES") for h in (1, 5, 21)}
                o["close"][side].update({f"{s}_E1_1": e(Do, s, 1) for s in ("K2", "S-RES")})
                Tc.append(row(nm if side == "A" else "", sl, nep(dd), fp(e(Dc, "K2", 1), 2), fp(e(Do, "K2", 1), 2),
                              fp(e(Dc, "K2", 5)), fp(e(Dc, "K2", 21)), fp(e(Dc, "S-RES", 1), 2), fp(e(Do, "S-RES", 1), 2),
                              fp(e(Dc, "S-RES", 5)), fp(e(Dc, "S-RES", 21)), flc if side == "A" else ""))
        out[nm] = o
    L += Tm + ["", "CLV · 갭 구분의 종가 t 진입 (E0) 과 1일 (E1 = 시가 t+1 → 종가 t+1). 사용 표시는 E0 의 K2 · S-RES × 지평 순.", ""] + Tc
    L.append("F-a 의 deepcap = t−40..t−1 T0 2번↑ & M_t/max(M_{t−249..t})−1 ≤ −25%. 연쇄 n10 = t−10..t−1 의 같은 트리거 날 수. "
             "변동성 기준 = mkt_vol20[t−250..t−1] 중앙. 연휴 다음 날 = 직전 거래일과 사이에 평일 휴장 1일↑ (설계 괄호 정의).")
    # T1 · T2 칸 (서술)
    Tt = head("트리거", "구분", "A 쪽", "A 에피", "A K2 Δ5", "B 쪽", "B 에피", "B K2 Δ5")
    desc = {}
    for tn, key in (("T1 Z=4", "trig_T1"), ("T2 b=0.75", "trig_T2")):
        own = np.asarray(getattr(F, key)[:t1])
        dd = np.where(own[t0:t1])[0] + t0
        if len(dd) < 2:
            continue
        v5 = dsw(F, dd, tops(F, "K2", dd, 2), "open", 5)
        for nm, m, la, lb, rep in classes(F, dd, own):
            if not (nm.startswith("F-a") or nm.startswith("F-c dd60[t−1] ≥ −8") or nm.startswith("변동성")
                    or nm.startswith("200일") or nm.startswith("연쇄")):
                continue
            a, b = ept(dd[m], v5[m]), ept(dd[~m], v5[~m])
            desc.setdefault(tn, {})[nm] = {"A": [la, a["N"], a["mean"]], "B": [lb, b["N"], b["mean"]]}
            Tt.append(row(tn, nm, la, a["N"], fp(a["mean"]), lb, b["N"], fp(b["mean"])))
    L += ["", "T1 · T2 칸 (서술, K2 top-2 Δswitch 5d 에피소드 평균)", ""] + Tt
    return L, {"splits": out, "t1_t2": desc}


# ════════════════════════════════════════════════════════════
# EB6 청산 · 슬롯
# ════════════════════════════════════════════════════════════
HM = 21


def bpath(F, days, J):
    """바구니 시가 t+1 진입 → 1..HM 일차 종가 gross 경로 (종목 평균)."""
    J = np.asarray(J, np.int64)
    P = np.full(J.shape + (HM,), nan)
    v = J >= 0
    if v.any():
        tt = np.broadcast_to(days[:, None], J.shape)
        P[v] = C.fwd_path(F, tt[v], J[v], "open", HM)
    return np.nanmean(P, 1)


def first_hit(cond, hmax, hmin=2):
    """보유일 hmin..hmax 중 조건이 처음 참인 날 (없으면 hmax). 시장 기준 청산은 진입 다음 날부터 (panic_sim2 와 같게)."""
    sub = cond[:, hmin - 1:hmax]
    return np.where(sub.any(1), sub.argmax(1) + hmin, hmax)


def eb6(F, t0, t1, X):
    x = np.asarray(F.x, float)
    days = np.where(x[t0:t1] <= -0.04)[0] + t0
    L = ["### EB6 청산 규칙 · 슬롯 방식 (T0)",
         "K2 · S-RES top-2 를 시가 t+1 에 사서 청산 규칙별로 판 Δswitch (평소 top-2 를 같은 날까지 들고 있던 것 대비, 비용 0.35%) 의 "
         "에피소드(G=10) 평균 · 중앙 · 최악, 그리고 슬롯 방식별 에피소드 값.", ""]
    if len(days) < 2:
        return L + ["(표본 부족)"], {}
    n = len(days)
    M = np.asarray(F.M, float)
    Mp = np.array([[M[t + h] if t + h < F.T else nan for h in range(1, HM + 1)] for t in days])
    M0 = M[days][:, None]
    Mpre = np.array([M[t - 20:t].max() for t in days])[:, None]
    Mlow = np.minimum(np.minimum.accumulate(np.where(np.isfinite(Mp), Mp, np.inf), 1), M0)
    P = {s: bpath(F, days, tops(F, k, days, 2)) for s, k in (("K2", "K2"), ("S-RES", "SRES"))}
    Pb = bpath(F, days, tops(F, "base", days, 2))
    rules = [(f"고정 H={H}", np.full(n, H)) for H in (3, 5, 7, 10, 15, 21)]
    for X_ in (0.03, 0.06, 0.09):
        for Hm in (10, 21):
            rules.append((f"X-MKT +{X_:.0%} · 최대 {Hm}", first_hit(Mp >= M0 * (1 + X_), Hm)))
    for Y in (0.06, 0.10):
        for Hm in (10, 21):
            rules.append((f"시장 손절 −{Y:.0%} · 최대 {Hm}", first_hit(Mp <= M0 * (1 - Y), Hm)))
    for Y in (0.06, 0.10):
        for Hm in (10, 21):
            rules.append((f"X-MKT +6% + 손절 −{Y:.0%} · 최대 {Hm}", first_hit((Mp >= M0 * 1.06) | (Mp <= M0 * (1 - Y)), Hm)))
    for f in (0.5, 1.0):
        rules.append((f"회복 f={f:g} · 최대 21", first_hit(Mp >= Mlow + f * (Mpre - Mlow), 21)))
    T = head("청산", "K2 Δ 평균", "중앙", "최악", "보유일", "K2 net", "S-RES Δ 평균", "중앙", "최악", "보유일", "S-RES net")
    out = {"exits": {}}
    ar = np.arange(n)
    for nm, he in rules:
        cells, o = [], {}
        for s in ("K2", "S-RES"):
            g = P[s][ar, he - 1]
            d = g - COST - Pb[ar, he - 1] - COST
            st, sn = ept(days, d), ept(days, g - COST)
            hold = float(np.mean(he[np.isfinite(d)])) if np.isfinite(d).any() else nan
            o[s] = {"mean": st["mean"], "median": st["median"], "worst": st["worst"], "hold": hold, "net": sn["mean"],
                    "N": st["N"]}
            cells += [fp(st["mean"]), fp(st["median"]), fp(st["worst"]), f"{hold:.1f}" if np.isfinite(hold) else "-",
                      fp(sn["mean"])]
        out["exits"][nm] = o
        T.append(row(nm, *cells))
    L += T
    pl = {}
    for s in ("K2", "S-RES"):
        for H, (a_, b_) in ((5, (3, 7)), (10, (7, 15))):
            m0, ma, mb = (out["exits"][f"고정 H={k}"][s]["mean"] for k in (H, a_, b_))
            ok = all(np.isfinite(v) for v in (m0, ma, mb)) and all(np.sign(v) == np.sign(m0) and abs(v) >= 0.5 * abs(m0)
                                                                   for v in (ma, mb))
            pl[f"{s}_H{H}"] = bool(ok)
    out["plateau"] = pl
    L.append("시장 기준 청산(X-MKT · 손절 · 회복)은 진입 다음 날(보유 2일차) 종가부터 판정 (panic_sim2 의 after 규칙), 기준 M_t0 = 그 신호일 M. "
             "회복: M_u ≥ M_low + f·(M_pre − M_low), M_pre = max(M_{t−20..t−1}), M_low = min(M_{t..u}). 고원(plateau) 검사 "
             "(H±2 같은 부호 · 크기 50%↑): " + ", ".join(f"{k} {'통과' if v else '실패'}" for k, v in pl.items()) + ".")

    # 슬롯 방식 (에피소드 단위)
    T0full = x <= -0.04
    ep = C.episodes_of(days, G)
    E = int(ep.max()) + 1

    def fv(t, key, held, cap=None, k=1):
        """t 의 후보 목록에서 보유 중이 아니고 (가격 상한) 시가 t+1 에 살 수 있는 첫 k 종목 → [(종목, net 5d)].
        건너뛰기는 진입 가능 여부(시가 t+1 에 아는 것)로만 정한다. 다시 거래되지 않으면 (상장폐지 등) 마지막 거래 종가로 평가."""
        o = []
        for j in getattr(F, "top_" + key)[t]:
            j = int(j)
            if j < 0 or j in held or (cap and not F.raw[t, j] <= cap):
                continue
            tj, jj = np.array([t]), np.array([j])
            if not np.isfinite(C.entry_price(F, tj, jj, "open")[0]):
                continue
            o.append((j, float(C.fwd(F, tj, jj, "open", 5)[0][0])))
            held = held | {j}
            if len(o) >= k:
                break
        return o

    def bg(t, w):
        """슬롯이 바꾼 평소 보유 (순위 w) 의 같은 창 gross. 없으면 top-2 평균."""
        v = C.base_gross(F, np.array([t]), "open", 5, which=(w,))[0]
        return v if np.isfinite(v) else C.base_gross(F, np.array([t]), "open", 5)[0]

    sch = {}
    cause = {D_: {"T0": 0, "M": 0, "없음": 0} for D_ in (0.03, 0.06)}      # 2슬롯 발동 원인 (K2 기준 셈)
    for s, key in (("K2", "K2"), ("S-RES", "SRES")):
        res = {k: [] for k in ("ALL-IN 2", "분할 1+1 D=3%", "분할 1+1 D=6%", "3슬롯 패닉일마다 (20만원↓)", "3슬롯 전량 (20만원↓)")}
        used = {k: [] for k in res}
        for e in range(E):
            ed = days[ep == e]
            t_ = int(ed[0])
            if t_ + 5 >= F.T:                                     # 데이터 끝: 창이 안 끝난 에피소드는 0 이 아니라 제외
                for k_ in res:
                    res[k_].append(nan)
                    used[k_].append(0)
                continue
            pk = fv(t_, key, set(), k=2)                          # 전량 2: 첫 슬롯이 순위 2 평소 보유를 바꾼다
            res["ALL-IN 2"].append(sum(v - bg(t_, 1 - i) - COST for i, (j, v) in enumerate(pk)) / 2)
            used["ALL-IN 2"].append(len(pk))
            for D_ in (0.03, 0.06):
                nmk = f"분할 1+1 D={D_:.0%}"
                p1 = fv(t_, key, set())
                val = sum(v - bg(t_, 1) - COST for j, v in p1)
                cnt = len(p1)
                why = "없음"
                for t2 in range(t_ + 1, min(t_ + 11, F.T)):
                    if T0full[t2] or M[t2] <= M[t_] * (1 - D_):
                        why = "T0" if T0full[t2] else "M"
                        p2 = fv(t2, key, {j for j, _ in p1})
                        val += sum(v - bg(t2, 0) - COST for j, v in p2) if t2 + 5 < F.T else nan
                        cnt += len(p2)
                        break
                else:
                    if t_ + 10 >= F.T:                            # 2슬롯 발동 창이 데이터 끝에서 잘림 → 모름
                        val = nan
                if s == "K2":
                    cause[D_][why] += 1
                res[nmk].append(val / 2)
                used[nmk].append(cnt)
            held, val, cnt = set(), 0.0, 0                          # 3슬롯: 에피소드 안 패닉일마다 한 슬롯
            for d in ed:
                if cnt >= 3:
                    break
                if d + 5 >= F.T:
                    val = nan
                    break
                p = fv(int(d), key, held, cap=CAP3)
                for j, v in p:
                    val += v - bg(int(d), 2 - cnt) - COST
                    held.add(j)
                    cnt += 1
            res["3슬롯 패닉일마다 (20만원↓)"].append(val / 3)
            used["3슬롯 패닉일마다 (20만원↓)"].append(cnt)
            p3 = fv(t_, key, set(), cap=CAP3, k=3)
            res["3슬롯 전량 (20만원↓)"].append(sum(v - bg(t_, 2 - i) - COST for i, (j, v) in enumerate(p3)) / 3)
            used["3슬롯 전량 (20만원↓)"].append(len(p3))
        sch[s] = {k: (np.array(v, float), float(np.mean(used[k]))) for k, v in res.items()}
    Ts = head("슬롯 방식", "에피소드 (K2/S-RES)", "K2 평균", "중앙", "최악", "쓴 슬롯", "S-RES 평균", "중앙", "최악", "쓴 슬롯")
    out["slots"] = {}
    for k in sch["K2"]:
        cells, ns = [], []
        for s in ("K2", "S-RES"):
            v, u = sch[s][k]
            st = em_stats(v)
            wv = float(np.nanmin(v)) if np.isfinite(v).any() else nan
            out["slots"].setdefault(k, {})[s] = {"N": st["N"], "mean": st["mean"], "median": st["median"], "worst": wv,
                                                 "slots": u, "ep_vals": v.tolist()}
            cells += [fp(st["mean"]), fp(st["median"]), fp(wv), f"{u:.1f}"]
            ns.append(st["N"])
        Ts.append(row(k, f"{ns[0]}/{ns[1]}", *cells))
    out["stage2_cause"] = {f"D={k:.0%}": v for k, v in cause.items()}
    L += ["", "슬롯 방식 (에피소드 첫 T0 날 t0 기준, 슬롯마다 FIX5, 값 = 슬롯 Δ 의 합 / 슬롯 수, 안 쓴 슬롯은 평소 전략 = 0)", ""] + Ts
    L.append("분할 1+1: t0+1 시가에 1슬롯 (평소 순위 2 를 판다), t0+1..t0+10 중 새 T0 또는 M ≤ M_t0·(1−D) 인 첫 종가 다음 시가에 "
             "그날 목록 1위 (보유 제외) 로 2슬롯 (평소 순위 1 을 판다). 2슬롯 발동 원인 (새 T0 / M 하락 / 없음): "
             + ", ".join(f"D={k:.0%} {v['T0']}/{v['M']}/{v['없음']}" for k, v in cause.items())
             + ". 3슬롯: 같은 에피소드 패닉일마다 20만원 이하 1위 종목, 평소 비교는 top_base 3위→1위 순. "
               "시가 t+1 에 살 수 없는 후보만 건너뛰고 다음 순위 (청산가 유무는 보지 않음; 다시 거래되지 않으면 마지막 거래 종가로 평가).")
    return L, out


# ════════════════════════════════════════════════════════════
# EB7 위약
# ════════════════════════════════════════════════════════════
def pge(dist, actual):
    """위약 분포에서 실제 이상인 비율 (값이 없으면 NaN)."""
    d = np.asarray(dist, float)
    d = d[np.isfinite(d)]
    return float((d >= actual).mean()) if len(d) and np.isfinite(actual) else nan


def fpv(p):
    return "-" if not np.isfinite(p) else f"{p:.3f}"


def vol_terc(v20, u):
    d = v20[max(0, u - 750):u]
    d = d[np.isfinite(d)]
    if not len(d) or not np.isfinite(v20[u - 1]):
        return -1
    p = (d <= v20[u - 1]).mean()
    return 0 if p <= 1 / 3 else (1 if p <= 2 / 3 else 2)


def eb7(F, t0, t1, X):
    x = np.asarray(F.x, float)
    T0d = np.where(x[t0:t1] <= -0.04)[0] + t0
    L = ["### EB7 위약 (짝지은 평온일 · 시간 이동 · 무작위 종목)",
         "에피소드(G=10) 첫 T0 날 t_e 의 K2 · S-RES top-2 · top-5 (시가 다음 날, net 0.35%) 를 세 가지 위약과 비교. "
         "p = 위약 에피소드 평균(중앙) ≥ 실제 인 비율 (2000회).", ""]
    if len(T0d) < 2:
        return L + ["(표본 부족)"], {}
    te = T0d[first_days(T0d)]
    E = len(te)
    near = np.where(x[max(0, t0 - 21):t1] <= -0.04)[0] + max(0, t0 - 21)
    cand = np.arange(t0, max(t0, t1 - 21))                  # u ≤ t1−22: 21일 창도 기간 안 신호일 규칙, 기간 밖 T0 는 안 본다
    cand = cand[x[cand] > -0.02]
    dist = np.min(np.abs(cand[:, None] - near[None, :]), 1) if len(near) else np.full(len(cand), 999)
    pool = cand[dist >= 21]
    if not len(pool):
        return L + ["(위약 후보일 없음)"], {"n_ep": E, "pool": 0}
    v20, ab = np.asarray(F.mkt_vol20, float), np.asarray(F.above200)
    kp = np.array([(vol_terc(v20, u), bool(ab[u - 1])) for u in pool], dtype=object)
    ke = [(vol_terc(v20, int(t)), bool(ab[int(t) - 1])) for t in te]
    rng = np.random.default_rng(SEED)
    fb = [0, 0]
    draws = np.zeros((E, NB), np.int64)
    for e, (tv, ta) in enumerate(ke):
        P = [i for i, (a_, b_) in enumerate(kp) if a_ == tv and b_ == ta]
        if not P:
            fb[0] += 1
            P = [i for i, (a_, _) in enumerate(kp) if a_ == tv]
        if not P:
            fb[1] += 1
            P = list(range(len(pool)))
        draws[e] = rng.choice(np.array(P, np.int64), NB)
    T = head("규칙", "크기", "h", "측정", "에피소드", "실제 평균", "실제 중앙", "위약 평균", "p(평균)", "p(중앙)", "+21 평균",
             "+42 평균", "셔플 p(평균)", "셔플 p(중앙)")
    out = {"n_ep": E, "pool": int(len(pool)), "fallback": fb, "cells": {}}
    Uc = {}
    for rn, key in (("K2", "K2"), ("S-RES", "SRES")):
        for k in (2, 5):
            Je, Jp = tops(F, key, te, k), tops(F, key, pool, k)
            for h in (5, 21):
                be, bp_ = C.base_gross(F, te, "open", h), C.base_gross(F, pool, "open", h)
                ne, npl = bask(F, te, Je, "open", h) - COST, bask(F, pool, Jp, "open", h) - COST
                meas = {"net": (ne, npl, np.zeros(E)), "Δ": (ne - be - COST, npl - bp_ - COST, be + COST)}
                sh = {}
                for kk, sft in (("+21", 21), ("+42", 42)):
                    s_ = te + sft
                    ok = s_ < t1
                    vn = np.full(E, nan)
                    vd = np.full(E, nan)
                    if ok.any():
                        vn[ok] = bask(F, s_[ok], Je[ok], "open", h) - COST
                        vd[ok] = vn[ok] - C.base_gross(F, s_[ok], "open", h) - COST
                    sh[kk] = {"net": vn, "Δ": vd}
                # 셔플: t_e 의 U200 무작위 k 종목
                ck = (h, k)
                if ck not in Uc:
                    Rm = np.full((E, NB), nan)
                    for e, t in enumerate(te):
                        jj = np.where(np.asarray(F.U200[t]))[0]
                        g = C.fwd(F, np.full(len(jj), t), jj, "open", h)[1]
                        g = g[np.isfinite(g)]
                        if len(g) >= k:
                            Rm[e] = g[rng.random((NB, len(g))).argsort(1)[:, :k]].mean(1)
                    Uc[ck] = Rm - COST
                for mn, (va, vp, adj) in meas.items():
                    ok = np.isfinite(va)                                    # 실제 값이 있는 에피소드만 위약과 짝짓는다
                    am, amed = (float(np.mean(va[ok])), float(np.median(va[ok]))) if ok.any() else (nan, nan)
                    pv = vp[draws[ok]]                                      # E × NB
                    pm, pmed = np.nanmean(pv, 0), np.nanmedian(pv, 0)
                    su = Uc[ck][ok] - adj[ok, None]
                    sm, smed = np.nanmean(su, 0), np.nanmedian(su, 0)
                    c = {"n_ep": int(ok.sum()), "actual_mean": am, "actual_median": amed,
                         "placebo_mean": float(np.nanmean(pm)) if ok.any() else nan,
                         "p_mean": pge(pm, am), "p_median": pge(pmed, amed),
                         "shift21": float(np.nanmean(sh["+21"][mn])), "shift42": float(np.nanmean(sh["+42"][mn])),
                         "shift21_median": float(np.nanmedian(sh["+21"][mn])), "shift42_median": float(np.nanmedian(sh["+42"][mn])),
                         "shuffle_mean": float(np.nanmean(sm)) if ok.any() else nan,
                         "shuf_p_mean": pge(sm, am), "shuf_p_median": pge(smed, amed)}
                    out["cells"][f"{rn}_top{k}_h{h}_{mn}"] = c
                    T.append(row(rn, k, h, mn, int(ok.sum()), fp(am), fp(amed), fp(c["placebo_mean"]), fpv(c["p_mean"]),
                                 fpv(c["p_median"]), fp(c["shift21"]), fp(c["shift42"]), fpv(c["shuf_p_mean"]),
                                 fpv(c["shuf_p_median"])))
    L += T
    L.append(f"짝지은 평온일 후보 {len(pool)}일: 기간 안 x_u > −2%, 모든 T0 날과 21 거래일↑ 떨어짐, u ≤ 기간 끝−22 (21일 창도 기간 안), "
             f"t_e 와 mkt_vol20[u−1] 의 직전 750일 분포 3분위 · above200[u−1] 가 같은 날에서 에피소드마다 하나씩 뽑음 "
             f"(맞는 날 없어 3분위만 맞춤 {fb[0]}번, 아무 날 {fb[1]}번). 시간 이동 = t_e 바구니를 t_e+21 · +42 날 다음 시가에 삼 "
             f"(그날이 기간 밖이면 뺌). 셔플 = t_e 의 U200 에서 무작위 k 종목. Δ 셔플은 실제 t_e 의 평소 top-2 를 뺌.")
    return L, out


# ════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════
def run(F, period: str):
    """(보고서 줄, 요약 dict). 모든 통계는 period 신호일만."""
    t0, t1 = C.period_range(F, period)
    lines = [f"트랙 B 사건 조사 — 기간 {period} (신호일 {F.dates[t0]} ~ {F.dates[t1 - 1]}), T0 = x ≤ −4% "
             f"{int((np.asarray(F.x[t0:t1]) <= -0.04).sum())}일. 비용 왕복 0.35%, Δswitch = 바구니 net − 평소 top-2 gross − 0.35%.", ""]
    summary = {"period": period, "t0": F.dates[t0], "t1": F.dates[t1 - 1]}
    X = {}
    for nm, fn_ in (("EB1", eb1), ("EB2", eb2), ("EB3", eb3), ("EB4", eb4), ("EB5", eb5), ("EB6", eb6), ("EB7", eb7)):
        L, s = fn_(F, t0, t1, X)
        lines += L + [""]
        summary[nm] = s
    return lines, clean(summary)


if __name__ == "__main__":
    _t = time.time()
    _F = PF.load()
    _lines, _s = run(_F, "IS")
    print("\n".join(_lines))
    print(f"\n({time.time() - _t:.0f}s)", file=sys.stderr)
