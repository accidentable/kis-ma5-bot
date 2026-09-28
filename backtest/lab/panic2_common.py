"""
backtest/lab/panic2_common.py — 패닉 2차 연구 공통 도구 (사건 수익률 · 통계 · 에피소드)

사건 수익률 규칙 (모든 사건 조사와 관문 판정이 같은 규칙을 쓴다)
  진입  close      t 종가 (그날 거래돼야 함)
        open       t+1 시가 (t+1 에 거래되고 시가 < 1.295 × c_t)
        open_same  t 시가 (갭 확인 직후 매수, 추가 슬리피지 0.3%)
        open2      t+2 시가 (t+2 에 거래되고 시가 < 1.295 × c_{t+1})
        close1 / close2   t+1 / t+2 종가
  청산  보유 h 일 뒤 종가. 진입일을 1일차로 센다 (대회 FIX 와 같다):
        close → t+h,  open → t+h,  open_same → t+h−1,  open2 → t+1+h,  close1 → t+1+h,  close2 → t+2+h
        청산일에 거래정지면 그 뒤 처음 거래된 날 종가 (cnext). 다시 거래되지 않으면 (상장폐지) 마지막 거래 종가 (clast).
  비용  왕복 0.35% (COST). gross 는 비용 전.
"""
from __future__ import annotations

import numpy as np

COST = 0.0035
PERIODS = {"IS": ("20110101", "20190101"), "VAL": ("20190101", "20241001"), "TEST": ("20241001", "99999999")}
ERAS = {"E1": ("20100701", "20150615"), "E2": ("20150615", "20190101"), "E3": ("20190101", "20241001"),
        "E4": ("20241001", "99999999")}
ENTRY_DAY = {"close": 0, "open": 1, "open_same": 0, "open_same0": 0, "open2": 2, "close1": 1, "close2": 2}
EXIT_OFF = {"close": 0, "open": 0, "open_same": -1, "open_same0": -1, "open2": 1, "close1": 1, "close2": 2}


def didx(F, d: str) -> int:
    return int(np.searchsorted(F.dates, d))


def period_range(F, name: str):
    a, z = PERIODS[name]
    return didx(F, a), didx(F, z)


# ════════════════════════════════════════════════════════════
# 사건 목록
# ════════════════════════════════════════════════════════════
def events_from_top(top: np.ndarray, n: int, t0: int, t1: int):
    """T×12 후보 목록 (−1 없음) → 날짜별 상위 n 개 사건 (t, j, 순위)."""
    sub = np.asarray(top[t0:t1, :n])
    tt, rr = np.where(sub >= 0)
    return tt + t0, sub[tt, rr].astype(np.int64), rr


def events_from_mask(mask: np.ndarray, score: np.ndarray | None, n: int | None, t0: int, t1: int):
    """T×N bool 사건 → (t, j). n 이 있으면 날짜별 score 큰 순 상위 n 개만 (동점은 인덱스 작은 쪽)."""
    msk = np.asarray(mask[t0:t1])
    if n is None:
        tt, jj = np.where(msk)
        return tt + t0, jj.astype(np.int64)
    sc = np.where(msk, np.asarray(score[t0:t1], dtype=np.float64), -np.inf)
    sc = np.where(np.isfinite(sc) | ~msk, sc, -1e300)          # 사건인데 점수가 NaN 이면 맨 뒤
    out_t, out_j = [], []
    for i in np.where(msk.any(1))[0]:
        idx = np.argsort(-sc[i], kind="stable")[:n]
        idx = idx[msk[i, idx]]
        out_t.append(np.full(len(idx), i + t0))
        out_j.append(idx)
    if not out_t:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_t), np.concatenate(out_j).astype(np.int64)


# ════════════════════════════════════════════════════════════
# 사건 수익률
# ════════════════════════════════════════════════════════════
def entry_price(F, t, j, entry: str):
    """진입 가격 (조정가) — 조건이 안 맞으면 NaN."""
    t = np.asarray(t, np.int64)
    j = np.asarray(j, np.int64)
    T = F.T
    d = ENTRY_DAY[entry]
    te = t + d
    okd = te < T
    te_c = np.minimum(te, T - 1)
    tr = F.traded[te_c, j] & okd
    if entry in ("close", "close1", "close2"):
        px = F.c[te_c, j].astype(np.float64)
        prevc = F.c[np.maximum(te_c - 1, 0), j].astype(np.float64)
        tr = tr & ~(px >= prevc * 1.295)
    elif entry in ("open_same", "open_same0"):               # open_same0: 추가 슬리피지 없는 당일 시가 (평소 보유 비교용)
        px = F.o[te_c, j].astype(np.float64) * (1 + (0.003 if entry == "open_same" else 0.0))
    else:
        px = F.o[te_c, j].astype(np.float64)
        prevc = F.c[te_c - 1, j].astype(np.float64)
        tr = tr & (px > 0) & (px < prevc * 1.295)
    return np.where(tr & (px > 0), px, np.nan)


def exit_price(F, t, j, entry: str, h: int):
    t = np.asarray(t, np.int64)
    u = t + h + EXIT_OFF[entry]
    okd = u < F.T
    u = np.minimum(u, F.T - 1)
    jj = np.asarray(j, np.int64)
    px = F.cnext[u, jj].astype(np.float64)
    px = np.where(np.isfinite(px), px, F.clast[u, jj])    # 다시 거래되지 않으면 (상장폐지) 마지막 거래 종가
    return np.where(okd, px, np.nan)


def fwd(F, t, j, entry: str, h: int):
    """(net, gross). 진입 불가 · 창이 데이터 끝을 넘으면 NaN."""
    g = exit_price(F, t, j, entry, h) / entry_price(F, t, j, entry) - 1
    return g - COST, g


def fwd_path(F, t, j, entry: str, hmax: int):
    """진입 뒤 1..hmax 일차 종가 수익률 행렬 (E × hmax, gross, 정지일은 다음 거래 종가)."""
    ep = entry_price(F, t, j, entry)
    out = np.full((len(ep), hmax), np.nan)
    for h in range(1, hmax + 1):
        out[:, h - 1] = exit_price(F, t, j, entry, h) / ep - 1
    return out


def base_gross(F, t, entry: str, h: int, which=(0, 1)):
    """같은 창에서 평소 전략 순위 which 종목들의 평균 비용 전 수익률. 순위는 날짜 t 의 top_base
    (당일 시가 진입 open_same 은 시가에 아는 t−1 순위 — 시뮬레이터가 그 순위로 판다)."""
    t = np.asarray(t, np.int64)
    rd = t - 1 if entry == "open_same" else t
    entry = "open_same0" if entry == "open_same" else entry     # 평소 보유는 추가 슬리피지 없이 시가에 판다
    acc = np.zeros(len(t))
    cnt = np.zeros(len(t))
    for w in which:
        jj = F.top_base[rd, w].astype(np.int64)
        v = jj >= 0
        g = np.full(len(t), np.nan)
        if v.any():
            g[v] = fwd(F, t[v], jj[v], entry, h)[1]
        ok = np.isfinite(g)
        acc[ok] += g[ok]
        cnt[ok] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def same_day_ew(F, days, universe: np.ndarray, entry: str, h: int) -> dict:
    """날짜별 유니버스 동일가중 비용 전 수익률 (같은 진입 · 청산). {t: 평균}."""
    out = {}
    for t in np.unique(days):
        jj = np.where(universe[t])[0]
        if not len(jj):
            continue
        g = fwd(F, np.full(len(jj), t), jj, entry, h)[1]
        if np.isfinite(g).any():
            out[int(t)] = float(np.nanmean(g))
    return out


# ════════════════════════════════════════════════════════════
# 통계
# ════════════════════════════════════════════════════════════
def nw_t(x: np.ndarray, lag: int) -> float:
    """평균의 Newey–West t (Bartlett 가중)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    s = (e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(mu / np.sqrt(max(s, 1e-18) / n))


def daily_mean(t, v):
    """사건 값을 날짜 평균으로 (날짜 순)."""
    t = np.asarray(t)
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if not len(t):
        return np.zeros(0, int), np.zeros(0)
    u, inv = np.unique(t, return_inverse=True)
    s = np.bincount(inv, weights=v)
    n = np.bincount(inv)
    return u, s / n


def episodes_of(days: np.ndarray, gap: int = 10) -> np.ndarray:
    """정렬된 날짜 배열 → 같은 사건 번호 (gap 거래일 이내면 같은 사건)."""
    days = np.asarray(days)
    ep = np.zeros(len(days), int)
    cur, last = -1, -10 ** 9
    for i, t in enumerate(days):
        if t - last > gap:
            cur += 1
        ep[i] = cur
        last = t
    return ep


def ep_table(days, vals, gap: int = 10) -> dict:
    """날짜별 값 → 에피소드 평균들의 통계 (N, 평균, 중앙값, 양수 비율, 최악, LOO 최소·최대, 최고 1·2개 뺀 평균)."""
    days = np.asarray(days)
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    days, vals = days[ok], vals[ok]
    o = np.argsort(days)
    days, vals = days[o], vals[o]
    if not len(days):
        return {"N": 0}
    ep = episodes_of(days, gap)
    em = np.array([vals[ep == e].mean() for e in range(ep.max() + 1)])
    first = np.array([days[ep == e][0] for e in range(ep.max() + 1)])
    loo = np.array([np.delete(em, i).mean() for i in range(len(em))]) if len(em) > 1 else np.array([np.nan])
    srt = np.sort(em)[::-1]
    return {"N": int(len(em)), "mean": float(em.mean()), "median": float(np.median(em)),
            "pos": float((em > 0).mean()), "worst": float(em.min()), "best": float(em.max()),
            "loo_min": float(np.nanmin(loo)), "loo_max": float(np.nanmax(loo)),
            "no_best1": float(srt[1:].mean()) if len(srt) > 1 else float("nan"),
            "no_best2": float(srt[2:].mean()) if len(srt) > 2 else float("nan"),
            "ep_means": em.tolist(), "ep_first": first.tolist(), "day_mean": float(vals.mean())}


def ev_stats(F, t, j, net, universe: np.ndarray | None, entry: str, h: int, gross=None) -> dict:
    """사건 통계: 건수 · 주 수 · 평균 · 중앙값 · 승률 · 초과(같은 날 같은 유니버스 동일가중 대비) · NW t ·
    상위 5일 비중 · 최대 연도 비중 · 시대별 초과."""
    t = np.asarray(t)
    net = np.asarray(net, float)
    ok = np.isfinite(net)
    t, j, net = t[ok], np.asarray(j)[ok], net[ok]
    out = {"n": int(len(net))}
    if not len(net):
        return out
    wk = np.array([(np.datetime64(f"{F.dates[x][:4]}-{F.dates[x][4:6]}-{F.dates[x][6:]}") + np.timedelta64(3, "D"))
                   .astype("datetime64[W]") for x in np.unique(t)])          # 월요일 시작 주
    out["weeks"] = int(len(np.unique(wk)))
    out["mean"] = float(net.mean())
    out["median"] = float(np.median(net))
    out["win"] = float((net > 0).mean())
    lag = max(h - 1, 0)
    u, dm = daily_mean(t, net)
    out["t_nw"] = nw_t(dm, lag)
    if universe is not None:
        ew = same_day_ew(F, t, universe, entry, h)
        exc = np.array([net[i] + COST - ew.get(int(t[i]), np.nan) for i in range(len(net))])
        out["excess"] = float(np.nanmean(exc))
        u2, dx = daily_mean(t, exc)
        out["t_exc"] = nw_t(dx, lag)
        for era, (a, z) in ERAS.items():
            a_, z_ = didx(F, a), didx(F, z)
            s = (t >= a_) & (t < z_) & np.isfinite(exc)
            out[f"exc_{era}"] = float(exc[s].mean()) if s.sum() >= 10 else float("nan")
            out[f"n_{era}"] = int(s.sum())
        out["_exc"] = exc
    # 손익 집중도
    tot = net.sum()
    daysum = np.bincount(np.searchsorted(u, t), weights=net)
    top5 = np.sort(daysum)[::-1][:5].sum()
    out["top5_share"] = float(top5 / tot) if tot > 0 else float("inf")
    yrs = np.array([F.dates[x][:4] for x in t])
    ys = {y: net[yrs == y].sum() for y in np.unique(yrs)}
    out["maxyear_share"] = float(max(ys.values()) / tot) if tot > 0 else float("inf")
    out["by_year"] = {y: float(net[yrs == y].mean()) for y in np.unique(yrs)}
    return out


def boot_ci(x, n=5000, q=(0.05, 0.95), seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.quantile(b, q[0])), float(np.quantile(b, q[1]))


def pct(x, d=1):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"
