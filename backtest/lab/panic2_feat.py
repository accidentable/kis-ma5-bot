"""
backtest/lab/panic2_feat.py — 패닉 매수 2차 연구: 지표 · 시장 트리거 · 후보 목록 (한 번 계산해서 저장)

  python -m backtest.lab.panic2_feat          → data/lab/panic2/*.npy + meta.json
  from backtest.lab.panic2_feat import load   → F (속성으로 배열 접근, mmap)

모든 값은 그날 종가까지의 정보로만 만든다. 정규화(베타 · 변동성 · 중앙값)는 t−1 에서 끝나는 창을 쓴다.
정의는 backtest/results/lab_panic2_design.json (grid.validation_protocol + critique) 을 따른다.

종목 × 날짜 (T × N)
  ok       거래됨(거래대금 > 0) · 주가 1,000원↑ · 20일 평균 거래대금 10억↑ · 주가 30만원 이하
  cr1      전날 시총 순위.   U200 = ok & cr1 ≤ 200
  MIDQ     ok & 200 < cr1 ≤ 800 & 전날 20일 거래대금 30억↑ & 전날 회전율 ≤ 그날 후보 중앙값 & 관심주(attn20) 아님
  bstar    120일 베타(t−120..t−1, 유효 100일↑)를 0.67β+0.33 로 줄이고 [0.2, 2.5] 로 자름
  idio     clip(ret1, ±30%) − bstar · x       (x = 거래대금 30억↑ 종목 동일가중 등락률)
  sig      idio 의 60일 RMS (t−60..t−1, 유효 40일↑), 하한 0.8%
  z        idio / sig
  vr       당일 거래대금 / 전날 20일 평균 거래대금
  gap intra ibs   시가 갭, 시가→종가, 종가 위치 (h = l 이면 0.5)
  limdown  하한가 마감 (가격제한 15%: 2015-06-12 까지, 이후 30%)
  bad20    t−20..t−1 에 거래정지(거래대금 0) 또는 하한가가 있었음 (관리 · 투자경고 · 정리매매 대용)
  earn     63 · 252 거래일 전 ±5일 안에 큰 갭(|gap|/sig ≥ 3) 이나 거래대금 3배가 있었음 (실적 발표 계절성 대용)
  cnext    t 이후 처음 거래된 날의 종가 (정지 중인 종목의 '가짜 종가' 로 청산하지 않게)
날짜 (T)
  x M zm breadth3 k200 ... 시장 등락률 · 지수 · z · 하락 종목 비중 · KOSPI200 대용 등
  trig_*   시장 패닉 트리거 T0..T7 · F-a · F-c · T6(확인 진입)
후보 목록 (T × 12, −1 = 없음): top_<이름>
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

from backtest.lab import engine as E

DIR = os.path.join(E.ROOT, "data", "lab", "panic2")
SECTOR_CSV = os.path.join(E.ROOT, "data", "lab", "stock_master.csv.gz")
LABEL_NPZ = os.path.join(E.ROOT, "data", "lab", "market_label.npz")
K = 12
LIM_DATE = "20150615"


# ════════════════════════════════════════════════════════════
# 도구
# ════════════════════════════════════════════════════════════
def rsum(a: np.ndarray, n: int):
    """t 에서 끝나는 n 행 합 (NaN 은 0 으로) 과 유효 개수. t < n−1 은 부분합이므로 개수로 거른다."""
    v = np.isfinite(a)
    cs = np.cumsum(np.where(v, a, 0.0), axis=0, dtype=np.float64)
    cn = np.cumsum(v, axis=0, dtype=np.int32)
    cs[n:] = cs[n:] - cs[:-n].copy()
    cn[n:] = cn[n:] - cn[:-n].copy()
    return cs, cn


def shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan if x.dtype.kind == "f" else 0)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def row_median(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.where(mask, a, np.nan), axis=1)


def refractory(cond: np.ndarray, n: int = 5) -> np.ndarray:
    out = np.zeros_like(cond)
    last = -10 ** 9
    for t in np.where(cond)[0]:
        if t - last > n:
            out[t] = True
            last = t
    return out


def top_lists(score: np.ndarray, k: int = K) -> np.ndarray:
    """T×N 점수 (NaN = 후보 아님, 클수록 먼저) → T×k 종목 인덱스 (−1 = 없음). 동점은 인덱스 작은 쪽 먼저."""
    s = np.where(np.isfinite(score), score, -np.inf)
    out = np.full((s.shape[0], k), -1, dtype=np.int32)
    for t0 in range(0, s.shape[0], 512):
        blk = s[t0:t0 + 512]
        idx = np.argsort(-blk, axis=1, kind="stable")[:, :k]
        val = np.take_along_axis(blk, idx, axis=1)
        out[t0:t0 + 512] = np.where(np.isfinite(val), idx, -1)
    return out


def lists_of(arr: np.ndarray) -> list:
    return [r[r >= 0] for r in np.asarray(arr)]


# ════════════════════════════════════════════════════════════
# 계산
# ════════════════════════════════════════════════════════════
def build() -> dict:
    t_start = time.time()
    m = E.load()
    I = m.ind
    T, N = m.c.shape
    dates = m.dates
    out: dict = {}
    os.makedirs(DIR, exist_ok=True)

    def put(name, arr, dtype=None):
        """바로 디스크에 쓰고 (메모리 절약) 모양만 기록한다. 1차원 · 목록 배열은 메모리에도 남긴다."""
        a = np.asarray(arr)
        if dtype is not None:
            a = a.astype(dtype)
        elif a.dtype == np.float64:
            a = a.astype(np.float32)
        np.save(os.path.join(DIR, f"{name}.npy"), a)
        out[name] = a if (a.ndim == 1 or name.startswith("top_")) else (list(a.shape), str(a.dtype))
        return a

    c = m.c.astype(np.float64)
    o, h, l = (x.astype(np.float64) for x in (m.o, m.h, m.l))
    val = m.val.astype(np.float64)
    prev = shift(c, 1)
    for nm in ("o", "h", "l", "c", "raw", "val", "cap"):
        put(nm, getattr(m, nm), np.float32)
    lab = np.load(LABEL_NPZ)["label"]
    put("lab", lab, np.int8)
    ret1 = I["ret1"]
    for nm in ("ret1", "ret5", "ret20", "ret60", "ret120", "ret250", "mom12_1", "max20", "vol60", "hi250",
               "ma5", "ma20", "ma120", "atr14", "val20", "caprank", "rsi2"):
        put(nm, I[nm], np.float32)
    ret3 = c / shift(c, 3) - 1
    put("ret3", ret3)

    # ── 거래 가능 · 유니버스
    traded = np.isfinite(c) & (val > 0)
    tradable = traded & (m.raw >= 1000) & (I["val20"] >= 1e9)
    ok = tradable & (m.raw <= 300_000)
    liquid = tradable & (I["val20"] >= 3e9)
    cr1 = shift(I["caprank"].astype(np.float64), 1)
    U200 = ok & (cr1 <= 200)
    val20p = shift(I["val20"], 1)
    turn1 = shift(I["val20"] / m.cap.astype(np.float64), 1)
    vrank = (-np.nan_to_num(val, nan=-1)).argsort(axis=1).argsort(axis=1) + 1
    attn20 = shift((E._roll_max((vrank <= 50).astype(np.float64), 20) > 0).astype(np.float64), 1) > 0
    del vrank
    midc = ok & (cr1 > 200) & (cr1 <= 800) & (val20p >= 3e9)
    MIDQ = midc & (turn1 <= row_median(turn1, midc)[:, None]) & ~attn20
    for nm, a in (("traded", traded), ("tradable", tradable), ("ok", ok), ("liquid", liquid), ("U200", U200),
                  ("MIDQ", MIDQ), ("attn20", attn20)):
        put(nm, a, bool)
    put("cr1", cr1)
    put("turn1", turn1)

    # ── 시장
    x = I["mkt_ret1"].astype(np.float64)
    M = I["mkt"].astype(np.float64)
    put("x", x, np.float64)
    put("M", M, np.float64)
    put("mkt_on", I["mkt_on"], bool)
    s2, n2 = rsum(x ** 2, 60)
    sig_m = shift(np.sqrt(s2 / np.maximum(n2, 1)), 1)
    sig_m[:61] = np.nan
    zm = x / sig_m
    put("sig_m", sig_m, np.float64)
    put("zm", zm, np.float64)
    r = np.clip(ret1, -0.3, 0.3)
    br = np.where(liquid, (ret1 <= -0.03).astype(float), np.nan)
    breadth3 = np.nanmean(br, axis=1)
    put("breadth3", breadth3, np.float64)
    del br
    ibs = np.where(h > l, (c - l) / np.where(h > l, h - l, 1), 0.5)
    ibs[~traded] = np.nan
    put("ibs", ibs)
    ew_ibs = np.nanmean(np.where(liquid & (h > l), ibs, np.nan), axis=1)
    put("ew_ibs", ew_ibs, np.float64)
    gap = o / prev - 1
    gap[~traded] = np.nan
    put("gap", gap)
    liquid_p = shift(liquid.astype(float), 1) > 0          # 시가에 아는 정보: 전날 기준 유동 종목
    ew_gap = np.nanmean(np.where(liquid_p & np.isfinite(gap), np.clip(gap, -0.3, 0.3), np.nan), axis=1)
    put("ew_gap", ew_gap, np.float64)
    intra = c / o - 1
    intra[~traded] = np.nan
    put("intra", intra)
    mmax60 = E._roll_max(M[:, None], 60)[:, 0]
    mkt_dd60 = M / mmax60 - 1
    put("mkt_dd60", mkt_dd60, np.float64)
    dd20m = M / E._roll_max(M[:, None], 20)[:, 0] - 1
    put("dd20m", dd20m, np.float64)
    dd250m = M / E._roll_max(M[:, None], 250)[:, 0] - 1
    put("dd250m", dd250m, np.float64)
    put("above200", M > E._roll_mean(M[:, None], 200)[:, 0], bool)
    mkt_vol20 = np.sqrt(E._roll_mean(x[:, None] ** 2, 20))[:, 0]
    put("mkt_vol20", mkt_vol20, np.float64)
    s5, _ = rsum(x ** 2, 5)
    s250, n250 = rsum(x ** 2, 250)
    rms250p = shift(np.sqrt(s250 / np.maximum(n250, 1)), 1)
    rms250p[:251] = np.nan
    S5 = np.sqrt(s5 / 5) / rms250p
    put("S5", S5, np.float64)
    lim = np.where(dates < LIM_DATE, 0.15, 0.30)
    put("lim", lim, np.float64)
    limdown = traded & (c <= prev * (1 - lim[:, None] + 0.005))
    limup = traded & (c >= prev * (1 + lim[:, None] - 0.005))
    put("limdown", limdown, bool)
    put("limup", limup, bool)
    halt = np.isfinite(c) & (val <= 0)
    flag = (halt | limdown).astype(np.float64)
    bad20 = shift(E._roll_max(flag, 20), 1) > 0
    put("bad20", bad20, bool)
    del flag, halt

    # KOSPI200 · KOSDAQ150 대용 (전날 시총 가중, 전날 시장 구분)
    capp = np.nan_to_num(shift(m.cap.astype(np.float64), 1))
    labp = shift(lab.astype(np.float64), 1)
    r0 = np.nan_to_num(r)
    g0 = np.nan_to_num(np.clip(gap, -0.3, 0.3))
    for key, code, n in (("k200", 1, 200), ("kq150", 2, 150)):
        w = np.where(labp == code, capp, 0.0)
        thr = -np.sort(-w, axis=1)[:, n - 1:n]
        w = np.where((w >= thr) & (w > 0), w, 0.0)
        ws = np.maximum(w.sum(1), 1)
        rr = (w * r0).sum(1) / ws
        gg = (w * g0).sum(1) / ws
        nav = np.cumprod(1 + rr)
        navo = np.concatenate([[nav[0]], nav[:-1]]) * (1 + gg)
        put(f"{key}_ret1", rr, np.float64)
        put(f"{key}_nav", nav, np.float64)
        put(f"{key}_open", navo, np.float64)
    del capp, labp, w
    # KOSDAQ 안 시총 순위 (전날)
    capk = np.where(lab == 2, np.nan_to_num(m.cap.astype(np.float64), nan=-1), -1)
    kqrank = (-capk).argsort(axis=1).argsort(axis=1).astype(np.float64) + 1
    kqrank[capk <= 0] = np.nan
    kqrank1 = shift(kqrank, 1)
    del capk, kqrank
    capk = np.where(lab == 1, np.nan_to_num(m.cap.astype(np.float64), nan=-1), -1)
    kprank = (-capk).argsort(axis=1).argsort(axis=1).astype(np.float64) + 1
    kprank[capk <= 0] = np.nan
    kprank1 = shift(kprank, 1)
    del capk, kprank
    put("kqrank1", kqrank1)
    put("kprank1", kprank1)
    print(f"  시장 지표 {time.time() - t_start:.0f}s", flush=True)

    # ── 베타 · 잔차
    v = np.isfinite(r)
    X2 = np.where(v, x[:, None], np.nan)
    Sx, n = rsum(X2, 120)
    Sy, _ = rsum(r, 120)
    Sxx, _ = rsum(X2 * X2, 120)
    Sxy, _ = rsum(X2 * r, 120)
    den = n * Sxx - Sx * Sx
    beta = np.where((n >= 100) & (den > 0), (n * Sxy - Sx * Sy) / np.where(den > 0, den, 1), np.nan)
    del Sx, Sy, Sxx, Sxy, den, X2, n
    beta = shift(beta, 1)
    bstar = np.clip(0.67 * beta + 0.33, 0.2, 2.5)
    del beta
    idio = r - bstar * x[:, None]
    s2, k2 = rsum(idio * idio, 60)
    sig = np.where(k2 >= 40, np.sqrt(s2 / np.maximum(k2, 1)), np.nan)
    del s2, k2
    sig = np.maximum(shift(sig, 1), 0.008)
    z = idio / sig
    put("bstar", bstar)
    put("idio", idio)
    put("sig", sig)
    put("z", z)
    vr = val / val20p
    put("vr", vr)
    print(f"  베타·잔차 {time.time() - t_start:.0f}s", flush=True)

    # ── 실적 계절성 대용 · 배당락 · 옵션 만기
    fl = ((np.abs(np.nan_to_num(gap)) / sig >= 3) | (np.nan_to_num(vr) >= 3)).astype(np.float64)
    rm = E._roll_max(fl, 11)
    earn = (shift(rm, 247) > 0) | (shift(rm, 58) > 0)
    put("earn", earn, bool)
    del fl, rm
    yrs = np.array([d[:4] for d in dates])
    divx = np.zeros(T, bool)
    for y in sorted(set(yrs)):
        if int(y) > 2022:
            continue
        idx = np.where(yrs == y)[0]
        if dates[idx[-1]][4:6] == "12" and int(y) < int(dates[-1][:4]):
            divx[idx[-2:]] = True
    put("divx", divx, bool)
    # 옵션 만기일: 매월 둘째 목요일 (휴장이면 그 전 거래일)
    import datetime as _dt
    optx = np.zeros(T, bool)
    for y in range(int(dates[0][:4]), int(dates[-1][:4]) + 1):
        for mo in range(1, 13):
            d1 = _dt.date(y, mo, 1)
            th = [d1 + _dt.timedelta(days=i) for i in range(14) if (d1 + _dt.timedelta(days=i)).weekday() == 3][1]
            ds = th.strftime("%Y%m%d")
            i = int(np.searchsorted(dates, ds, side="right")) - 1
            if 0 <= i < T and dates[i][:6] == ds[:6]:
                optx[i] = True
    put("optx", optx, bool)

    # ── 연속 하락 · 스트릭
    down = np.nan_to_num(ret1) < 0
    nd = np.zeros((T, N), np.int16)
    worst = np.full((T, N), np.nan, np.float32)
    for t in range(1, T):
        nd[t] = np.where(down[t], nd[t - 1] + 1, 0)
        worst[t] = np.where(down[t], np.fmin(np.where(nd[t - 1] > 0, worst[t - 1], np.inf), ret1[t]), np.nan)
    put("down_days", nd, np.int16)
    put("worst_streak", worst, np.float32)
    wst = worst.astype(np.float64)
    ti = np.arange(T)[:, None] - nd.astype(np.int64)
    ti = np.clip(ti, 0, T - 1)
    cum = c / np.take_along_axis(c, ti, axis=0) - 1
    mcum = M[:, None] / M[ti] - 1
    icum = np.where(nd > 0, cum - bstar * mcum, np.nan)
    put("icum", icum)
    del ti, cum, mcum, down

    # ── 다음 거래 종가 (정지 건너뛰기)
    cn_ = np.where(traded, c, np.nan)
    nxt = np.full((T, N), np.nan)
    last = np.full(N, np.nan)
    for t in range(T - 1, -1, -1):
        last = np.where(np.isfinite(cn_[t]), cn_[t], last)
        nxt[t] = last
    put("cnext", nxt)
    # 마지막 거래 종가 (그날 이전 · 포함) — 다시 거래되지 않는 (상장폐지) 종목의 청산 가격
    lst = np.full((T, N), np.nan)
    last = np.full(N, np.nan)
    for t in range(T):
        last = np.where(np.isfinite(cn_[t]), cn_[t], last)
        lst[t] = last
    put("clast", lst)
    del cn_, nxt, lst

    # ── 업종 (2018 스냅샷, 없는 종목은 '미분류' 한 묶음)
    import pandas as pd
    d = pd.read_csv(SECTOR_CSV, dtype=str)
    labd = dict(zip(d["Symbol"].str.zfill(6), d["Sector"]))
    secs = sorted({s for s in labd.values() if isinstance(s, str)})
    sid = {s: i for i, s in enumerate(secs)}
    col = np.array([sid.get(labd.get(cd), -1) for cd in m.codes], dtype=np.int32)
    S = len(secs)
    colx = np.where(col >= 0, col, S)                  # 미분류 = S
    put("sector", col, np.int32)
    # 같은 업종 다른 종목 평균 (secx) · 3%↓ 비중 (sbreadth): 가격 유효 · 1,000원↑ · 전날 20일 거래대금 10억↑
    peer = np.isfinite(c) & (m.raw >= 1000) & (val20p >= 1e9)
    rp = np.where(peer, np.nan_to_num(r), 0.0)
    dn3 = np.where(peer, (np.nan_to_num(ret1) <= -0.03).astype(float), 0.0)
    oh = np.zeros((N, S + 1))
    oh[np.arange(N), colx] = 1.0
    sum_r = rp @ oh
    sum_n = peer.astype(float) @ oh
    sum_d = dn3 @ oh
    own_r, own_n, own_d = rp, peer.astype(float), dn3
    pr = sum_r[:, colx] - own_r
    pn = sum_n[:, colx] - own_n
    pd_ = sum_d[:, colx] - own_d
    secx = np.where(pn >= 4, pr / np.maximum(pn, 1), np.nan)            # 미분류(2018 뒤 상장 등)는 한 묶음 (검토 의견 2)
    sbreadth = np.where(pn >= 4, pd_ / np.maximum(pn, 1), np.nan)
    put("secx", secx)
    put("sbreadth", sbreadth)
    # S-SECREL 용 업종 평균 (ok & val20 ≥ 30억, 5개↑; 미분류도 한 묶음)
    mem = ok & (I["val20"] >= 3e9)
    sr_sum = np.where(mem, np.nan_to_num(r), 0.0) @ oh
    sr_n = mem.astype(float) @ oh
    SR = np.where(sr_n >= 5, sr_sum / np.maximum(sr_n, 1), np.nan)
    SRi = SR[:, colx]
    put("SRi", SRi)
    del rp, dn3, pr, pn, pd_, sum_r, sum_n, sum_d, own_r, own_n, own_d, sr_sum, sr_n, SR, oh, peer, mem
    print(f"  업종 {time.time() - t_start:.0f}s", flush=True)

    # ── 시장 트리거 (날짜 bool)
    T0 = x <= -0.04
    trig = {
        "T0": T0,
        "T0_3": x <= -0.03,
        "T0_5": x <= -0.05,
        "T1": (zm <= -4) & (x <= -0.025),
        "T2": breadth3 >= 0.75,
        "T3": out["k200_ret1"] <= -0.03,
        "T4": refractory((M / shift(M, 3) - 1 <= -0.08) & (x < 0), 5),
        "T5": refractory((S5 >= 2.5) & (mkt_dd60 <= -0.10), 5),
        "T7": (x <= -0.03) & (ew_ibs >= 0.5) & (dd20m <= -0.08),
    }
    t0cnt = np.concatenate([[0], np.cumsum(T0)])
    n40 = np.array([t0cnt[t] - t0cnt[max(0, t - 40)] for t in range(T)])          # t−40..t−1
    deepcap = (n40 >= 2) & (dd250m <= -0.25)
    mkt_on1 = shift(I["mkt_on"].astype(float), 1) > 0
    trig["Fa"] = T0 & (mkt_on1 | deepcap)
    trig["Fc"] = T0 & (shift(mkt_dd60, 1) >= -0.08)
    # T6 확인 진입: T0 일 t0 뒤 10일 안에 x ≥ +1% 인 첫날 u*. 그 전에 새 T0 가 오면 t0 를 갱신
    conf = np.zeros(T, bool)
    anchor = np.full(T, -1, np.int32)
    t0 = -1
    for t in range(T):
        if T0[t]:
            t0 = t
            continue
        if t0 >= 0 and t - t0 > 10:
            t0 = -1
        if t0 >= 0 and x[t] >= 0.01:
            conf[t] = True
            anchor[t] = t0
            t0 = -1
    trig["T6"] = conf
    for kname, a in trig.items():
        put(f"trig_{kname}", np.asarray(a, bool), bool)
    put("t6_anchor", anchor, np.int32)
    put("n40", n40, np.int32)
    put("deepcap", deepcap, bool)

    # ── 평소 전략 (현재 봇 R0) 순위
    okb = (np.isfinite(c)) & (m.raw >= 1000) & (val > 0) & (I["val20"] >= 1e9) & (m.raw <= 300_000)
    base_mask = okb & (I["caprank"] <= 200) & (I["ret60"] > 0) & (I["max20"] < 0.10) & np.isfinite(I["hi250"])
    base_score = np.where(base_mask, c / I["hi250"], np.nan)
    base = E.Spec("base", base_score, "rotate", r=21).prepare()
    bt = np.full((T, K), -1, np.int32)
    for t in range(T):
        a = base.top[t][:K]
        bt[t, :len(a)] = a
    put("top_base", bt, np.int32)
    del base_score, base, base_mask

    # ── 시장 패닉 후보 (그날 종가 기준 순위, 클수록 먼저)
    med_u = row_median(np.where(U200, ret1, np.nan), U200)[:, None]
    dec = U200 & (ret1 <= med_u)
    rrv = vr / row_median(vr, U200)[:, None]
    cut = np.array([np.nanquantile(np.where(dec[t], rrv[t], np.nan), 2 / 3) if dec[t].sum() >= 3 else np.nan
                    for t in range(T)])
    pool = shift(((I["caprank"] <= 200) & (c / I["hi250"] >= 0.90) & (I["ret60"] > 0)
                  & (I["max20"] < 0.10)).astype(float), 5) > 0
    ep_dd = c / E._roll_max(c, 11) - 1
    sel = {
        "K2": np.where(U200, -ret1, np.nan),
        "SRES": np.where(U200 & (idio >= -0.20), -z, np.nan),
        "SLEAD": np.where(U200 & pool, -ret3, np.nan),
        "SLOSER": np.where(U200, -shift(I["ret120"], 1), np.nan),
        "SSECREL": np.where(U200, -(r - SRi), np.nan),
        "SHVOL": np.where(dec & (rrv >= cut[:, None]), -ret1, np.nan),
        "SIBS": np.where(U200 & (ibs >= 0.3), -ret1, np.nan),
        "KQL": np.where(ok & (lab == 2) & (kqrank1 <= 100), -ret1, np.nan),
        "KPL": np.where(U200 & (lab == 1), -ret1, np.nan),                 # 설계: KOSPI & 시총 200
        "SBETA": np.where(dec, bstar, np.nan),
        "SMEGA": np.where(ok & (cr1 <= 30), -ret1, np.nan),
        "K2EP": np.where(U200, -ep_dd, np.nan),
        "LVOL": np.where(U200, -shift(I["vol60"], 1), np.nan),
        "HMOM": np.where(U200, shift(I["mom12_1"], 1), np.nan),
        "K1": np.where(ok & attn20 & (ret1 <= -0.10), -ret1, np.nan),
        "SRES3": None,
    }
    # 3일 잔차 z (w=3)
    id3, k3 = rsum(idio, 3)
    sel["SRES3"] = np.where(U200 & (idio >= -0.20) & (k3 == 3), -(id3 / (sig * np.sqrt(3))), np.nan)
    del id3, k3, rrv, dec, pool, ep_dd
    for nm, sc in sel.items():
        put(f"top_{nm}", top_lists(sc), np.int32)
    put("sc_SRES", sel["SRES"])
    put("sc_K2", sel["K2"])
    del sel
    print(f"  시장 패닉 후보 {time.time() - t_start:.0f}s", flush=True)

    # T6 목록: t0 의 상위 12 (K2 / S-RES) 를 고정해 두고, u* 에 아직 ok 인 종목을 c[u*]/c[t0−1]−1 오름차순
    for nm, key in (("T6K2", "top_K2"), ("T6RES", "top_SRES")):
        arr = np.full((T, K), -1, np.int32)
        for u in np.where(conf)[0]:
            a0 = anchor[u]
            L = [int(j) for j in out[key][a0] if j >= 0 and ok[u, j] and np.isfinite(c[a0 - 1, j])]
            L.sort(key=lambda j: c[u, j] / c[a0 - 1, j] - 1)
            arr[u, :len(L)] = L[:K]
        put(f"top_{nm}", arr, np.int32)

    # ── 종목 패닉 (트랙 A)
    t0_recent = np.zeros(T, bool)
    for t in range(T):
        t0_recent[t] = T0[max(0, t - 5):t + 1].any()
    base_cf = ok & (x > -0.02)[:, None] & ~t0_recent[:, None] & (ret1 > -0.20) & ~limdown & ~divx[:, None]
    CF = base_cf & ~bad20
    put("CF_nostatus", base_cf, bool)
    put("CF", CF, bool)
    medturn = row_median(turn1, U200)[:, None]
    medvol = row_median(shift(I["vol60"], 1), U200)[:, None]
    newsgap = (gap <= -0.03) & (intra <= 0.005)
    SA1 = CF & U200 & (z <= -3.5) & (ret1 <= -0.04)
    SA2 = CF & U200 & (z <= -3) & (ret1 <= -0.04) & (vr <= 2) & (turn1 <= medturn) & (shift(I["vol60"], 1) <= medvol)
    hi1 = shift(c / I["hi250"], 1)
    SA3 = (SA1 & ~(ret1 <= -0.15) & ~(shift(ret1, 1) <= -0.10)
           & ~((shift(I["ret60"], 1) >= 0.30) | (shift(I["max20"], 1) >= 0.15)) & ~(hi1 <= 0.5) & ~earn & ~newsgap)
    dd = (c - I["ma20"]) / shift(I["atr14"], 1)
    pre = ((cr1 <= 200) & (hi1 >= 0.85) & (shift(I["ret60"], 1) > 0) & (shift(I["ret60"], 1) <= 0.40)
           & (shift(I["max20"], 1) < 0.10) & (shift(c - I["ma120"], 1) > 0))
    SA4 = CF & pre & (dd <= -2.5) & (z <= -2) & (ret1 > -0.15)
    SA5 = CF & MIDQ & (z <= -3.5) & (ret1 <= -0.05) & (vr <= 2)
    SA6 = CF & U200 & (nd >= 5) & (icum <= -0.06) & (wst >= -0.04) & ((M / shift(M, 5) - 1) > -0.04)[:, None]
    SA7 = CF & U200 & (ret1 <= -0.05) & (secx <= -0.03) & (sbreadth >= 0.5) & ((ret1 - secx) > -0.05)
    # SA8 시가 갭 되돌림: 시가에 아는 정보만 (전날 유니버스, 오늘 갭, 오늘 시장 갭)
    U200p = shift(U200.astype(float), 1) > 0
    gz = (gap - bstar * ew_gap[:, None]) / sig
    rawp = shift(m.raw.astype(np.float64), 1)
    gapbase = (U200p & traded & (rawp <= 300_000) & ~bad20 & ~divx[:, None]
               & (ew_gap > -0.01)[:, None] & (gz <= -3))
    SA8 = gapbase & (gap <= -0.04)
    SA8b = gapbase & (gap <= -0.07)
    put("gz", gz)
    put("dd_atr", dd)
    # 확인 진입: 전날 사건 & 오늘 저가·종가가 전날보다 높음 & 오늘 시장 평온
    def confirm(mask):
        pm = shift(mask.astype(float), 1) > 0
        return pm & ok & (l >= shift(l, 1)) & (c >= prev) & (x > -0.02)[:, None]
    A = {
        "SA1": (SA1, -z), "SA2": (SA2, -z), "SA3": (SA3, -z), "SA4": (SA4, -dd), "SA5": (SA5, -z),
        "SA6": (SA6, -icum), "SA7": (SA7, -ret1), "SA8": (SA8, -gz), "SA8b": (SA8b, -gz),
        "SA1C1": (confirm(SA1), shift(-z, 1)), "SA3C1": (confirm(SA3), shift(-z, 1)),
        "SA1ns": (base_cf & U200 & (z <= -3.5) & (ret1 <= -0.04), -z),       # 상태 대용 필터 없음 (효과 보고용)
    }
    for nm, (msk, sc) in A.items():
        s_ = np.where(msk, sc, np.nan)
        put(f"top_{nm}", top_lists(s_), np.int32)
        put(f"m_{nm}", msk, bool)
    print(f"  종목 패닉 {time.time() - t_start:.0f}s", flush=True)
    out["_meta"] = {"dates": [str(d) for d in dates], "codes": [str(x) for x in m.codes],
                    "names": [str(x) for x in m.names], "sectors": secs, "K": K}
    return out


def save(out: dict):
    meta = out.pop("_meta")
    meta["arrays"] = {k: ([list(a.shape), str(a.dtype)] if isinstance(a, np.ndarray) else list(a)) for k, a in out.items()}
    with open(os.path.join(DIR, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)


class _F(SimpleNamespace):
    def __getattr__(self, k):              # 처음 쓰는 배열만 mmap 으로 연다
        p = os.path.join(DIR, f"{k}.npy")
        if not os.path.exists(p):
            raise AttributeError(k)
        a = np.load(p, mmap_mode="r")
        a = np.asarray(a)
        setattr(self, k, a)
        return a


def load() -> _F:
    with open(os.path.join(DIR, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    F = _F()
    F.dates = np.array(meta["dates"])
    F.codes = np.array(meta["codes"])
    F.names = np.array(meta["names"])
    F.sectors = meta["sectors"]
    F.arrays = meta["arrays"]
    F.T, F.N = len(F.dates), len(F.codes)
    return F


def didx(F, d: str) -> int:
    return int(np.searchsorted(F.dates, d))


def market_of(F, extra_index: str | None = None):
    """시뮬레이터용 Market 객체 (panic_sim2.run_month 가 쓰는 필드만). extra_index='k200' 이면 합성 지수 열을 붙인다."""
    ind = {"mkt": F.M, "ret1": F.ret1, "ma5": F.ma5, "cnext": F.cnext}
    arrs = {k: getattr(F, k) for k in ("o", "h", "l", "c", "raw", "val", "cap")}
    if extra_index:
        nav, navo = getattr(F, f"{extra_index}_nav"), getattr(F, f"{extra_index}_open")
        scale = 35_000 / nav[0]
        col = {"o": navo * scale, "c": nav * scale, "raw": nav * scale, "val": np.full(F.T, 1e12), "cap": np.full(F.T, np.nan)}
        col["h"] = np.maximum(col["o"], col["c"])
        col["l"] = np.minimum(col["o"], col["c"])
        arrs = {k: np.hstack([a, col[k][:, None].astype(np.float32)]) for k, a in arrs.items()}
        ind = {"mkt": F.M, "ret1": np.hstack([F.ret1, np.zeros((F.T, 1), np.float32)]),
               "ma5": np.hstack([F.ma5, np.full((F.T, 1), np.inf, np.float32)]),
               "cnext": np.hstack([F.cnext, col["c"][:, None].astype(np.float32)])}
    return E.Market(F.dates, F.codes, F.names, arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["raw"], arrs["val"],
                    arrs["cap"], ind)


def main() -> int:
    t = time.time()
    out = build()
    save(out)
    print(f"저장: {DIR} ({len(out)}개, {time.time() - t:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
