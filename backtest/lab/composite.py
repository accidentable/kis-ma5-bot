"""
backtest/lab/composite.py — 여러 기술적 지표를 한 점수로 묶은 종목 선정 모델, 한 달 대회 기준 검증

  python -m backtest.lab.composite    → backtest/results/lab_composite.md

1. features.py 의 지표 42개를 그날 유니버스 안 백분위(0 ~ 1)로 바꾼다.
2. 가중치는 IS(2011 ~ 2018) 표본일에서만 추정한다.
     IC가중   w = 지표별 IS 평균 IC
     t필터    |IS t| ≥ 3 인 지표만, 부호만 (±1)
     릿지     IS 표본일들을 모아 '다음 달 수익률 백분위' 를 지표 백분위로 회귀 (λ = 표본 수의 1%)
3. 점수 상위 k 종목을 r 거래일마다 교체하는 전략을 60만원·한 달 시뮬레이터로 돌린다.
   격자 (사전 고정 18개): 방법 3 × 유니버스 {시총 상위 200, 시총 상위 500, 거래대금 30억↑ 전체} × 교체 {10, 21}일.
   전부 주가 30만원 이하 (2슬롯 60만원 계좌에서 살 수 있는 종목).
4. IS 에서 1위 → VAL → TEST 한 번.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.features import build, forward, spearman_rows
from backtest.lab.search import HDR, OUT, fmt, market_row, row

warnings.filterwarnings("ignore")

IS = ("20110101", "20181231")
VAL = ("20190101", "20240930")
TEST = ("20241001", "99999999")


def pct_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """행마다 mask 안에서 0 ~ 1 백분위 (float32). 밖은 NaN."""
    v = np.where(mask & np.isfinite(x), x, np.nan).astype(np.float64)
    order = np.argsort(np.where(np.isnan(v), np.inf, v), axis=1)
    r = np.empty(v.shape, dtype=np.float32)
    rows = np.arange(v.shape[0])[:, None]
    r[rows, order] = np.arange(v.shape[1], dtype=np.float32)[None, :]
    n = np.sum(~np.isnan(v), axis=1, keepdims=True).astype(np.float32)
    return np.where(np.isnan(v), np.nan, r / np.maximum(n - 1, 1)).astype(np.float32)


def universes(m) -> dict:
    I = m.ind
    base = ((~np.isnan(m.c)) & (m.raw >= 1000) & (m.raw <= 300_000) & (I["val20"] >= 1e9) & (m.val > 0)
            & np.isfinite(I["hi250"]))
    return {"LARGE200": base & (I["caprank"] <= 200),
            "LARGE500": base & (I["caprank"] <= 500),
            "LIQ": base & (I["val20"] >= 3e9)}


def sample_days(m, a, z, step=21):
    s0, s1 = m.didx(a), min(m.didx(z), m.T - 22)
    return list(range(max(s0, 260), s1, step))


def fit_weights(R: dict, y: np.ndarray, mask: np.ndarray, days: list) -> dict:
    names = list(R)
    ic = {k: np.array([spearman_rows(R[k][t], y[t], mask[t]) for t in days]) for k in names}
    mean = {k: float(np.nanmean(v)) for k, v in ic.items()}
    tval = {k: float(np.nanmean(v) / (np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()))) for k, v in ic.items()}
    w_ic = mean
    w_t = {k: (np.sign(mean[k]) if abs(tval[k]) >= 3 else 0.0) for k in names}
    # 릿지: 표본일 전부 쌓기
    X, Y = [], []
    for t in days:
        ok = mask[t] & np.isfinite(y[t])
        if ok.sum() < 30:
            continue
        yr = y[t][ok].argsort().argsort() / max(ok.sum() - 1, 1) - 0.5
        cols = [np.nan_to_num(R[k][t][ok] - 0.5) for k in names]
        X.append(np.column_stack(cols))
        Y.append(yr)
    X, Y = np.vstack(X), np.concatenate(Y)
    lam = 0.01 * len(Y)
    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    w_r = dict(zip(names, beta.tolist()))
    return {"IC가중": w_ic, "t필터": w_t, "릿지": w_r, "_t": tval, "_ic": mean}


def score(R: dict, w: dict) -> np.ndarray:
    s = None
    for k, wk in w.items():
        if not wk:
            continue
        part = np.nan_to_num(R[k] - 0.5) * wk
        s = part if s is None else s + part
    return s


def main() -> int:
    t0 = time.time()
    m = E.load()
    F = build(m)
    y = forward(m, 21)
    US = universes(m)
    print("features", len(F), round(time.time() - t0), flush=True)

    def st(p, step):
        s0, s1 = m.didx(p[0]), min(m.didx(p[1]), m.T - E.MONTH)
        return list(range(max(s0, 260), s1, step))

    S = {"IS": st(IS, 3), "VAL": st(VAL, 2), "TEST": st(TEST, 1)}
    L = ["# 기술적 지표 합성 모델 — 한 달 대회 기준", "", (__doc__ or "").strip().split("\n\n", 1)[1], ""]
    specs, weights = [], {}
    for un, U in US.items():
        R = {k: pct_rank(x, U) for k, x in F.items()}
        W = fit_weights(R, y, U, sample_days(m, *IS))
        weights[un] = W
        for meth in ("IC가중", "t필터", "릿지"):
            sc = score(R, W[meth])
            sc = np.where(U, sc, np.nan)
            # 합성 점수의 기간별 IC (참고)
            for r in (10, 21):
                sp = E.Spec(f"합성({meth}) {un} r{r}", sc, "rotate", r=r, params={"u": un, "meth": meth})
                sp.prepare()
                specs.append((sp, sc if r == 21 else None))
        del R
        print(un, round(time.time() - t0), flush=True)

    # 합성 점수 IC
    L += ["## 합성 점수의 예측력 (IC, 한 달)", "", "| 모델 | IS | VAL | TEST |", "|---|---|---|---|"]
    for sp, sc in specs:
        if sc is None:
            continue
        U = US[sp.params["u"]]
        cells = []
        for p in (IS, VAL, TEST):
            ic = np.array([spearman_rows(sc[t], y[t], U[t]) for t in sample_days(m, *p)])
            cells.append(f"{np.nanmean(ic):+.3f}")
        L.append(f"| {sp.name[:-4]} | " + " | ".join(cells) + " |")
    for sp, _ in specs:
        sp.score = None

    res = [(sp, E.month_dist(m, sp, S["IS"], 2)) for sp, _ in specs]
    L += ["", "## ① IS (2슬롯)", ""] + HDR + [market_row(m, S["IS"], "IS")]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "IS", d))
    best = max(res, key=lambda x: x[1]["mean"])[0]
    L += ["", f"## ② IS 1위 → VAL · TEST: {best.name}", ""] + HDR
    for tag in ("VAL", "TEST"):
        L.append(market_row(m, S[tag], tag))
        for k in (1, 2, 3, 5):
            L.append(row(best, k, tag, E.month_dist(m, best, S[tag], k)))
    L += ["", "## ③ 참고: 전 설정 2슬롯 VAL · TEST (선택에 안 씀)", "", "| 설정 | IS | VAL | TEST |", "|---|---|---|---|"]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        v, t = E.month_dist(m, sp, S["VAL"], 2), E.month_dist(m, sp, S["TEST"], 2)
        L.append(f"| {sp.name} | {fmt(d['mean'])} | {fmt(v['mean'])} (−10%↓ {v['p10d'] * 100:.0f}%) | "
                 f"{fmt(t['mean'])} (−10%↓ {t['p10d'] * 100:.0f}%) |")
    # 가중치 기록
    un = best.params["u"]
    W = weights[un]
    L += ["", f"## ④ {un} 가중치 (IS 추정)", "", "| 지표 | IS IC | IS t | 릿지 계수 | t필터 |", "|---|---|---|---|---|"]
    for k in sorted(W["_ic"], key=lambda k: -abs(W["_t"][k])):
        L.append(f"| {k} | {W['_ic'][k]:+.3f} | {W['_t'][k]:+.1f} | {W['릿지'][k]:+.4f} | {W['t필터'][k]:+.0f} |")
    # 연도별
    allst = list(range(max(m.didx("20110101"), 260), m.T - E.MONTH, 2))
    d = E.month_dist(m, best, allst, 2)
    mk = m.ind["mkt"]
    L += ["", f"## ⑤ 연도별 — {best.name} (2슬롯)", "", "| 연도 | 전략 | 시장(비용 전) | +10%↑ | −10%↓ |", "|---|---|---|---|---|"]
    for yv in sorted({m.dates[s][:4] for s in allst}):
        idx = [n for n, s in enumerate(allst) if m.dates[s][:4] == yv]
        a = d["rets"][idx]
        mm = np.array([mk[allst[n] + 20] / mk[allst[n] - 1] - 1 for n in idx])
        L.append(f"| {yv} | {fmt(a.mean())} | {fmt(mm.mean())} | {(a >= .1).mean() * 100:.0f}% | {(a <= -.1).mean() * 100:.0f}% |")
    import json
    with open(os.path.join(OUT, "lab_composite_weights.json"), "w", encoding="utf-8") as fh:
        json.dump({u: {k: v for k, v in w.items()} for u, w in weights.items()}, fh, ensure_ascii=False, indent=1)
    p = os.path.join(OUT, "lab_composite.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p, round(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
