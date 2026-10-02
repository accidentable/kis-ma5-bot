"""
backtest/lab/search2.py — 2차: 조사에서 나온 후보 (WorldQuant 느린 알파, 저변동·저MAX·반전 합성, 고베타, 지수 ETF 대용)

  python -m backtest.lab.search2     → backtest/results/lab_search2.md

1차(search.py)와 같은 IS / VAL / TEST 구간·선택 규칙. 격자는 1차 결과를 본 뒤, 조사 브리프에서
나온 후보로 새로 정했다 (그래서 VAL·TEST 가 이 격자에는 한 번씩만 쓰인다).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.search import HDR, IS, OUT, TEST, VAL, fmt, market_row, row, starts, where

warnings.filterwarnings("ignore")


def _rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """행(날짜)마다 mask 안에서 0~1 백분위. mask 밖은 NaN."""
    v = np.where(mask & np.isfinite(x), x, np.nan)
    order = np.argsort(np.where(np.isnan(v), np.inf, v), axis=1)
    r = np.empty_like(order, dtype=np.float64)
    rows = np.arange(v.shape[0])[:, None]
    r[rows, order] = np.arange(v.shape[1])[None, :]
    n = np.sum(~np.isnan(v), axis=1, keepdims=True)
    out = np.where(np.isnan(v), np.nan, r / np.maximum(n - 1, 1))
    return out


def build_specs(m) -> list:
    I = m.ind
    c = m.c.astype(np.float64)
    specs = []

    def add(fam, sp):
        sp.params["fam"] = fam
        sp.prepare()
        sp.score = None
        specs.append(sp)

    ret7 = c / E._shift(c, 7) - 1
    ret3 = c / E._shift(c, 3) - 1
    ma100 = E._roll_mean(c, 100)
    slope100 = ma100 / E._shift(ma100, 100) - 1
    min100 = E._roll_min(c, 100)
    # 베타 (250일, 시장 = 거래대금 30억↑ 동일가중)
    mr = I["mkt_ret1"][:, None]
    r1 = np.nan_to_num(I["ret1"])
    cov = E._roll_mean(r1 * mr, 250) - E._roll_mean(r1, 250) * E._roll_mean(np.repeat(mr, r1.shape[1], 1), 250)
    varm = E._roll_mean(mr ** 2, 250) - E._roll_mean(mr, 250) ** 2
    beta = cov / varm
    del cov

    for u in ("LARGE", "MID", "LIQ"):
        U = E.universe(m, u)
        # WQ #19: 7일 반전 × (1 + 1년 수익률 순위)
        a19 = -np.sign(2 * (c - E._shift(c, 7))) * (1 + _rank(I["ret250"], U))
        for r in (5, 21):
            add("J.WQ19", E.Spec(f"WQ19 r{r} {u}", where(U, a19 + 1e-6 * -ret7), "rotate", r=r, params={"u": u}))
        # WQ #24: 100일선 기울기 ≤ 5% 면 100일 저점 대비 상승폭이 작은 순, 아니면 3일 반전
        a24 = np.where(slope100 <= 0.05, -(c / min100 - 1), -ret3)
        for r in (5, 21):
            add("K.WQ24", E.Spec(f"WQ24 r{r} {u}", where(U, a24), "rotate", r=r, params={"u": u}))
        # 합성: 저변동 + 저MAX + 1개월 반전 (순위 합)
        comp = _rank(-I["vol60"], U) + _rank(-I["max20"], U) + _rank(-I["ret20"], U)
        for reg in (False, True):
            add("L.저변동저MAX반전", E.Spec(f"저변동·저MAX·반전 r21{' 국면' if reg else ''} {u}", where(U, comp), "rotate",
                                         r=21, regime=reg, params={"u": u}))
        # 고베타 (기대수익 = 시장 노출 크기라는 가설)
        for reg in (False, True):
            add("M.고베타", E.Spec(f"고베타 r21{' 국면' if reg else ''} {u}", where(U & (beta > 0), beta), "rotate", r=21,
                                 regime=reg, params={"u": u}))
    return specs


def index_proxy(m, kind: str) -> np.ndarray:
    """지수 ETF 대용: 전일 시총 가중 일간 수익률의 누적 (KOSPI 상위 200 / KOSDAQ 상위 150)."""
    mk = np.array([str(x) for x in np.load(E.NPZ, allow_pickle=True)["market"]])
    kos = mk == "KOSPI" if kind == "KOSPI200" else np.isin(mk, ["KOSDAQ", "KOSDAQ GLOBAL"])
    capp = E._shift(m.cap.astype(np.float64), 1)
    r = np.nan_to_num(m.ind["ret1"])
    capp = np.where(kos[None, :] & np.isfinite(capp), capp, 0)
    n = 200 if kind == "KOSPI200" else 150
    thr = -np.sort(-capp, axis=1)[:, n - 1:n]
    w = np.where(capp >= thr, capp, 0)
    dr = (w * np.clip(r, -0.3, 0.3)).sum(1) / np.maximum(w.sum(1), 1)
    return np.cumprod(1 + dr)


def etf_row(lvl, ss, tag, name):
    # 첫날 시가 대신 전일 종가 기준 (ETF 는 괴리 작다), 비용: 수수료 양쪽 0.03% + 슬리피지 0.1%, 매도세 없음
    a = np.array([lvl[s + E.MONTH - 1] / lvl[s - 1] - 1 for s in ss]) - 0.0013
    return (f"| {name} | 1 | {tag} | {fmt(a.mean())} | {fmt(float(np.median(a)))} | {(a > 0).mean() * 100:.0f}% | "
            f"{(a >= .1).mean() * 100:.0f}% | {(a >= .2).mean() * 100:.0f}% | {(a <= -.1).mean() * 100:.0f}% | "
            f"{fmt(float(np.quantile(a, .1)))} / {fmt(float(np.quantile(a, .9)))} | - | - |")


def main() -> int:
    t0 = time.time()
    m = E.load()
    specs = build_specs(m)
    print("specs", len(specs), round(time.time() - t0), flush=True)
    S = {n: starts(m, *p) for n, p in (("IS", IS), ("VAL", VAL), ("TEST", TEST))}
    k2 = index_proxy(m, "KOSPI200")
    q150 = index_proxy(m, "KOSDAQ150")

    L = ["# 2차 — 조사 기반 후보 (전종목 16년, 한 달 대회)", "",
         "1차와 같은 규칙: IS(2011 ~ 2018)에서 계열별 선택 → VAL(2019 ~ 2024.9) → TEST(2024.10 ~) 한 번.", ""]
    res = [(sp, E.month_dist(m, sp, S["IS"], 2)) for sp in specs]
    L += ["## ① IS (2슬롯)", ""] + HDR + [market_row(m, S["IS"], "IS"),
                                          etf_row(k2, S["IS"], "IS", "KOSPI200 ETF 대용 (시총 상위 200 가중)"),
                                          etf_row(q150, S["IS"], "IS", "KOSDAQ150 ETF 대용")]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "IS", d))
    best = {}
    for sp, d in res:
        f = sp.params["fam"]
        if d["p10d"] <= 0.30 and (f not in best or d["mean"] > best[f][1]["mean"]):
            best[f] = (sp, d)
    for tag in ("VAL", "TEST"):
        L += ["", f"## ② 계열별 IS 최선 → {tag}", ""] + HDR + [
            market_row(m, S[tag], tag),
            etf_row(k2, S[tag], tag, "KOSPI200 ETF 대용 (시총 상위 200 가중)"),
            etf_row(q150, S[tag], tag, "KOSDAQ150 ETF 대용")]
        for f, (sp, _) in sorted(best.items()):
            for k in (1, 2, 3):
                L.append(row(sp, k, tag, E.month_dist(m, sp, S[tag], k)))
    p = os.path.join(OUT, "lab_search2.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p, round(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
