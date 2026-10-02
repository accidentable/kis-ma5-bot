"""
backtest/lab/combo.py — 52주 신고가 근접(현재 봇 기본) × 기술적 지표 합성 점수(나쁜 종목 거르기) 조합

  python -m backtest.lab.combo    → backtest/results/lab_combo.md

composite.py 결과: 합성 점수의 순위 예측력(IC ≈ +0.15)은 강하지만 대부분 '하위 종목을 골라내는' 힘이라,
상위 2종목만 사는 전략으로는 시장 수준에 그쳤다. 그래서 합성 점수를 필터로 쓴다.
합성 가중치는 IS(2011 ~ 2018) 릿지 추정치 (거래대금 30억↑ 유니버스). 이 파일의 변형 4개는 composite 결과를 본 뒤 정했다.
  V1 기본: 시총 200, 30만원 이하, 60일 수익률 > 0, 20일 내 +10% 급등일 없음, 종가/250일 최고가 순, 2종목 21일
  V2 V1 + 합성 점수 상위 50% (거래대금 30억↑ 전체 안에서)
  V3 V1 + 합성 점수 상위 30%
  V4 V1 유니버스에서 순위 = 신고가 근접 백분위 + 합성 점수 백분위
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.composite import IS, fit_weights, pct_rank, sample_days, score, universes
from backtest.lab.features import build, forward
from backtest.lab.search import OUT, fmt, starts

warnings.filterwarnings("ignore")


def main() -> int:
    m = E.load()
    I = m.ind
    F = build(m)
    y = forward(m, 21)
    U = universes(m)["LIQ"]
    R = {k: pct_rank(x, U) for k, x in F.items()}
    W = fit_weights(R, y, U, sample_days(m, *IS))
    comp = np.where(U, score(R, W["릿지"]), np.nan)
    del R, F
    cpct = pct_rank(comp, U)
    base = ((~np.isnan(m.c)) & (m.raw >= 1000) & (m.raw <= 300_000) & (I["val20"] >= 1e9) & (m.val > 0)
            & (I["caprank"] <= 200) & (I["ret60"] > 0) & (I["max20"] < 0.10))
    nh = m.c / I["hi250"]
    P = {"2011~2018": starts(m, "20110101", "20181231", 2), "2019~2024.9": starts(m, "20190101", "20240930", 2),
         "2024.10~": starts(m, "20241001", "99999999", 1), "최근 12개월": starts(m, "20250901", "99999999", 1)}
    mk = I["mkt"]
    L = ["# 52주 신고가 근접 × 기술적 지표 합성 점수 조합", "", (__doc__ or "").split("\n\n", 1)[1].strip(), "",
         "| 변형 | " + " | ".join(P) + " |", "|---|" + "---|" * len(P),
         "| 시장 (거래대금 30억↑ 동일가중, 비용 전) | " + " | ".join(
             fmt(np.mean([mk[s + 20] / mk[s - 1] - 1 for s in ss])) for ss in P.values()) + " |"]
    variants = {
        "V1 기본 (현재 봇)": np.where(base, nh, np.nan),
        "V2 + 합성 상위 50%": np.where(base & (cpct >= 0.5), nh, np.nan),
        "V3 + 합성 상위 30%": np.where(base & (cpct >= 0.7), nh, np.nan),
        "V4 신고가 + 합성 순위 합": np.where(base, pct_rank(nh, base) + np.nan_to_num(cpct, nan=0.0), np.nan),
    }
    for name, sc in variants.items():
        sp = E.Spec(name, sc, "rotate", r=21).prepare()
        cells = []
        for ss in P.values():
            d = E.month_dist(m, sp, ss, 2)
            cells.append(f"{fmt(d['mean'])} (중앙 {fmt(d['median'])}, −10%↓ {d['p10d'] * 100:.0f}%, +10%↑ {d['p10u'] * 100:.0f}%)")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
        print(name, cells, flush=True)
    import json
    with open(os.path.join(OUT, "lab_combo_weights.json"), "w", encoding="utf-8") as fh:
        json.dump({"릿지(거래대금30억↑, IS 2011~2018)": W["릿지"]}, fh, ensure_ascii=False, indent=1)
    p = os.path.join(OUT, "lab_combo.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
