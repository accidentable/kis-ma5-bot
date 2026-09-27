"""
backtest/lab/sens_nearhigh.py — 최종 후보(52주 신고가 근접, 시총 상위 200, 2종목, 21일) 민감도

선택이 끝난 뒤 설정을 바꿔 결과가 한 점에만 기대는지 본다 (여기서 다시 고르지 않는다).
  python -m backtest.lab.sens_nearhigh   → backtest/results/lab_sens_nearhigh.md
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.search import OUT, fmt, starts, where

warnings.filterwarnings("ignore")


def main():
    m = E.load()
    I, c = m.ind, m.c
    base = (~np.isnan(c)) & (m.raw >= 1000) & (I["val20"] >= 1e9) & (m.val > 0)
    hi120 = E._roll_max(m.h.astype(np.float64), 120)
    periods = {"2011~2018": starts(m, "20110101", "20181231", 2), "2019~2024.9": starts(m, "20190101", "20240930", 2),
               "2024.10~": starts(m, "20241001", "99999999", 1), "최근 12개월": starts(m, "20250901", "99999999", 1)}
    mk = I["mkt"]
    L = ["# 52주 신고가 근접 — 민감도 (선택 뒤 점검)", "",
         "기본: 시총 상위 200 (코스피+코스닥, 그날 기준), 종가/250일 최고가, 60일 수익률 > 0, 2종목, 21거래일 보유", "",
         "| 변형 | " + " | ".join(periods) + " |", "|---|" + "---|" * len(periods)]
    L.append("| 시장 (거래대금 30억↑ 동일가중, 비용 전) | " + " | ".join(
        fmt(np.mean([mk[s + 20] / mk[s - 1] - 1 for s in ss])) for ss in periods.values()) + " |")

    def run(name, rank_max=200, hi=None, k=2, r=21, mom=True, max_raw=None, max_gain=None):
        U = base & (I["caprank"] <= rank_max) & ((m.raw <= max_raw) if max_raw else True)
        if max_gain:
            U = U & (I["max20"] < max_gain)
        hh = I["hi250"] if hi is None else hi
        sc = where(U & ((I["ret60"] > 0) if mom else True), c / hh)
        sp = E.Spec(name, sc, "rotate", r=r).prepare()
        cells = []
        for ss in periods.values():
            d = E.month_dist(m, sp, ss, k)
            cells.append(f"{fmt(d['mean'])} (−10%↓ {d['p10d'] * 100:.0f}%)")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
        print(name, cells, flush=True)

    run("기본")
    run("시총 상위 100", rank_max=100)
    run("시총 상위 300", rank_max=300)
    run("120일 최고가", hi=hi120)
    run("60일 수익률 필터 없음", mom=False)
    run("1종목", k=1)
    run("3종목", k=3)
    run("10일마다 교체", r=10)
    run("30만원 이하만 순위 (실전 봇 방식)", max_raw=300_000)
    # 아래 둘은 테마주 조사(theme.py) 뒤에 추가한 것 — 급등주는 다음 날 장중에 되밀린다는 관찰, MAX 효과 연구
    run("30만원 이하 + 20일 내 +15%↑ 급등일 제외", max_raw=300_000, max_gain=0.15)
    run("30만원 이하 + 20일 내 +10%↑ 급등일 제외 (실전 봇 기본값)", max_raw=300_000, max_gain=0.10)
    with open(os.path.join(OUT, "lab_sens_nearhigh.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
