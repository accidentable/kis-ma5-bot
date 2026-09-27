"""
backtest/lab/surge_exit.py — 보유 종목이 하루 급등하면 그날 종가에 판다 (급등 뒤 부진 신호를 청산에 쓰기)

  python -m backtest.lab.surge_exit    → backtest/results/lab_surge_exit.md

기본 = 현재 봇 (52주 신고가 근접, 시총 200, 30만원 이하, 60일 > 0, 20일 내 +10% 급등일 없음, 2종목).
빈 슬롯은 매일 그날 순위로 채운다 (봇의 NH_REFILL 과 같은 방식). 21거래일 보유 뒤 판다.
  S0 급등 청산 없음   S1 하루 +10%↑ 면 그날 종가 매도   S2 하루 +7%↑ 면 그날 종가 매도
  (참고) R0 현재 봇 방식 = 21일마다 교체(순위 안이면 유지)
"""
import os, sys, warnings
import numpy as np
from backtest.lab import engine as E
from backtest.lab.search import OUT, fmt, starts
warnings.filterwarnings("ignore")


def main():
    m = E.load(); I = m.ind
    base = ((~np.isnan(m.c)) & (m.raw >= 1000) & (m.raw <= 300_000) & (I["val20"] >= 1e9) & (m.val > 0)
            & (I["caprank"] <= 200) & (I["ret60"] > 0) & (I["max20"] < 0.10))
    sc = np.where(base, m.c / I["hi250"], np.nan)
    P = {"2011~2018": starts(m, "20110101", "20181231", 2), "2019~2024.9": starts(m, "20190101", "20240930", 2),
         "2024.10~": starts(m, "20241001", "99999999", 1), "최근 12개월": starts(m, "20250901", "99999999", 1)}
    V = {"R0 21일 교체 (현재 봇)": E.Spec("R0", sc, "rotate", r=21),
         "S0 21일 보유, 빈 슬롯 매일 채움": E.Spec("S0", sc, "signal", hold=21),
         "S1 + 하루 +10%↑ 종가 매도": E.Spec("S1", sc, "signal", hold=21, exit=I["ret1"] >= 0.10),
         "S2 + 하루 +7%↑ 종가 매도": E.Spec("S2", sc, "signal", hold=21, exit=I["ret1"] >= 0.07)}
    L = [(__doc__ or "").strip(), "", "| 변형 | " + " | ".join(P) + " |", "|---|" + "---|" * len(P)]
    for name, sp in V.items():
        sp.prepare()
        cells = []
        for ss in P.values():
            d = E.month_dist(m, sp, ss, 2)
            cells.append(f"{fmt(d['mean'])} (−10%↓ {d['p10d']*100:.0f}%, +10%↑ {d['p10u']*100:.0f}%, 월 {d['trades']:.1f}건)")
        L.append(f"| {name} | " + " | ".join(cells) + " |"); print(name, cells, flush=True)
    open(os.path.join(OUT, "lab_surge_exit.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
