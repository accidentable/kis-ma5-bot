"""
backtest/lab/knife.py — 관심 쏠린 종목의 급락(과매도) 순간 매수 = '칼날 잡기' 검증

  python -m backtest.lab.knife    → backtest/results/lab_knife.md

데이터: KRX 전종목(상장폐지 포함), 가격제한 ±30% 이후 2015.7 ~ 2026.9.

관심 종목 (급락 전날까지의 정보)
  A1 최근 20거래일 중 하루라도 시장 거래대금 50위 안
  A2 60거래일 수익률 +50% 이상 (급등했던 종목)
급락 순간 (당일 종가 기준)
  D1 하루 −10% 이하      D2 하루 −15% 이하      D3 3일 −20% 이하
  D4 20일 고점 대비 −30% 이하      D5 RSI(2) < 5      D6 하루 −10% 이하 + 거래대금 20일 평균 3배↑ (투매)

1) 사건 조사: 당일 종가 매수 / 다음 날 시가 매수 → 1·3·5·10·21일 뒤 종가. 비용(왕복 약 0.35%) 차감,
   같은 날 전체 평균 대비 초과수익도. 연도별 5일 수익률.
2) 대회 시뮬레이션 (60만원 · 한 달 · 2종목; IS 2015.7 ~ 2020 선택 → VAL 2021 ~ 2023 → TEST 2024 ~)
   격자 (사전 고정 24개): 관심 {A1, A2} × 급락 {D1, D3, D4, D5, D6, D2}
                           × 청산 {5일 보유, +10% 익절 · −10% 손절 · 10일}
   매수는 다음 날 시가 (시뮬레이터 'signal' 모드). 같은 날 후보가 여럿이면 가장 많이 빠진 순.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.search import HDR, OUT, fmt, market_row, row

warnings.filterwarnings("ignore")
COST = 0.0035


def main() -> int:
    m = E.load()
    I = m.ind
    c, o = m.c.astype(np.float64), m.o.astype(np.float64)
    val = m.val.astype(np.float64)
    ret1 = I["ret1"]
    ok = (~np.isnan(c)) & (m.raw >= 1000) & (val > 0) & (I["val20"] >= 1e9)
    vrank = (-np.nan_to_num(val, nan=-1)).argsort(axis=1).argsort(axis=1) + 1
    top50 = E._shift((E._roll_max((vrank <= 50).astype(np.float64), 20) > 0).astype(np.float64), 1) > 0
    runup = E._shift(I["ret60"], 1) >= 0.5
    ret3 = c / E._shift(c, 3) - 1
    dd20 = c / E._roll_max(m.h.astype(np.float64), 20) - 1
    vr = val / E._shift(I["val20"], 1)
    A = {"A1 거래대금 50위 경험": top50, "A2 60일 +50% 급등주": runup}
    D = {"D1 하루 −10%↓": ret1 <= -0.10, "D2 하루 −15%↓": ret1 <= -0.15, "D3 3일 −20%↓": ret3 <= -0.20,
         "D4 20일 고점 −30%↓": dd20 <= -0.30, "D5 RSI2<5": I["rsi2"] < 5,
         "D6 투매 (−10%↓ & 거래대금 3배↑)": (ret1 <= -0.10) & (vr >= 3)}
    t0 = m.didx("20150701")
    yrs = np.array([d[:4] for d in m.dates])

    # 1) 사건 조사
    L = ["# 칼날 잡기 — 관심 종목 급락 매수", "", (__doc__ or "").split("\n\n", 1)[1].strip(), "",
         "## 1) 사건 조사 (비용 0.35% 차감 후 평균 수익률 / 같은 날 전체 대비 초과 / 승률)", ""]
    fwd_c = {n: E._shift(c, -n) / c - 1 for n in (1, 3, 5, 10, 21)}            # 당일 종가 매수
    nxo = E._shift(o, -1)
    fwd_o = {n: E._shift(c, -n) / nxo - 1 for n in (1, 3, 5, 10, 21)}          # 다음 날 시가 매수
    allmask = ok.copy()
    allmask[:t0] = False
    L += ["| 관심 × 급락 | 건수 | 매수 | 1일 | 3일 | 5일 | 10일 | 21일 |", "|---|---|---|---|---|---|---|---|"]
    bases = {}
    for lab, fw in (("당일 종가", fwd_c), ("다음날 시가", fwd_o)):
        for n, y in fw.items():
            bases[(lab, n)] = np.nanmean(np.where(allmask, y, np.nan), axis=1)
    ev_rows = []
    for an, a in A.items():
        for dn, d in D.items():
            msk = ok & a & d
            msk[:t0] = False
            for lab, fw in (("당일 종가", fwd_c), ("다음날 시가", fwd_o)):
                cells = []
                for n, y in fw.items():
                    s = msk & np.isfinite(y)
                    if s.sum() < 30:
                        cells.append("-")
                        continue
                    base = bases[(lab, n)]
                    ex = y[s] - np.broadcast_to(base[:, None], y.shape)[s]
                    net = y[s] - COST
                    cells.append(f"{net.mean() * 100:+.1f}% / {ex.mean() * 100:+.1f}% / {(net > 0).mean() * 100:.0f}%")
                n_ev = int((msk & np.isfinite(fw[5])).sum())
                L.append(f"| {an} × {dn} | {n_ev:,} | {lab} | " + " | ".join(cells) + " |")
                if lab == "다음날 시가":
                    y5 = fw[5]
                    s = msk & np.isfinite(y5)
                    ev_rows.append((f"{an} × {dn}", [np.nanmean(np.where(s & (yrs == y)[:, None], y5, np.nan)) - COST
                                                      for y in ("2016", "2018", "2020", "2022", "2024", "2025", "2026")]))
            print(an, dn, flush=True)
    L += ["", "### 다음 날 시가 매수 → 5일 뒤 종가, 연도별 (비용 후)", "",
          "| 관심 × 급락 | 2016 | 2018 | 2020 | 2022 | 2024 | 2025 | 2026 |", "|---|---|---|---|---|---|---|---|"]
    for name, v in ev_rows:
        L.append(f"| {name} | " + " | ".join("-" if not np.isfinite(x) else f"{x * 100:+.1f}%" for x in v) + " |")

    # 2) 대회 시뮬레이션
    depth = -np.minimum(np.minimum(ret1, ret3), dd20)
    specs = []
    for an, a in A.items():
        for dn, d in D.items():
            sc = np.where(ok & a & d & (m.raw <= 300_000), depth, np.nan)
            for ex_name, kw in (("5일 보유", {"hold": 5}), ("+10% 익절·−10% 손절·10일", {"hold": 10, "tp": 0.10, "stop": 0.10})):
                sp = E.Spec(f"{an} × {dn} · {ex_name}", sc, "signal", params={"fam": "칼날"}, **kw)
                sp.prepare()
                sp.score = None
                specs.append(sp)

    def st(a_, z, step):
        s0, s1 = m.didx(a_), min(m.didx(z), m.T - E.MONTH)
        return list(range(max(s0, 260), s1, step))

    P = {"IS": st("20150701", "20201231", 2), "VAL": st("20210101", "20231231", 2), "TEST": st("20240101", "99999999", 1)}
    res = [(sp, E.month_dist(m, sp, P["IS"], 2)) for sp in specs]
    L += ["", "## 2) 대회 시뮬레이션 — IS (2슬롯)", ""] + HDR + [market_row(m, P["IS"], "IS")]
    for sp, dd in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "IS", dd))
    best = max(res, key=lambda x: x[1]["mean"])[0]
    L += ["", f"## IS 1위 → VAL · TEST: {best.name}", ""] + HDR
    for tag in ("VAL", "TEST"):
        L.append(market_row(m, P[tag], tag))
        for k in (1, 2, 3):
            L.append(row(best, k, tag, E.month_dist(m, best, P[tag], k)))
    L += ["", "## 참고: 전 설정 2슬롯 (선택에 안 씀)", "", "| 설정 | IS | VAL | TEST |", "|---|---|---|---|"]
    for sp, dd in sorted(res, key=lambda x: -x[1]["mean"]):
        v, t = E.month_dist(m, sp, P["VAL"], 2), E.month_dist(m, sp, P["TEST"], 2)
        L.append(f"| {sp.name} | {fmt(dd['mean'])} | {fmt(v['mean'])} | {fmt(t['mean'])} |")
    p = os.path.join(OUT, "lab_knife.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
