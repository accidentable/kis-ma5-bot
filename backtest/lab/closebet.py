"""
backtest/lab/closebet.py — 종가 베팅 (강세 마감 테마주를 종가에 사서 다음 날 시가에 판다), 한 달 대회 기준

  python -m backtest.lab.closebet     → backtest/results/lab_closebet.md

근거가 된 관찰 (theme.py 조사 중, 2015.7 ~ 2026.9 전종목):
  거래대금 30위 안에서 +10 ~ +29% 로 고가 근처(IBS>0.9) 마감한 종목의 종가→다음 날 시가 평균 +0.48%,
  전체 종목 평균 +0.17%. 대신 다음 날 종가까지 들고 가면 −0.31% (장중에 되밀린다).
이 관찰을 본 뒤 만든 계열이라, 선택은 IS(2015.7 ~ 2020) 에서만 하고 VAL(2021 ~ 2023)·TEST(2024 ~) 로 확인한다.

체결 가정: 15:20 직전 가격으로 조건을 판정하고 장마감 동시호가에 산다 → 판정가 ≈ 종가로 본다 (약간 낙관).
다음 날 장전 동시호가(시가)에 판다. 비용: 수수료 양쪽 + 매도세 0.20% + 틱 슬리피지 양쪽.
상한가(+29%↑) 마감 종목은 못 산다고 보고 뺀다.

격자 (사전 고정, 16개): 등락률 하한 {+5%, +10%} × IBS 하한 {0.8, 0.9} × 거래대금 순위 {30, 100} × 국면 {off, on}
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.search import HDR, OUT, fmt, market_row, row
from backtest.lab.theme import IS, TEST, VAL, starts

warnings.filterwarnings("ignore")


def specs_for(m) -> list:
    I = m.ind
    c, h, l = (x.astype(np.float64) for x in (m.c, m.h, m.l))
    val = m.val.astype(np.float64)
    vrank = (-np.nan_to_num(val, nan=-1)).argsort(axis=1).argsort(axis=1).astype(np.float32) + 1
    ibs = (c - l) / np.where(h > l, h - l, np.nan)
    ret1 = I["ret1"]
    ok = (~np.isnan(c)) & (m.raw >= 1000) & (val >= 5e9) & (ret1 < 0.29)
    out = []
    for lo in (0.05, 0.10):
        for ib in (0.8, 0.9):
            for vr in (30, 100):
                msk = ok & (ret1 >= lo) & (ibs >= ib) & (vrank <= vr)
                sc = np.where(msk, val, np.nan)
                for reg in (False, True):
                    sp = E.Spec(f"종가베팅 +{lo * 100:.0f}%↑ IBS≥{ib} 거래대금{vr}위{' 국면' if reg else ''}", sc, "signal",
                                entry_at_close=True, regime=reg, params={"fam": "종가베팅"})
                    sp.prepare()
                    sp.score = None
                    out.append(sp)
    return out


def main() -> int:
    t0 = time.time()
    m = E.load()
    specs = specs_for(m)
    S = {n: starts(m, *p) for n, p in (("IS", IS), ("VAL", VAL), ("TEST", TEST))}
    L = ["# 종가 베팅 — 한 달 대회 기준", "", (__doc__ or "").split("근거가 된 관찰")[1].join(["근거가 된 관찰", ""]) if False else "",
         f"IS {len(S['IS'])}개(2015.7 ~ 2020) → VAL {len(S['VAL'])}개(2021 ~ 2023) → TEST {len(S['TEST'])}개(2024 ~)", ""]
    res = [(sp, E.month_dist(m, sp, S["IS"], 2)) for sp in specs]
    L += ["## ① IS 전 설정 (2슬롯)", ""] + HDR + [market_row(m, S["IS"], "IS")]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "IS", d))
    ranked = sorted(res, key=lambda x: -x[1]["mean"])
    best = ranked[0][0]
    L += ["", f"## ② IS 1위 → VAL · TEST: {best.name}", ""] + HDR
    for tag in ("VAL", "TEST"):
        L.append(market_row(m, S[tag], tag))
        for k in (1, 2, 3):
            L.append(row(best, k, tag, E.month_dist(m, best, S[tag], k)))
    # 참고: 전 설정의 VAL (선택엔 안 씀 — 결과가 한 설정에만 기대는지 보려는 것)
    L += ["", "## ③ 참고: 전 설정 VAL · TEST 2슬롯 (선택에 안 씀)", "", "| 설정 | IS | VAL | TEST |", "|---|---|---|---|"]
    for sp, d in ranked:
        v = E.month_dist(m, sp, S["VAL"], 2)
        t = E.month_dist(m, sp, S["TEST"], 2)
        L.append(f"| {sp.name} | {fmt(d['mean'])} | {fmt(v['mean'])} | {fmt(t['mean'])} |")
    # ④ 연도별
    allst = starts(m, "20150701", "99999999", 2)
    d = E.month_dist(m, best, allst, 2)
    mk = m.ind["mkt"]
    L += ["", f"## ④ 연도별 — {best.name} (2슬롯)", "", "| 연도 | 전략 한 달 평균 | 시장 | +10%↑ | −10%↓ |", "|---|---|---|---|---|"]
    for y in sorted({m.dates[s][:4] for s in allst}):
        idx = [n for n, s in enumerate(allst) if m.dates[s][:4] == y]
        a = d["rets"][idx]
        mm = np.array([mk[allst[n] + 20] / mk[allst[n] - 1] - 1 for n in idx])
        L.append(f"| {y} | {fmt(a.mean())} | {fmt(mm.mean())} | {(a >= .1).mean() * 100:.0f}% | {(a <= -.1).mean() * 100:.0f}% |")
    p = os.path.join(OUT, "lab_closebet.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p, round(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
