"""
backtest/lab/sector.py — 당일 강세 섹터(업종) 매수 전략 검증 ("오늘 강한 섹터 ETF 의 종목을 산다")

  python -m backtest.lab.sector <stock_master.csv.gz>    → backtest/results/lab_sector.md

업종: github.com/FinanceData/stock_master (2018-11 기준 KRX 표준산업분류 'Sector', 157개, 상장폐지 종목 포함).
  2018 이후 상장 종목은 업종이 없어 빠진다. 분류는 2018 시점 것을 전 기간에 쓴다.
섹터 등락률: 그날 거래대금 10억↑·주가 1,000원↑ 종목이 5개 이상인 업종의 동일가중 평균 등락률 (≈ 섹터 ETF).

1) 사건 조사: 강세 섹터 종목의 다음 날 · 5일 · 21일 수익률 (다음 날 시가 매수 기준, 비용 전) − 같은 날 전체 평균
2) 대회 시뮬레이션 (60만원 · 한 달 · 2종목, IS 2011 ~ 2018 에서 선택 → VAL 2019 ~ 2024.9 → TEST 2024.10 ~)
   격자 (사전 고정 18개): 섹터 {당일 1위, 당일 상위 3, 5일 1위} × 종목 {대장(시총 1위), 당일 최고 상승, 후발(당일 최저 상승)}
                           × 보유 {5일, 21일}
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from backtest.lab import engine as E
from backtest.lab.search import HDR, OUT, fmt, market_row, row

warnings.filterwarnings("ignore")


def sector_index(m, path):
    d = pd.read_csv(path, dtype=str)
    lab = dict(zip(d["Symbol"].str.zfill(6), d["Sector"]))
    secs = sorted({s for s in lab.values() if isinstance(s, str)})
    sid = {s: i for i, s in enumerate(secs)}
    col = np.array([sid.get(lab.get(c), -1) for c in m.codes])
    return col, secs


def sector_returns(m, col, nsec, ret, ok):
    """T × S 섹터 평균 수익률 (구성 5개 미만 NaN) 과 종목별 소속 섹터 수익률 T × N."""
    T = m.T
    sr = np.full((T, nsec), np.nan)
    for s in range(nsec):
        mem = col == s
        if mem.sum() < 5:
            continue
        x = np.where(ok[:, mem], ret[:, mem], np.nan)
        cnt = np.sum(np.isfinite(x), axis=1)
        sr[:, s] = np.where(cnt >= 5, np.nanmean(x, axis=1), np.nan)
    return sr


def main() -> int:
    path = sys.argv[1]
    m = E.load()
    I = m.ind
    col, secs = sector_index(m, path)
    S = len(secs)
    c = m.c.astype(np.float64)
    ok = (~np.isnan(c)) & (m.raw >= 1000) & (I["val20"] >= 1e9) & (m.val > 0)
    ok_lab = ok & (col >= 0)[None, :]
    sr1 = sector_returns(m, col, S, I["ret1"], ok_lab)
    sr5 = sector_returns(m, col, S, I["ret5"], ok_lab)
    cov = ok_lab.sum(1) / np.maximum(ok.sum(1), 1)
    L = ["# 당일 강세 섹터 매수 — 검증", "", (__doc__ or "").split("\n\n", 1)[1].split("1) 사건")[0].strip(), "",
         f"업종 라벨이 있는 종목 비중 (거래 가능 종목 중): 2012 {cov[m.didx('20120601')] * 100:.0f}% / "
         f"2018 {cov[m.didx('20180601')] * 100:.0f}% / 2022 {cov[m.didx('20220601')] * 100:.0f}% / "
         f"2026 {cov[m.didx('20260601')] * 100:.0f}%", ""]

    # 섹터 순위 (그날 기준)
    def top_secs(sr, k):
        out = np.zeros((m.T, S), bool)
        for t in range(m.T):
            v = sr[t]
            if np.isfinite(v).sum() < 10:
                continue
            idx = np.argsort(np.where(np.isfinite(v), -v, np.inf))[:k]
            out[t, idx] = True
        return out

    secsel = {"당일 1위 섹터": top_secs(sr1, 1), "당일 상위 3 섹터": top_secs(sr1, 3), "5일 1위 섹터": top_secs(sr5, 1)}
    member = {}
    for name, sel in secsel.items():
        mm = np.zeros_like(ok)
        for s in range(S):
            mm[:, col == s] |= sel[:, s:s + 1]
        member[name] = mm & ok_lab

    # 1) 사건 조사
    o = m.o.astype(np.float64)
    nxo = E._shift(o, -1)
    fwd = {n: E._shift(c, -n) / nxo - 1 for n in (1, 5, 21)}
    t0 = m.didx("20110101")
    L += ["## 1) 사건 조사 — 강세 섹터 종목의 이후 수익률 − 같은 날 전체 평균 (다음 날 시가 매수, 비용 전)", "",
          "| 섹터 선택 | 표본 | 다음 날 (시가→종가) | 5일 | 21일 | 21일, 2019 이후만 |", "|---|---|---|---|---|---|"]
    for name, mm in member.items():
        cells = []
        for n in (1, 5, 21):
            y = fwd[n]
            ex = []
            for t in range(t0, m.T - 22):
                sel = mm[t] & np.isfinite(y[t])
                allv = ok[t] & np.isfinite(y[t])
                if sel.sum() and allv.sum():
                    ex.append(y[t][sel].mean() - y[t][allv].mean())
            cells.append(f"{np.mean(ex) * 100:+.2f}%")
        ex21b = []
        y = fwd[21]
        for t in range(m.didx("20190101"), m.T - 22):
            sel = mm[t] & np.isfinite(y[t]); allv = ok[t] & np.isfinite(y[t])
            if sel.sum() and allv.sum():
                ex21b.append(y[t][sel].mean() - y[t][allv].mean())
        L.append(f"| {name} | {int(mm[t0:].sum()):,} | " + " | ".join(cells) + f" | {np.mean(ex21b) * 100:+.2f}% |")
        print(name, cells, flush=True)

    # 2) 대회 시뮬레이션
    cap = m.cap.astype(np.float64)
    pick = {"대장(시총 1위)": np.log(cap), "당일 최고 상승": I["ret1"], "후발(당일 최저 상승)": -I["ret1"]}
    specs = []
    afford = m.raw <= 300_000
    for sname, mm in member.items():
        for pname, sc in pick.items():
            base = np.where(mm & afford & np.isfinite(sc), sc, np.nan)
            if pname.startswith("대장"):
                # 섹터별 시총 1위만 남긴다
                lead = np.full_like(base, np.nan)
                for s in range(S):
                    cols = np.where(col == s)[0]
                    if not len(cols):
                        continue
                    sub = base[:, cols]
                    j = np.nanargmax(np.where(np.isfinite(sub), sub, -np.inf), axis=1)
                    has = np.isfinite(sub).any(1)
                    lead[np.where(has)[0], cols[j[has]]] = sub[np.where(has)[0], j[has]]
                base = lead
            for hold in (5, 21):
                sp = E.Spec(f"{sname} · {pname} · {hold}일", base, "signal", hold=hold,
                            params={"fam": "섹터"})
                sp.prepare()
                sp.score = None
                specs.append(sp)

    def st(a, z, step):
        s0, s1 = m.didx(a), min(m.didx(z), m.T - E.MONTH)
        return list(range(max(s0, 260), s1, step))

    P = {"IS": st("20110101", "20181231", 3), "VAL": st("20190101", "20240930", 2), "TEST": st("20241001", "99999999", 1)}
    res = [(sp, E.month_dist(m, sp, P["IS"], 2)) for sp in specs]
    L += ["", "## 2) 대회 시뮬레이션 — IS (2슬롯)", ""] + HDR + [market_row(m, P["IS"], "IS")]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "IS", d))
    best = max(res, key=lambda x: x[1]["mean"])[0]
    L += ["", f"## IS 1위 → VAL · TEST: {best.name}", ""] + HDR
    for tag in ("VAL", "TEST"):
        L.append(market_row(m, P[tag], tag))
        for k in (1, 2, 3):
            L.append(row(best, k, tag, E.month_dist(m, best, P[tag], k)))
    L += ["", "## 참고: 전 설정 2슬롯 (선택에 안 씀)", "", "| 설정 | IS | VAL | TEST |", "|---|---|---|---|"]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        v, t = E.month_dist(m, sp, P["VAL"], 2), E.month_dist(m, sp, P["TEST"], 2)
        L.append(f"| {sp.name} | {fmt(d['mean'])} | {fmt(v['mean'])} | {fmt(t['mean'])} |")
    p = os.path.join(OUT, "lab_sector.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
