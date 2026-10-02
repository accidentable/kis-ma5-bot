"""
backtest/lab/flows.py — 기관·외국인 수급 전략 검증 (2019 ~ 2024)

데이터: github.com/timesavingbee/google 의 csv_file_folder/A{코드}.csv (pykrx 형식, 약 1,395종목,
2019-01 ~ 2024-12). 종목별 기관합계·외국인합계 거래대금 순매수.

  python -m backtest.lab.flows <csv 폴더>     → backtest/results/lab_flows.md

주의: 이 1,395종목은 누군가 2024년 무렵 고른 목록이라 생존편향이 있다. 그래서 절대 성과보다
"같은 종목 집합 안에서 수급 신호가 있을 때 vs 없을 때" 를 비교한다 (수급 없는 동일 규칙이 기준선).
구간: 2019-07 ~ 2021-12 시작 = 선택, 2022-01 ~ 2024-11 시작 = 검증.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

from backtest.lab import engine as E
from backtest.lab.search import HDR, OUT, fmt, market_row, row, where
from backtest.lab.search2 import _rank

warnings.filterwarnings("ignore")


def load_flows(m, folder: str):
    codes = {c: j for j, c in enumerate(m.codes)}
    di = {d: i for i, d in enumerate(m.dates)}
    inst = np.full(m.c.shape, np.nan)
    frgn = np.full(m.c.shape, np.nan)
    have = np.zeros(len(m.codes), bool)
    for p in glob.glob(os.path.join(folder, "A*.csv")):
        code = os.path.basename(p)[1:7]
        j = codes.get(code)
        if j is None:
            continue
        try:
            df = pd.read_csv(p, usecols=["날짜", "기관합계_거래대금_순매수", "외국인합계_거래대금_순매수"])
        except Exception:
            continue
        idx = [di.get(d.replace("-", "")) for d in df["날짜"]]
        ok = [i is not None for i in idx]
        rows = np.array([i for i in idx if i is not None], dtype=int)
        if not len(rows):
            continue
        inst[rows, j] = df["기관합계_거래대금_순매수"].to_numpy(float)[ok]
        frgn[rows, j] = df["외국인합계_거래대금_순매수"].to_numpy(float)[ok]
        have[j] = True
    return inst, frgn, have


def main() -> int:
    folder = sys.argv[1]
    m = E.load()
    inst, frgn, have = load_flows(m, folder)
    print("수급 종목", int(have.sum()), flush=True)
    I = m.ind
    cap = m.cap.astype(np.float64)
    covered = np.isfinite(inst)
    specs = []

    def add(fam, sp):
        sp.params["fam"] = fam
        sp.prepare()
        sp.score = None
        specs.append(sp)

    for n in (5, 20):
        i_n = E._roll_mean(np.nan_to_num(inst), n) * n / cap
        f_n = E._roll_mean(np.nan_to_num(frgn), n) * n / cap
        for u in ("LARGE", "MID"):
            U = E.universe(m, u) & covered & have[None, :]
            for r in (5, 21):
                add("수급-기관", E.Spec(f"기관 {n}일 순매수/시총 r{r} {u}", where(U & (i_n > 0), i_n), "rotate", r=r, params={"u": u}))
                add("수급-외인+기관", E.Spec(f"외인+기관 {n}일/시총 r{r} {u}", where(U & (i_n + f_n > 0), i_n + f_n), "rotate",
                                          r=r, params={"u": u}))
            comp = _rank(-I["vol60"], U) + _rank(-I["max20"], U) + _rank(-I["ret20"], U)
            add("합성+기관", E.Spec(f"저변동·저MAX·반전 + 기관{n}일 r21 {u}", where(U, comp + _rank(i_n, U)), "rotate", r=21,
                                  params={"u": u}))
            if n == 5:
                add("기준선(수급 없음)", E.Spec(f"저변동·저MAX·반전 r21 {u} (같은 종목 집합)", where(U, comp), "rotate", r=21,
                                            params={"u": u}))
                add("기준선(수급 없음)", E.Spec(f"무작위 대용: 시총순 r21 {u} (같은 종목 집합)", where(U, -np.log(cap)), "rotate",
                                            r=21, params={"u": u}))

    def st(a, z, step):
        s0, s1 = m.didx(a), min(m.didx(z), m.T - E.MONTH)
        return list(range(s0, s1, step))

    SEL, CHK = st("20190701", "20211231", 2), st("20220101", "20241130", 1)
    L = ["# 수급(기관·외국인) 전략 검증 — 2019 ~ 2024", "",
         f"- 수급 데이터 종목 {int(have.sum())}개 (timesavingbee/google, pykrx 형식). 생존편향이 있어 같은 종목 집합 기준선과 비교",
         f"- 선택 구간 {len(SEL)}개 (2019.7 ~ 2021), 검증 구간 {len(CHK)}개 (2022 ~ 2024.11). 한 달·60만원·2슬롯 기준", ""]
    L += ["## 선택 구간 (2슬롯)", ""] + HDR + [market_row(m, SEL, "선택")]
    res = [(sp, E.month_dist(m, sp, SEL, 2)) for sp in specs]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "선택", d))
    best = {}
    for sp, d in res:
        f = sp.params["fam"]
        if f not in best or d["mean"] > best[f][1]["mean"]:
            best[f] = (sp, d)
    L += ["", "## 계열별 선택 → 검증 구간", ""] + HDR + [market_row(m, CHK, "검증")]
    for f, (sp, _) in sorted(best.items()):
        for k in (1, 2, 3):
            L.append(row(sp, k, "검증", E.month_dist(m, sp, CHK, k)))
    p = os.path.join(OUT, "lab_flows.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
