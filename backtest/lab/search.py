"""
backtest/lab/search.py — 전종목 16년(2010 ~ 2026) 한 달 대회 전략 탐색

  python -m backtest.lab.search      → backtest/results/lab_search.md

순서 (실행 전에 고정)
  IS   2011-01 ~ 2018-12 시작 구간: 계열별로 설정을 고른다 (2슬롯 한 달 평균 최고, −10%↓ 30% 이하)
  VAL  2019-01 ~ 2024-09 시작 구간: 계열별 대표끼리 비교해 최종 후보를 고른다
  TEST 2024-10 ~ 끝      시작 구간: 최종 후보를 한 번만 본다 (고르는 데 안 쓴다)
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from backtest.lab import engine as E

warnings.filterwarnings("ignore")
OUT = os.path.join(E.ROOT, "backtest", "results")

IS = ("20110101", "20181231", 3)
VAL = ("20190101", "20240930", 2)
TEST = ("20241001", "99999999", 1)


def starts(m, a, z, step):
    s0, s1 = m.didx(a), min(m.didx(z), m.T - E.MONTH)
    return list(range(max(s0, 260), s1, step))


def where(mask, score):
    return np.where(mask & np.isfinite(score), score, np.nan)


def build_specs(m) -> list:
    I = m.ind
    c = m.c
    Us = {u: E.universe(m, u) for u in ("LARGE", "MID", "LIQ")}
    trend120 = c > I["ma120"]
    ex_up5, ex_dn10 = c > I["ma5"], c < I["ma10"]
    specs = []

    def add(fam, sp):
        sp.params["fam"] = fam
        sp.prepare()
        sp.score = None          # 후보 목록만 남기고 행렬은 버린다 (메모리)
        specs.append(sp)

    for u, U in Us.items():
        # A. 모멘텀 로테이션
        for L in ("ret20", "ret60", "ret120", "mom12_1"):
            for r in (5, 21):
                for reg in (False, True):
                    sc = where(U & (c > I["ma60"]) & (I[L] > 0), I[L])
                    add("A.모멘텀", E.Spec(f"모멘텀 {L} r{r}{' 국면' if reg else ''} {u}", sc, "rotate", r=r, keep=0, regime=reg,
                                          params={"u": u}))
        # B. 신고가 근접 (George-Hwang, 250일)
        for r in (5, 21):
            for reg in (False, True):
                sc = where(U & (I["ret60"] > 0), c / I["hi250"])
                add("B.신고가근접", E.Spec(f"신고가근접 r{r}{' 국면' if reg else ''} {u}", sc, "rotate", r=r, regime=reg, params={"u": u}))
        # C. 단기 반전 (5일 낙폭 상위)
        for tf in (False, True):
            for stop in (0.0, 0.10):
                m_ = U & (I["ret5"] < -0.05) & (trend120 if tf else True)
                add("C.단기반전", E.Spec(f"단기반전{' 추세' if tf else ''}{' 손절10' if stop else ''} {u}", where(m_, -I["ret5"]),
                                        "signal", exit=ex_up5, hold=5, stop=stop, params={"u": u}))
        # D. 주도주 눌림: 60일 수익률 상위 20 안에서 RSI(2) < 25
        rk = np.where(U & (c > I["ma60"]), I["ret60"], -np.inf)
        thr = -np.sort(-rk, axis=1)[:, 19:20]
        lead = U & (rk >= thr) & np.isfinite(rk)
        for stop in (0.0, 0.10):
            add("D.주도주눌림", E.Spec(f"주도주눌림{' 손절10' if stop else ''} {u}", where(lead & (I["rsi2"] < 25), -I["rsi2"]),
                                      "signal", exit=ex_up5, hold=10, stop=stop, params={"u": u}))
        # E. 거래대금 급증 돌파
        vr = m.val / E._shift(I["val20"], 1)
        for mult in (3, 5):
            for stop in (0.0, 0.10):
                m_ = U & (c > I["hh20"]) & (vr >= mult) & (c > I["ma60"])
                add("E.거래대금돌파", E.Spec(f"거래대금돌파 x{mult}{' 손절10' if stop else ''} {u}", where(m_, vr), "signal",
                                         exit=ex_dn10, hold=15, stop=stop, params={"u": u}))
        # F. 저변동성 (방어형)
        for reg in (False, True):
            add("F.저변동", E.Spec(f"저변동 r21{' 국면' if reg else ''} {u}", where(U, -I["vol60"]), "rotate", r=21, regime=reg,
                                  params={"u": u}))
        # G. 소형 (그 유니버스 안에서 시총 작은 순)
        for reg in (False, True):
            add("G.소형", E.Spec(f"소형 r21{' 국면' if reg else ''} {u}", where(U, -np.log(m.cap)), "rotate", r=21, regime=reg,
                                params={"u": u}))
        # H. 상한가 다음 날 (종가 +29% 이상 마감 → 다음 날 시가 매수)
        for hold in (1, 3):
            add("H.상한가추종", E.Spec(f"상한가추종 {hold}일 {u}", where(U & (I["ret1"] >= 0.29), m.val.astype(float)), "signal",
                                     hold=hold, params={"u": u}))
        # I. 복권 회피 모멘텀: 60일 상승 + 최근 20일 최대 일간 상승 작은 순 (MAX 효과)
        for r in (5, 21):
            sc = where(U & (I["ret60"] > 0.1) & (c > I["ma60"]), -I["max20"])
            add("I.MAX회피모멘텀", E.Spec(f"MAX회피모멘텀 r{r} {u}", sc, "rotate", r=r, params={"u": u}))
    return specs


def fmt(x):
    return f"{x * 100:+.1f}%"


def row(sp, k, tag, d):
    win = "-" if d["win"] != d["win"] else f"{d['win'] * 100:.0f}%"
    return (f"| {sp.name} | {k} | {tag} | {fmt(d['mean'])} | {fmt(d['median'])} | {d['p_pos'] * 100:.0f}% | "
            f"{d['p10u'] * 100:.0f}% | {d['p20u'] * 100:.0f}% | {d['p10d'] * 100:.0f}% | {fmt(d['q10'])} / {fmt(d['q90'])} | "
            f"{d['trades']:.1f} | {win} |")


HDR = ["| 전략 | 슬롯 | 구간 | 한 달 평균 | 중앙값 | 플러스 | +10%↑ | +20%↑ | −10%↓ | 하위10 / 상위10 | 월 거래 | 승률 |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]


def market_row(m, ss, tag):
    mk = m.ind["mkt"]
    a = np.array([mk[s + E.MONTH - 1] / mk[s - 1] - 1 for s in ss])
    return (f"| 시장 (거래대금 30억↑ 동일가중, 비용 전) | - | {tag} | {fmt(a.mean())} | {fmt(float(np.median(a)))} | {(a > 0).mean() * 100:.0f}% | "
            f"{(a >= .1).mean() * 100:.0f}% | {(a >= .2).mean() * 100:.0f}% | {(a <= -.1).mean() * 100:.0f}% | - | - | - |")


def main() -> int:
    t0 = time.time()
    m = E.load()
    print("load", round(time.time() - t0), flush=True)
    specs = build_specs(m)
    print("specs", len(specs), round(time.time() - t0), flush=True)
    S = {n: starts(m, *p) for n, p in (("IS", IS), ("VAL", VAL), ("TEST", TEST))}

    L = ["# 전종목 16년 — 한 달 대회 전략 탐색", "",
         f"- 데이터: KRX 전종목 일별 (FinanceData/marcap, 상장폐지 포함) {m.dates[0]} ~ {m.dates[-1]}, {len(m.codes)}종목",
         "- 한 달 = 21거래일, 60만원, 정수 주식, 첫날 시가 매수 · 마지막 날 종가 전량 매도, 비용 매도세 0.20% + 수수료 + 틱 슬리피지",
         f"- IS {len(S['IS'])}개 구간(2011 ~ 2018) 에서 계열별 선택 → VAL {len(S['VAL'])}개(2019 ~ 2024.9) 에서 후보 결정 → TEST {len(S['TEST'])}개(2024.10 ~) 한 번",
         f"- 설정 {len(specs)}개 (실행 전에 고정)", ""]

    # ① IS 전부 (2슬롯)
    res = []
    for sp in specs:
        d = E.month_dist(m, sp, S["IS"], 2)
        res.append((sp, d))
    print("IS done", round(time.time() - t0), flush=True)
    L += ["## ① IS 전 설정 (2슬롯)", ""] + HDR + [market_row(m, S["IS"], "IS")]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "IS", d))

    best = {}
    for sp, d in res:
        f = sp.params["fam"]
        if d["p10d"] > 0.30:
            continue
        if f not in best or d["mean"] > best[f][1]["mean"]:
            best[f] = (sp, d)

    # ② 계열 대표 → VAL (1·2·3슬롯)
    L += ["", "## ② 계열별 IS 최선 → VAL", ""] + HDR + [market_row(m, S["VAL"], "VAL")]
    val = []
    for f, (sp, _) in sorted(best.items()):
        for k in (1, 2, 3):
            d = E.month_dist(m, sp, S["VAL"], k)
            L.append(row(sp, k, "VAL", d))
            if k == 2:
                val.append((sp, d))
    print("VAL done", round(time.time() - t0), flush=True)

    # ③ VAL 상위 3개 → TEST (한 번)
    top3 = sorted(val, key=lambda x: -x[1]["mean"])[:3]
    L += ["", "## ③ VAL 2슬롯 한 달 평균 상위 3 → TEST (한 번만)", ""] + HDR + [market_row(m, S["TEST"], "TEST")]
    for sp, _ in top3:
        for k in (1, 2, 3):
            L.append(row(sp, k, "TEST", E.month_dist(m, sp, S["TEST"], k)))

    # ④ 연도별 (VAL 1위, 2슬롯)
    sp0 = top3[0][0]
    L += ["", f"## ④ 연도별 한 달 평균 — {sp0.name} (2슬롯) vs 시장", "", "| 연도 | 전략 | 시장 | +10%↑ | −10%↓ |", "|---|---|---|---|---|"]
    allst = list(range(260, m.T - E.MONTH, 2))
    d = E.month_dist(m, sp0, allst, 2)
    mk = m.ind["mkt"]
    yrs = sorted({m.dates[s][:4] for s in allst})
    for y in yrs:
        idx = [n for n, s in enumerate(allst) if m.dates[s][:4] == y]
        a = d["rets"][idx]
        mm = np.array([mk[allst[n] + 20] / mk[allst[n] - 1] - 1 for n in idx])
        L.append(f"| {y} | {fmt(a.mean())} | {fmt(mm.mean())} | {(a >= .1).mean() * 100:.0f}% | {(a <= -.1).mean() * 100:.0f}% |")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "lab_search.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p, round(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
