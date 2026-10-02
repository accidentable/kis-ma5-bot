"""
backtest/lab/theme.py — 테마주(급등주) 전략 탐색, 한 달 대회 기준

  python -m backtest.lab.theme     → backtest/results/lab_theme.md

데이터는 engine.py 와 같다 (KRX 전종목, 상장폐지 포함 — 급등했다 사라진 테마주도 들어 있다).
가격제한폭이 ±30% 로 바뀐 2015-06-15 이후만 쓴다 (상한가 정의가 달라서).
  IS   2015-07 ~ 2020-12 시작 구간: 계열별 선택 (2슬롯 한 달 평균 최고, −10%↓ 35% 이하)
  VAL  2021-01 ~ 2023-12: 계열 대표 비교
  TEST 2024-01 ~ 끝     : VAL 상위 3개만 한 번

계열 (실행 전에 고정, 24개)
  T1 급등 대장주 추종  전일 +15%↑ 또는 상한가, 거래대금 100억↑·시장 거래대금 30위 안 → 시가 매수
                      청산: 당일 종가 / 다음 날 시가 / 3일(−7% 손절)
  T2 첫 눌림목        10일 안에 +25%↑ 급등(거래대금 100억↑)이 있었고, 고점 대비 −8 ~ −25% 눌림, 20일선 위
                      (거래 마름 조건 on/off) → 시가 매수, +10/+20% 익절 · −8% 손절 · 5일
  T3 신고가 점화      250일 신고가 + 당일 +10%↑ + 거래대금 20일 평균 5배↑·50억↑ → 시가 매수
                      청산: 5일선 이탈(10일, −10%) / 10일선 이탈(20일, −15%)
  T4 주간 테마 로테이션  5일 수익률 상위 (5일 평균 거래대금이 60일 평균의 2배↑) 를 5/10일마다 교체
  T6 과열주 눌림 매수   20일 +30%↑ · 거래대금 상위 20 인 종목이 당일 −5%↓ → 시가 매수, +7/+12% 익절 · −7% · 3일
  국면 필터: 시장(거래대금 30억↑ 동일가중) 100일선 위일 때만 신규 매수 — T1(3일), T2 에 on/off
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.search import HDR, OUT, fmt, market_row, row

warnings.filterwarnings("ignore")

IS = ("20150701", "20201231", 2)
VAL = ("20210101", "20231231", 2)
TEST = ("20240101", "99999999", 1)


def starts(m, a, z, step):
    s0, s1 = m.didx(a), min(m.didx(z), m.T - E.MONTH)
    return list(range(max(s0, 260), s1, step))


def roll_any(x: np.ndarray, n: int) -> np.ndarray:
    """최근 n일(오늘 포함) 중 하루라도 True."""
    return E._roll_max(x.astype(np.float64), n) > 0


def build(m) -> list:
    I = m.ind
    c = m.c.astype(np.float64)
    val = m.val.astype(np.float64)
    ok = (~np.isnan(c)) & (m.raw >= 1000) & (val > 0)
    # 그날 시장 전체 거래대금 순위
    vrank = (-np.nan_to_num(val, nan=-1)).argsort(axis=1).argsort(axis=1).astype(np.float32) + 1
    ret1 = I["ret1"]
    val20p = E._shift(I["val20"], 1)
    ex_up5, ex_dn5, ex_dn10 = c > I["ma5"], c < I["ma5"], c < I["ma10"]
    specs = []

    def add(fam, sp):
        sp.params["fam"] = fam
        sp.prepare()
        sp.score = None
        specs.append(sp)

    def sc(mask, score):
        return np.where(mask & np.isfinite(score), score, np.nan)

    # T1 급등 대장주 추종
    for thr, lab in ((0.15, "+15%↑"), (0.29, "상한가")):
        m1 = ok & (ret1 >= thr) & (val >= 1e10) & (vrank <= 30)
        s1 = sc(m1, val)
        add("T1.급등추종", E.Spec(f"급등추종 {lab} → 당일 종가", s1, "signal", hold=1))
        add("T1.급등추종", E.Spec(f"급등추종 {lab} → 다음날 시가", s1, "signal", hold=1, exit_next_open=True))
        for reg in (False, True):
            add("T1.급등추종", E.Spec(f"급등추종 {lab} → 3일 −7%{' 국면' if reg else ''}", s1, "signal", hold=3,
                                     stop=0.07, regime=reg))

    # T2 첫 눌림목
    surge = (ret1 >= 0.25) & (val >= 1e10)
    had = roll_any(E._shift(surge.astype(np.float64), 2) > 0, 9)        # 2 ~ 10일 전에 급등
    peak10 = E._roll_max(c, 10)
    pull = c / peak10 - 1
    vmax10 = E._roll_max(val, 10)
    base2 = ok & had & (pull <= -0.08) & (pull >= -0.25) & (c > I["ma20"])
    for dry in (False, True):
        m2 = base2 & ((val < 0.4 * vmax10) if dry else True)
        s2 = sc(m2, -pull)
        for tp in (0.10, 0.20):
            for reg in (False, True):
                add("T2.첫눌림", E.Spec(f"첫눌림{' 거래마름' if dry else ''} 익절{tp * 100:.0f} −8% 5일{' 국면' if reg else ''}",
                                       s2, "signal", hold=5, stop=0.08, tp=tp, regime=reg))

    # T3 신고가 점화
    m3 = ok & (c >= I["hi250"] * 0.999) & (ret1 >= 0.10) & (val >= 5 * val20p) & (val >= 5e9)
    s3 = sc(m3, val / val20p)
    for reg in (False, True):
        add("T3.신고가점화", E.Spec(f"신고가점화 5일선 10일 −10%{' 국면' if reg else ''}", s3, "signal", exit=ex_dn5, hold=10,
                                  stop=0.10, regime=reg))
        add("T3.신고가점화", E.Spec(f"신고가점화 10일선 20일 −15%{' 국면' if reg else ''}", s3, "signal", exit=ex_dn10, hold=20,
                                  stop=0.15, regime=reg))

    # T4 주간 테마 로테이션
    v5, v60 = E._roll_mean(val, 5), E._roll_mean(val, 60)
    m4 = ok & (I["val20"] >= 3e9) & (v5 >= 2 * v60) & (I["ret5"] > 0)
    s4 = sc(m4, I["ret5"])
    for r in (5, 10):
        add("T4.테마로테이션", E.Spec(f"테마로테이션 r{r}", s4, "rotate", r=r))

    # T6 과열주 눌림 매수
    m6 = ok & (I["ret20"] >= 0.30) & (vrank <= 20) & (ret1 <= -0.05)
    s6 = sc(m6, -ret1)
    for tp in (0.07, 0.12):
        add("T6.과열주눌림", E.Spec(f"과열주눌림 익절{tp * 100:.0f} −7% 3일", s6, "signal", hold=3, stop=0.07, tp=tp))
    return specs


def main() -> int:
    t0 = time.time()
    m = E.load()
    specs = build(m)
    print("specs", len(specs), round(time.time() - t0), flush=True)
    S = {n: starts(m, *p) for n, p in (("IS", IS), ("VAL", VAL), ("TEST", TEST))}
    L = ["# 테마주(급등주) 전략 탐색 — 한 달 대회 기준", "",
         "KRX 전종목(상장폐지 포함), 가격제한 ±30% 이후. 60만원·정수 주식·매도세 0.20%·틱 슬리피지·장중 손절 추가 0.2%.",
         "같은 날 손절선과 익절선에 둘 다 닿으면 손절로 본다 (보수적).",
         f"IS {len(S['IS'])}개 구간(2015.7 ~ 2020) → VAL {len(S['VAL'])}개(2021 ~ 2023) → TEST {len(S['TEST'])}개(2024 ~). 설정 {len(specs)}개 사전 고정.", ""]
    res = [(sp, E.month_dist(m, sp, S["IS"], 2)) for sp in specs]
    print("IS", round(time.time() - t0), flush=True)
    L += ["## ① IS 전 설정 (2슬롯)", ""] + HDR + [market_row(m, S["IS"], "IS")]
    for sp, d in sorted(res, key=lambda x: -x[1]["mean"]):
        L.append(row(sp, 2, "IS", d))
    best = {}
    for sp, d in res:
        f = sp.params["fam"]
        if d["p10d"] <= 0.35 and (f not in best or d["mean"] > best[f][1]["mean"]):
            best[f] = (sp, d)
    L += ["", "## ② 계열별 IS 최선 → VAL", ""] + HDR + [market_row(m, S["VAL"], "VAL")]
    val = []
    for f, (sp, _) in sorted(best.items()):
        for k in (1, 2, 3):
            d = E.month_dist(m, sp, S["VAL"], k)
            L.append(row(sp, k, "VAL", d))
            if k == 2:
                val.append((sp, d))
    top3 = sorted(val, key=lambda x: -x[1]["mean"])[:3]
    L += ["", "## ③ VAL 상위 3 → TEST (한 번)", ""] + HDR + [market_row(m, S["TEST"], "TEST")]
    for sp, _ in top3:
        for k in (1, 2, 3):
            L.append(row(sp, k, "TEST", E.month_dist(m, sp, S["TEST"], k)))
    # ④ VAL 1위 연도별
    sp0 = top3[0][0]
    allst = starts(m, "20150701", "99999999", 2)
    d = E.month_dist(m, sp0, allst, 2)
    mk = m.ind["mkt"]
    L += ["", f"## ④ 연도별 — {sp0.name} (2슬롯)", "", "| 연도 | 전략 | 시장 | +20%↑ | −10%↓ |", "|---|---|---|---|---|"]
    for y in sorted({m.dates[s][:4] for s in allst}):
        idx = [n for n, s in enumerate(allst) if m.dates[s][:4] == y]
        a = d["rets"][idx]
        mm = np.array([mk[allst[n] + 20] / mk[allst[n] - 1] - 1 for n in idx])
        L.append(f"| {y} | {fmt(a.mean())} | {fmt(mm.mean())} | {(a >= .2).mean() * 100:.0f}% | {(a <= -.1).mean() * 100:.0f}% |")
    p = os.path.join(OUT, "lab_theme.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p, round(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
