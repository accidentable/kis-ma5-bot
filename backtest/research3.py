"""
backtest/research3.py — "딱 한 달만 매매한다면" (투자대회 기간 가정)

  python -m backtest.research3     → backtest/results/research3_<날짜>.md

3년치 데이터에서 가능한 모든 한 달(21거래일) 구간마다 빈 계좌로 시작해 매매하고,
마지막 날 종가에 전부 판다고 보고 그 달의 수익률을 모은다. 전략별로 "한 달 수익률의 분포"를 본다.

  - 시작일을 하루씩 밀어가며 전부 돌린다 (겹치는 구간이라 서로 독립이 아니다. 독립 표본은 대략 36개월치).
  - 60만원 계좌 제약: 슬롯당 금액보다 비싼 종목은 못 산다 (1슬롯 60만, 2슬롯 30만, 3슬롯 20만원 이하).
  - 설정은 1·2차에서 고른 그대로. 여기서 새로 고르지 않는다.
  - 현재 실전 규칙(MA5 돌파)은 실전 봇 판정 함수를 쓰는 engine.py 로 같은 방식(한 달·빈 계좌 시작)으로 돌린다.
"""
from __future__ import annotations

import copy
import os
import sys
from datetime import date
from statistics import fmean, median

from backtest import research2 as r2
from backtest.research import (COST, IS_START, OUT, MaxPrice, Momentum, calendar, load, make_ctx, pct)

MONTH = 21   # 거래일


def summarize(rets: list, extra: dict | None = None) -> dict:
    s = sorted(rets)
    n = len(s)
    q = lambda p: s[min(n - 1, max(0, int(p * (n - 1))))]
    out = {"n": n, "mean": fmean(s), "median": median(s), "p_pos": sum(x > 0 for x in s) / n,
           "p10u": sum(x >= 0.10 for x in s) / n, "p20u": sum(x >= 0.20 for x in s) / n,
           "p10d": sum(x <= -0.10 for x in s) / n, "q10": q(0.10), "q90": q(0.90), "min": s[0], "max": s[-1]}
    out.update(extra or {})
    return out


def month_runs(make, cal: list, starts: list, k: int, cap: float) -> dict:
    rets, ntr, wins = [], [], []
    for a in starts:
        z = cal[cal.index(a) + MONTH - 1]
        st = MaxPrice(make(), cap)
        st.inner.k = k
        if isinstance(st.inner, Momentum):
            st.inner.n_day = 0
        r = r2.run(st, cal, a, z, k)
        rets.append(r["liq"] - 1)
        closed = [t for t in r["trades"] if t.reason != "기간 끝"]
        ntr.append(len(r["trades"]))
        wins.extend(t.net > 0 for t in closed)
    return summarize(rets, {"trades": fmean(ntr), "win": fmean(wins) if wins else float("nan")})


def live_rule_months(cal_starts: list, cal: list) -> dict | None:
    """현재 실전 규칙을 한 달씩 빈 계좌로. 실전 봇 코드(core.strategy)를 쓴다."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    config.CROSS_LOOKBACK = 1
    from backtest import data as bdata, engine, universes
    from backtest.run import CACHE, UNIV_CACHE, restore_snapshot_if_needed
    from core import strategy

    restore_snapshot_if_needed()
    u = universes.build(UNIV_CACHE)
    names = {s["ticker"]: s["name"] for k in u for s in u[k]}
    members = {s["ticker"] for s in u["KOSPI100"]}
    bars = bdata.load_all(sorted(members), CACHE, date(2000, 1, 1), date.today())
    hits = engine.precompute_signals(bars, names, cal_starts[0])
    ranked = {}
    for d, lst in hits.items():
        mine = [(t, i, copy.copy(c)) for t, i, c in lst if t in members]
        order = strategy.rank([c for _, _, c in mine])
        pos = {id(c): (t, i) for t, i, c in mine}
        ranked[d] = [(pos[id(c)][0], pos[id(c)][1], c) for c in order]

    didx = {d: k for k, d in enumerate(cal)}
    rets, ntr, wins = [], [], []
    for a in cal_starts:
        k0 = didx[a]
        z = cal[k0 + MONTH - 1]
        eq, k, n = 1.0, k0, 0
        while k < len(cal) and cal[k] <= z:
            tr = None
            for t, i, c in ranked.get(cal[k], []):
                tr = engine.simulate_trade(bars[t], i, c, names.get(t, t))
                if tr:
                    break
            if tr is None:
                k += 1
                continue
            n += 1
            if tr.exit_date <= z:
                eq *= 1 + tr.net
                wins.append(tr.net > 0)
                k = didx.get(tr.exit_date, k) + 1
            else:
                # 한 달 끝날 때 아직 들고 있으면 그날 종가에 판다 (mtm 은 매도 비용 반영된 배수)
                mult = [m for dd, m in tr.mtm if dd <= z]
                eq *= mult[-1] if mult else 1.0
                break
        rets.append(eq - 1)
        ntr.append(n)
    return summarize(rets, {"trades": fmean(ntr), "win": fmean(wins) if wins else float("nan")})


def bh_months(data, members, cal, starts, cap) -> dict:
    """동일가중 보유 한 달 (주가 상한 없이 — 소액으로는 실제로 불가능한 기준선)."""
    from backtest.research import bench
    rets = [bench(data, members, a, cal[cal.index(a) + MONTH - 1])["total"] for a in starts]
    return summarize(rets, {"trades": 100, "win": float("nan")})


HDR = ["| 전략 | 슬롯 · 주가 상한 | 평균 | 중앙값 | 플러스 확률 | **+10% 이상** | +20% 이상 | **−10% 이하** | 하위 10% / 상위 10% | 최악 / 최고 | 월 거래 | 승률 |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]


def line(lab, cond, m):
    w = "-" if m["win"] != m["win"] else f"{m['win'] * 100:.0f}%"
    return (f"| {lab} | {cond} | {pct(m['mean'])} | {pct(m['median'])} | {m['p_pos'] * 100:.0f}% | **{m['p10u'] * 100:.0f}%** | "
            f"{m['p20u'] * 100:.0f}% | **{m['p10d'] * 100:.0f}%** | {pct(m['q10'])} / {pct(m['q90'])} | {pct(m['min'])} / {pct(m['max'])} | "
            f"{m['trades']:.1f} | {w} |")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    data, meta = load()
    r2.enrich(data)
    cal = calendar(data)
    end = cal[-1]
    U = meta["univ"]["KOSPI100"]
    ctx = make_ctx(data, U, cal)

    all_starts = [d for d in cal if d >= IS_START][: -(MONTH - 1)]
    recent = all_starts[-230:]    # 최근 1년 남짓 (대략 2025-09 이후 시작)

    strategies = [
        ("① 주도주 눌림", lambda: r2.Pullback(ctx, U, rsi=25, exit="ma5")),
        ("② 모멘텀 로테이션", lambda: Momentum(ctx, U, L=60, r=20, regime=False)),
        ("(탈락) Connors RSI2", lambda: r2.RSI2(ctx, U, rsi=5, exit="ma5")),
        ("(탈락) 변동성 돌파", lambda: r2.VolBreakout(ctx, U, k=0.6, ma_filter=True)),
    ]
    slots = [(1, 600_000), (2, 300_000), (3, 200_000)]

    L = ["# 한 달만 매매했을 때 (투자대회 가정)", "",
         f"- 가능한 모든 21거래일 구간을 빈 계좌로 시작 (시작일 {all_starts[0]} ~ {all_starts[-1]}, {len(all_starts)}개 — 겹치므로 독립 표본은 약 36개)",
         "- 마지막 날 종가에 전부 판다고 보고 비용까지 뺀 한 달 수익률. 60만원 계좌라 슬롯당 금액보다 비싼 종목은 제외",
         "- 설정은 1·2차에서 고른 그대로 (여기서 새로 고르지 않음)", ""]

    for tag, starts in (("전체 3년", all_starts), ("최근 1년 (시작일 " + recent[0] + " 이후)", recent)):
        L += [f"## {tag}", ""] + HDR
        b = bh_months(data, U, cal, starts, 0)
        L.append(line("동일가중 보유 (100종목, 소액 불가)", "-", b))
        print(tag, "보유", pct(b["mean"]), flush=True)
        lv = live_rule_months(starts, cal)
        L.append(line("현재 실전 규칙 (MA5 돌파)", "1 · 제한 없음", lv))
        print(tag, "실전", pct(lv["mean"]), flush=True)
        for lab, mk in strategies:
            for k, cap in slots:
                if lab.startswith("(탈락)") and k != 1:
                    continue
                m = month_runs(mk, cal, starts, k, cap)
                L.append(line(lab, f"{k} · {cap // 10000}만원 이하", m))
                print(tag, lab, k, pct(m["mean"]), f"p10u={m['p10u']:.2f}", flush=True)
        L.append("")

    L += ["## 읽는 법", "",
          "- **+10% 이상**: 한 달에 10% 넘게 번 달의 비율. 대회 상위권을 노린다면 이 칸이 중요하다.",
          "- **−10% 이하**: 한 달에 10% 넘게 잃은 달의 비율.",
          "- 월 거래·승률은 그 달 안에 끝난 거래 기준 (마지막 날 강제 청산은 승률에서 뺐다).",
          "- 구간이 하루씩 겹쳐서 표본 수가 부풀려 보인다. 실제로 서로 다른 달은 36개 정도라 확률은 ±10%p 안팎 오차로 읽어야 한다."]

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"research3_{end}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
