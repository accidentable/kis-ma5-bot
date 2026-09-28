"""
backtest/lab/panic.py — 시장 패닉일 칼날 잡기 (평소엔 현재 봇 전략, 패닉일엔 급락 관심주로 갈아탄다)

  python -m backtest.lab.panic    → backtest/results/lab_panic.md

knife.py 조사에서: 개별 종목 급락 매수는 대회 시뮬레이션에서 전부 손실, 그런데 이익이 난 사건은
시장 전체가 무너진 날(2020-03, 2024-08-05, 2026-03-04 …)에 몰려 있었다. 그래서 '시장 패닉일에만' 산다.

패닉일: 거래대금 30억↑ 종목 동일가중 등락률(당일)이 −P% 이하.
패닉 매수 후보 (당일 종가 기준, 다음 날 시가에 산다; 주가 30만원 이하)
  K1 관심주: 최근 20일 중 시장 거래대금 50위 안에 든 적 있고, 당일 −10% 이하 → 낙폭 큰 순
  K2 대형주: 시총 200 안에서 당일 낙폭 큰 순
패닉 포지션은 H 거래일 들고 판 뒤, 평소 전략(신고가 근접 + 급등 필터 + 급등 매도)으로 돌아간다.
패닉일이 오면 평소 보유를 다음 날 시가에 팔고 패닉 후보로 바꾼다 (이미 패닉 포지션이면 유지).

격자 (사전 고정 8개): P {3, 4} × 후보 {K1, K2} × H {5, 21}
검증: IS 2011 ~ 2018 에서 1위 선택 → VAL 2019 ~ 2024.9 → TEST 2024.10 ~, 60만원 · 한 달 · 2종목.
비교: 평소 전략만 (현재 봇).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from backtest.lab import engine as E
from backtest.lab.search import fmt, starts

warnings.filterwarnings("ignore")
OUT = os.path.join(E.ROOT, "backtest", "results")


def run_hybrid(m, base: E.Spec, panic_top: list, panic_day: np.ndarray, hold_p: int, s: int, k: int,
               surge_exit: np.ndarray, capital: float = E.CAPITAL):
    """평소: base(rotate r=21) + 급등 매도 · 빈 슬롯 다음 날 채움. 패닉일 다음 날 시가: 전량 교체 → hold_p 일 뒤 복귀."""
    e = min(s + E.MONTH - 1, m.T - 1)
    cash = float(capital)
    pos: dict = {}           # j -> [qty, entry_adj, cost, t0, kind]
    slot = capital / k
    mode_until = -1          # 패닉 모드 종료일 (이 날 종가에 판다)
    trades = 0
    next_base_rebal = s

    def raw_at(t, j, px_adj):
        return px_adj * m.raw[t, j] / m.c[t, j] if m.c[t, j] > 0 else px_adj

    def sell(j, t, px_adj):
        nonlocal cash
        q, ea, cost, t0, kind = pos.pop(j)
        net = px_adj * (1 - E.slip(raw_at(t, j, px_adj))) * (1 - E.FEE - E.TAX)
        cash += cost * net / ea

    def buy(j, t):
        nonlocal cash, trades
        o, cp = m.o[t, j], m.c[t - 1, j]
        if j in pos or len(pos) >= k or not (o > 0) or not (cp > 0) or o >= cp * 1.295:
            return
        ro = raw_at(t, j, o)
        px = ro * (1 + E.slip(ro))
        q = int(min(cash, slot) / (px * (1 + E.FEE)))
        if q < 1:
            return
        cost = q * px * (1 + E.FEE)
        cash -= cost
        pos[j] = [q, o * (1 + E.slip(ro)) * (1 + E.FEE), cost, t, "?"]
        trades += 1

    for t in range(s, e + 1):
        sig = t - 1
        in_panic = t <= mode_until
        # 패닉 진입: 전날이 패닉일이면 오늘 시가에 전량 교체
        if panic_day[sig] and len(panic_top[sig]) and not in_panic:
            for j in list(pos):
                if m.o[t, j] > 0:
                    sell(j, t, float(m.o[t, j]))
            for j in panic_top[sig]:
                if len(pos) >= k:
                    break
                buy(int(j), t)
            mode_until = t + hold_p - 1
            in_panic = True
        elif not in_panic:
            # 평소 전략: 21일 교체 주기 + 빈 슬롯 채우기
            if t >= next_base_rebal:
                keep = set(base.top[sig][:k].tolist())
                for j in list(pos):
                    if j not in keep and m.o[t, j] > 0:
                        sell(j, t, float(m.o[t, j]))
                next_base_rebal = t + 21
            for j in base.top[sig]:
                if len(pos) >= k:
                    break
                buy(int(j), t)
        # 종가: 패닉 모드 끝나는 날 전량 매도 / 평소엔 급등 매도
        if in_panic and t == mode_until:
            for j in list(pos):
                if m.c[t, j] > 0:
                    sell(j, t, float(m.c[t, j]))
            next_base_rebal = t + 1
        elif not in_panic:
            for j in list(pos):
                if surge_exit[t, j] and m.c[t, j] > 0 and pos[j][3] < t:
                    sell(j, t, float(m.c[t, j]))
    for j in list(pos):
        c = m.c[e, j]
        if not (c > 0):
            back = np.where(m.c[:e + 1, j] > 0)[0]
            c = m.c[back[-1], j] if len(back) else pos[j][1]
        sell(j, e, float(c))
    return cash / capital - 1, trades


def dist(rets):
    a = np.array(rets)
    return {"mean": a.mean(), "median": np.median(a), "p10d": (a <= -.1).mean(), "p10u": (a >= .1).mean()}


def main() -> int:
    m = E.load()
    I = m.ind
    c = m.c.astype(np.float64)
    val = m.val.astype(np.float64)
    ok = (~np.isnan(c)) & (m.raw >= 1000) & (val > 0) & (I["val20"] >= 1e9) & (m.raw <= 300_000)
    base_mask = ok & (I["caprank"] <= 200) & (I["ret60"] > 0) & (I["max20"] < 0.10) & np.isfinite(I["hi250"])
    base = E.Spec("base", np.where(base_mask, c / I["hi250"], np.nan), "rotate", r=21).prepare()
    surge_exit = I["ret1"] >= 0.10
    vrank = (-np.nan_to_num(val, nan=-1)).argsort(axis=1).argsort(axis=1) + 1
    top50 = E._shift((E._roll_max((vrank <= 50).astype(np.float64), 20) > 0).astype(np.float64), 1) > 0
    K = {"K1 관심주 −10%↓": ok & top50 & (I["ret1"] <= -0.10), "K2 시총200 낙폭순": ok & (I["caprank"] <= 200)}
    tops = {}
    for kn, msk in K.items():
        sp = E.Spec(kn, np.where(msk, -I["ret1"], np.nan), "signal").prepare()
        tops[kn] = sp.top
    mr = I["mkt_ret1"]
    P = {"2011~2018 (IS)": starts(m, "20110101", "20181231", 2), "2019~2024.9 (VAL)": starts(m, "20190101", "20240930", 2),
         "2024.10~ (TEST)": starts(m, "20241001", "99999999", 1), "최근 12개월": starts(m, "20250901", "99999999", 1)}
    never = np.zeros(m.T, bool)
    rows = []
    res_base = {pn: dist([run_hybrid(m, base, [np.array([], int)] * m.T, never, 5, s, 2, surge_exit)[0] for s in ss])
                for pn, ss in P.items()}
    configs = []
    for thr in (0.03, 0.04):
        pday = mr <= -thr
        for kn in K:
            for h in (5, 21):
                name = f"패닉 {thr * 100:.0f}%↓ · {kn} · {h}일"
                r = {pn: dist([run_hybrid(m, base, tops[kn], pday, h, s, 2, surge_exit)[0] for s in ss]) for pn, ss in P.items()}
                configs.append((name, r, pday))
                print(name, {k: f"{v['mean'] * 100:+.2f}%" for k, v in r.items()}, flush=True)
    best = max(configs, key=lambda x: x[1]["2011~2018 (IS)"]["mean"])
    L = [(__doc__ or "").strip(), "", "## 한 달 평균 (중앙값, −10%↓, +10%↑) — 2종목, 비용 후", "",
         "| 전략 | " + " | ".join(P) + " |", "|---|" + "---|" * len(P)]

    def cell(d):
        return f"{fmt(d['mean'])} ({fmt(d['median'])}, {d['p10d'] * 100:.0f}%, {d['p10u'] * 100:.0f}%)"
    L.append("| 평소 전략만 (현재 봇) | " + " | ".join(cell(res_base[p]) for p in P) + " |")
    for name, r, _ in sorted(configs, key=lambda x: -x[1]["2011~2018 (IS)"]["mean"]):
        L.append(f"| {name}{' ← IS 1위' if name == best[0] else ''} | " + " | ".join(cell(r[p]) for p in P) + " |")
    # 패닉일이 든 달만 따로 (IS 1위 설정)
    pday = best[2]
    L += ["", f"## 패닉일이 포함된 달만 — {best[0]}", "", "| 구간 | 해당 달 수 | 평소 전략 | 패닉 모드 |", "|---|---|---|---|"]
    kn = next(k for k in K if k in best[0])
    h = int(best[0].split("·")[-1].strip().rstrip("일"))
    for pn, ss in P.items():
        sel = [s for s in ss if pday[s - 1:s + E.MONTH - 1].any()]
        if not sel:
            continue
        a = [run_hybrid(m, base, [np.array([], int)] * m.T, never, 5, s, 2, surge_exit)[0] for s in sel]
        b = [run_hybrid(m, base, tops[kn], pday, h, s, 2, surge_exit)[0] for s in sel]
        L.append(f"| {pn} | {len(sel)} | {fmt(np.mean(a))} | {fmt(np.mean(b))} |")
    L += ["", f"패닉일(시장 −{best[0].split()[1]}) 수: " + ", ".join(
        f"{y}년 {int(sum(pday[t] for t in range(m.T) if m.dates[t][:4] == y))}" for y in sorted({d[:4] for d in m.dates}))]
    p = os.path.join(OUT, "lab_panic.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("리포트:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
