"""
backtest/research2.py — 승률과 수익률 절충 전략 탐색 (2차)

  python -m backtest.research2     → backtest/results/research2_<날짜>.md

1차(research.py)와 같은 데이터·구간·비용·체결 엔진을 쓴다. 후보는 공개 자료에서 검증 이력이 있는 단기 전략들.
  A. Connors RSI(2)     장기선 위 + RSI(2) 과매도 → RSI(2) > 65 또는 5일선 회복에 청산 (Connors 2008)
  B. IBS 반전           종가가 당일 범위 하단(IBS 낮음) + 하락 마감 + 장기선 위 → 전일 고가 위 마감에 청산
  C. 변동성 돌파        당일 시가 + k × 전일 범위를 넘으면 장중 매수 → 다음 날 시가 매도 (Larry Williams)
  D. 주도주 눌림        60일 수익률 상위 20위 안 종목이 RSI(2) 로 밀리면 매수 (모멘텀 + 반전 절충)

선택 규칙 (실행 전에 고정): IS 에서 승률 55% 이상·거래 30건 이상인 설정 중 5슬롯 샤프 최고를 계열별로 고른다.
그 설정만 OOS 에 한 번 돌린다.

체결 추가 가정
  종가 매수(B 의 close 모드): 15:20 직전 현재가로 신호를 판정하고 장마감 동시호가에 산다. 판정가 ≈ 종가로 본다 (약간 낙관).
  장중 매수(C): 목표가 도달 시 매수. 시가가 이미 목표가 위면 시가에 산다. 슬리피지 편도 0.2%.
"""
from __future__ import annotations

import math
import os
import sys
from statistics import fmean, median

from backtest.research import (COST, IS_END, IS_START, OOS_START, OUT, Momentum, Pos, Strategy, Trade,
                               bench, calendar, load, make_ctx, pct, stats)


# ── 지표 보강 ─────────────────────────────────────────────────
def enrich(data: dict) -> None:
    for s in data.values():
        n = len(s.c)
        s.ind["ibs"] = [(s.c[i] - s.l[i]) / (s.h[i] - s.l[i]) if s.h[i] > s.l[i] else 0.5 for i in range(n)]
        noise = [1 - abs(s.o[i] - s.c[i]) / (s.h[i] - s.l[i]) if s.h[i] > s.l[i] else 1.0 for i in range(n)]
        s.ind["noise20"] = [math.nan if i < 20 else fmean(noise[i - 20:i]) for i in range(n)]   # 전일까지 20일


def momentum_rank(ctx: dict, members: set, d: str, top: int) -> set:
    cache = ctx.setdefault("_mrank", {})
    key = (d, top)
    if key not in cache:
        out = []
        for t in members:
            s = ctx["data"].get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125:
                continue
            m = s.ind["mom60"][i]
            if m == m and s.c[i] > s.ind["ma60"][i]:
                out.append((t, m))
        cache[key] = {t for t, _ in sorted(out, key=lambda x: -x[1])[:top]}
    return cache[key]


# ── 전략 ─────────────────────────────────────────────────────
class RSI2(Strategy):
    name = "A.ConnorsRSI2"

    def candidates(self, d):
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i):
                continue
            if s.c[i] > s.ind["ma120"][i] and s.ind["rsi2"][i] < self.p["rsi"]:
                out.append((t, -s.ind["rsi2"][i]))
        return sorted(out, key=lambda x: -x[1])

    def check_exit(self, pos, s, i):
        if pos.days >= 1:
            if self.p["exit"] == "rsi65" and s.ind["rsi2"][i] > 65:
                return ("close", "RSI2>65")
            if self.p["exit"] == "ma5" and s.c[i] > s.ind["ma5"][i]:
                return ("close", "5일선 회복")
        if pos.days >= 10:
            return ("close", "만료")
        return None


class IBS(Strategy):
    name = "B.IBS반전"

    def __init__(self, ctx, members, **p):
        super().__init__(ctx, members, **p)
        self.entry_at_close = p["entry"] == "close"

    def candidates(self, d):
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i):
                continue
            if s.c[i] > s.ind["ma120"][i] and s.c[i] < s.c[i - 1] and s.ind["ibs"][i] < self.p["ibs"]:
                out.append((t, -s.ind["ibs"][i]))
        return sorted(out, key=lambda x: -x[1])

    def check_exit(self, pos, s, i):
        if s.d[i] == pos.entry_d:
            return None
        if s.c[i] > s.h[i - 1]:
            return ("close", "전일 고가 돌파")
        if pos.days >= 5:
            return ("close", "만료")
        return None


class VolBreakout(Strategy):
    name = "C.변동성돌파"
    intraday = True

    def intraday_entries(self, d):
        """전일까지 정보로 목표가를 정하고, 당일 고가가 닿은 종목을 잡음(노이즈) 낮은 순으로."""
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i - 1):
                continue
            rng = s.h[i - 1] - s.l[i - 1]
            if rng <= 0:
                continue
            target = s.o[i] + self.p["k"] * rng
            if self.p["ma_filter"] and not s.o[i] > s.ind["ma5"][i - 1]:
                continue
            if s.h[i] < target:
                continue
            nz = s.ind["noise20"][i]
            out.append((t, max(target, s.o[i]), -(nz if nz == nz else 1)))
        return sorted(out, key=lambda x: -x[2])

    def check_exit(self, pos, s, i):
        return ("next_open", "익일 시가")


class Pullback(Strategy):
    name = "D.주도주눌림"

    def candidates(self, d):
        top = momentum_rank(self.ctx, self.members, d, 20)
        out = []
        for t in top:
            s = self.data[t]
            i = s.idx.get(d)
            if i is None or not self._liquid(s, i):
                continue
            if s.ind["rsi2"][i] < self.p["rsi"]:
                out.append((t, -s.ind["rsi2"][i]))
        return sorted(out, key=lambda x: -x[1])

    def check_exit(self, pos, s, i):
        if pos.days >= 1:
            if self.p["exit"] == "rsi65" and s.ind["rsi2"][i] > 65:
                return ("close", "RSI2>65")
            if self.p["exit"] == "ma5" and s.c[i] > s.ind["ma5"][i]:
                return ("close", "5일선 회복")
        if pos.days >= 10:
            return ("close", "만료")
        return None


# ── 엔진 (research.run 확장: 종가 매수·장중 매수) ─────────────
def run(strat: Strategy, cal: list, start: str, end: str, k: int, cost: dict = COST) -> dict:
    data = strat.data
    if isinstance(strat, Momentum):
        strat.k = k
    at_close = getattr(strat, "entry_at_close", False)
    intraday = getattr(strat, "intraday", False)
    days = [d for d in cal if start <= d <= end]
    cash, pos, pending, pend_exit = 1.0, {}, [], {}
    trades, curve = [], []

    def buy(t, px, slip, d, i):
        nonlocal cash
        alloc = min(cash, last_eq / k)
        if alloc <= 1e-9:
            return
        qty = alloc / (px * (1 + slip) * (1 + cost["fee"]))
        p = Pos(t, d, px, alloc, qty)
        strat.init_pos(p, data[t], i)
        pos[t] = p
        cash -= alloc

    def sell(p, px, slip, d, reason):
        net_px = px * (1 - slip) * (1 - cost["fee"] - cost["tax"])
        trades.append(Trade(p.t, p.entry_d, d, p.qty * net_px / p.cost - 1, p.days, reason))
        return p.qty * net_px

    last_eq = 1.0
    for d in days:
        strat.begin_day(d)
        for t, why in list(pend_exit.items()):
            i = data[t].idx.get(d)
            if i is None:
                continue
            cash += sell(pos.pop(t), data[t].o[i], cost["slip_auction"], d, why)
            del pend_exit[t]
        for t, _ in pending:
            if len(pos) >= k or t in pos:
                continue
            i = data[t].idx.get(d)
            if i is not None:
                buy(t, data[t].o[i], cost["slip_auction"], d, i)
        pending = []
        if intraday:
            for t, px, _ in strat.intraday_entries(d):
                if len(pos) >= k:
                    break
                if t not in pos:
                    buy(t, px, cost["slip_stop"], d, data[t].idx[d])
        cands = strat.candidates(d)
        for t in list(pos):
            p, s = pos[t], data[t]
            i = s.idx.get(d)
            if i is None or t in pend_exit:
                continue
            p.days += 1
            ex = strat.check_exit(p, s, i)
            if ex is None:
                continue
            if ex[0] == "stop":
                cash += sell(pos.pop(t), ex[1], cost["slip_stop"], d, ex[2])
            elif ex[0] == "close":
                cash += sell(pos.pop(t), s.c[i], cost["slip_auction"], d, ex[1])
            else:
                pend_exit[t] = ex[1]
        if at_close:
            eq_now = cash + sum(p.qty * data[t].c[data[t].idx[d]] if d in data[t].idx else p.cost for t, p in pos.items())
            last_eq = eq_now
            for t, _ in cands:
                if len(pos) >= k:
                    break
                if t not in pos:
                    buy(t, data[t].c[data[t].idx[d]], cost["slip_auction"], d, data[t].idx[d])
        eq = cash
        for t, p in pos.items():
            s = data[t]
            i = s.idx.get(d)
            px = s.c[i] if i is not None else s.c[max(j for j, x in enumerate(s.d) if x <= d)]
            eq += p.qty * px
        curve.append((d, eq))
        last_eq = eq
        if not at_close and not intraday:
            free = k - len(pos) + len(pend_exit)
            if free > 0 and cands:
                pending = [(t, sc) for t, sc in cands if t not in pos][:free]
    for t, p in pos.items():
        s = data[t]
        i = max(j for j, x in enumerate(s.d) if x <= days[-1])
        trades.append(Trade(t, p.entry_d, s.d[i], p.qty * s.c[i] * (1 - cost["fee"] - cost["tax"]) / p.cost - 1,
                            p.days, "기간 끝"))
    return {"trades": trades, "curve": curve}


def stats2(r: dict) -> dict:
    m = stats(r)
    nets = [t.net for t in r["trades"]]
    w = [x for x in nets if x > 0]
    lo = [x for x in nets if x <= 0]
    m.update(avg_w=fmean(w) if w else 0, avg_l=fmean(lo) if lo else 0,
             pf=sum(w) / -sum(lo) if lo and sum(lo) < 0 else float("inf"))
    return m


def grid() -> list:
    """실행 전 고정."""
    g = []
    for rsi in (5, 10):
        for ex in ("rsi65", "ma5"):
            g.append((RSI2, {"rsi": rsi, "exit": ex}))
    for ibs in (0.1, 0.2):
        for en in ("close", "next_open"):
            g.append((IBS, {"ibs": ibs, "entry": en}))
    for kk in (0.4, 0.6):
        for mf in (False, True):
            g.append((VolBreakout, {"k": kk, "ma_filter": mf}))
    for rsi in (10, 25):
        for ex in ("rsi65", "ma5"):
            g.append((Pullback, {"rsi": rsi, "exit": ex}))
    return g


def row(lab, k, tag, m):
    return (f"| {lab} | {k} | {tag} | {m['n']} | **{m['win'] * 100:.0f}%** | {pct(m['avg_w'], 2)} / {pct(m['avg_l'], 2)} | "
            f"{pct(m['exp'], 2)} | {m['pf']:.2f} | {pct(m['total'])} | {pct(m['cagr'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} | {m['hold']:.1f} |")


HDR = ["| 설정 | 슬롯 | 구간 | 거래 | 승률 | 평균 이익 / 손실 | 거래당 | PF | 누적 | CAGR | MDD | 샤프 | 보유일 |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    data, meta = load()
    enrich(data)
    cal = calendar(data)
    end = cal[-1]
    U = meta["univ"]
    U0 = "KOSPI100"
    ctx = make_ctx(data, U[U0], cal)

    L = ["# 승률·수익률 절충 전략 탐색 (2차)", "",
         f"- IS {IS_START} ~ {IS_END} / OOS {OOS_START} ~ {end}, KOSPI100, 비용·체결은 1차와 같음",
         "- 선택 규칙(사전 고정): IS 승률 ≥ 55% · 거래 ≥ 30 인 설정 중 5슬롯 샤프 최고를 계열별로", ""]

    L += ["## ① IS 격자 (5슬롯)", ""] + HDR
    rows = []
    for cls, p in grid():
        st = cls(ctx, U[U0], **p)
        m = stats2(run(st, cal, IS_START, IS_END, 5))
        rows.append((cls, p, st.label(), m))
        L.append(row(st.label(), 5, "IS", m))
        print(f"IS {st.label():60s} n={m['n']:4d} win={m['win'] * 100:.0f}% exp={pct(m['exp'], 2)} sh={m['sharpe']:.2f} cagr={pct(m['cagr'])} mdd={pct(m['mdd'])}", flush=True)

    best = {}
    for cls, p, lab, m in rows:
        if m["n"] < 30 or m["win"] < 0.55:
            continue
        if cls not in best or m["sharpe"] > best[cls][2]["sharpe"]:
            best[cls] = (p, lab, m)
    missing = [c.name for c in (RSI2, IBS, VolBreakout, Pullback) if c not in best]

    b_is, b_oos = bench(data, U[U0], IS_START, IS_END), bench(data, U[U0], OOS_START, end)
    L += ["", "## ② 계열별 선택 설정 → OOS", "",
          f"동일가중 보유 — IS 누적 {pct(b_is['total'])} (MDD {pct(b_is['mdd'])}), OOS 누적 {pct(b_oos['total'])} (MDD {pct(b_oos['mdd'])})",
          f"승률 55% 조건을 통과한 설정이 없는 계열: {', '.join(missing) or '없음'}", ""] + HDR
    for cls, (p, lab, _) in best.items():
        for k in (1, 5):
            for tag, a, z in (("IS", IS_START, IS_END), ("OOS", OOS_START, end)):
                m = stats2(run(cls(ctx, U[U0], **p), cal, a, z, k))
                L.append(row(lab, k, tag, m))
                print(tag, k, lab, f"win={m['win'] * 100:.0f}% exp={pct(m['exp'], 2)} total={pct(m['total'])} mdd={pct(m['mdd'])}", flush=True)
    # 참고: 1차 모멘텀 로테이션
    for k in (1, 5):
        for tag, a, z in (("IS", IS_START, IS_END), ("OOS", OOS_START, end)):
            m = stats2(run(Momentum(ctx, U[U0], L=60, r=20, regime=False), cal, a, z, k))
            L.append(row("(참고) 1차 모멘텀로테이션", k, tag, m))

    # ③ 강건성: 비용↑, 유니버스 확대, 반기별, 1슬롯 시작일 위상
    cost2 = dict(COST, slip_auction=COST["slip_auction"] * 4, slip_stop=COST["slip_stop"] * 2)
    L += ["", "## ③ 강건성 (선택 뒤, 설정 변경 없음 — 전체 구간 " + IS_START + " ~ " + end + ", 5슬롯)", "",
          "| 설정 | 조건 | 거래 | 승률 | 거래당 | 누적 | MDD | 샤프 |", "|---|---|---|---|---|---|---|---|"]
    for cls, (p, lab, _) in best.items():
        for cname, u, c in (("기본", U0, COST), ("슬리피지 4배", U0, cost2), ("KOSPI200", "KOSPI200", COST)):
            cx = ctx if u == U0 else make_ctx(data, U[u], cal)
            m = stats2(run(cls(cx, U[u], **p), cal, IS_START, end, 5, c))
            L.append(f"| {lab} | {cname} | {m['n']} | {m['win'] * 100:.0f}% | {pct(m['exp'], 2)} | {pct(m['total'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} |")

    halves = []
    for y in range(int(IS_START[:4]), int(end[:4]) + 1):
        for a, z in ((f"{y}0101", f"{y}0630"), (f"{y}0701", f"{y}1231")):
            a, z = max(a, IS_START), min(z, end)
            if a < z and any(a <= d <= z for d in cal):
                halves.append((a, z))
    L += ["", "### 반기별 누적 (5슬롯)", "", "| 반기 | 동일가중 보유 | " + " | ".join(lab for _, lab, _ in best.values()) + " |",
          "|---|---|" + "---|" * len(best)]
    for a, z in halves:
        cells = [pct(bench(data, U[U0], a, z)["total"])]
        for cls, (p, lab, _) in best.items():
            m = stats2(run(cls(ctx, U[U0], **p), cal, a, z, 5))
            cells.append(f"{pct(m['total'])} ({m['win'] * 100:.0f}%)")
        L.append(f"| {a[:6]}~{z[:6]} | " + " | ".join(cells) + " |")
    L.append("\n괄호는 그 반기 승률.")

    L += ["", "### 1슬롯 시작일 위상 20가지 (전체 구간) — 1종목 집중이 얼마나 운에 좌우되나", "",
          "| 설정 | 누적 최소 / 중앙 / 최대 | 승률 중앙 | MDD 중앙 |", "|---|---|---|---|"]
    days = [d for d in cal if IS_START <= d <= end]
    for cls, (p, lab, _) in best.items():
        ms = [stats2(run(cls(ctx, U[U0], **p), cal, days[off], end, 1)) for off in range(20)]
        tot = sorted(m["total"] for m in ms)
        L.append(f"| {lab} | {pct(tot[0])} / {pct(median(tot))} / {pct(tot[-1])} | {median(m['win'] for m in ms) * 100:.0f}% | {pct(median(m['mdd'] for m in ms))} |")

    # ④ 소액 계좌: 슬롯 수와 주가 상한 (60만원 기준 슬롯당 금액 이하 종목만)
    from backtest.research import MaxPrice
    L += ["", "### 소액 계좌 (60만원) — 슬롯 수 × 주가 상한", "",
          "| 전략 | 슬롯 · 주가 상한 | 구간 | 거래 | 승률 | 거래당 | 누적 | MDD |", "|---|---|---|---|---|---|---|---|"]
    for k, cap in ((5, 120_000), (3, 200_000), (2, 300_000)):
        for tag, a, z in (("IS", IS_START, IS_END), ("OOS", OOS_START, end), ("전체", IS_START, end)):
            for lab, mk in (("D.주도주눌림", lambda: Pullback(ctx, U[U0], rsi=25, exit="ma5")),
                            ("모멘텀로테이션", lambda: Momentum(ctx, U[U0], L=60, r=20, regime=False))):
                st = MaxPrice(mk(), cap)
                st.inner.k = k
                m = stats2(run(st, cal, a, z, k))
                L.append(f"| {lab} | {k}슬롯 · {cap // 10000}만원 이하 | {tag} | {m['n']} | {m['win'] * 100:.0f}% | {pct(m['exp'], 2)} | {pct(m['total'])} | {pct(m['mdd'])} |")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"research2_{end}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
