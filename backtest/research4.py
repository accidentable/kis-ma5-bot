"""
backtest/research4.py — 한 달 기준 전략 탐색 (투자대회용, 4차)

  python -m backtest.research4                                   3.4년 스냅샷
  BT_SNAPSHOT=backtest/snapshot5y.json.gz python -m backtest.research4   5년 스냅샷 (있으면 2021~2023 을 홀드아웃으로 본다)

목표: 60만원·한 달 동안 빈 계좌로 시작해 마지막 날 전부 판다고 볼 때 한 달 수익률의 평균을 키운다.
과적합을 막는 순서 (실행 전에 고정)
  1. 전략 계열 5개 · 설정 40개를 먼저 정한다 (FAMILIES). 결과를 보고 늘리지 않는다.
  2. IS(2023-10 ~ 2025-06 시작 구간)에서 계열별로 "한 달 평균 수익률" 최고 설정을 고른다.
     조건: 2슬롯 · 30만원 이하 종목 (60만원 계좌). 10% 넘게 잃는 달이 25% 를 넘으면 탈락.
  3. 고른 설정만 OOS(2025-07 이후 시작 구간)에서 본다.
  4. 5년 스냅샷이 있으면 2021-09 ~ 2023-09 시작 구간(2022 하락장 포함)을 홀드아웃으로 한 번 더 본다.
     이 구간은 선택에 전혀 안 쓴다.
"""
from __future__ import annotations

import math
import os
import sys
from statistics import fmean

from backtest import research as R
from backtest import research2 as r2
from backtest.research import (IS_END, IS_START, OOS_START, OUT, MaxPrice, Momentum, Strategy,
                               _ret, _sma, bench, calendar, load, make_ctx, pct)
from backtest.research3 import MONTH, summarize

HOLDOUT_END = "20230930"


# ── 지표 보강 ─────────────────────────────────────────────────
def enrich4(data: dict) -> None:
    r2.enrich(data)
    for s in data.values():
        c = s.c
        s.ind["mom20"] = _ret(c, 20)
        s.ind["ma10"] = _sma(c, 10)
        s.ind["hi120"] = [math.nan if i < 119 else max(s.h[i - 119:i + 1]) for i in range(len(c))]


# ── 전략 ─────────────────────────────────────────────────────
class Mom(Momentum):
    """모멘텀 로테이션. L 에 20 을 추가 (최근 5일 제외 없이 20일 수익률)."""
    name = "모멘텀"

    def _ranked(self, d):
        key = f"mom{self.p['L']}"
        cap = getattr(self, "cap", math.inf)
        memo = self.ctx.setdefault("_memo", {})
        mk = (self.name, self.p.get("L"), cap, d)
        if mk in memo:
            return memo[mk]
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i) or s.c[i] > cap:
                continue
            m = s.ind[key][i]
            if m == m and m > 0 and s.c[i] > s.ind["ma60"][i]:
                out.append((t, m))
        memo[mk] = sorted(out, key=lambda x: -x[1])
        return memo[mk]


class NearHigh(Mom):
    """신고가 근접 (George & Hwang 2004): 종가 / 최근 120일 최고가 가 1 에 가까운 순. 동점은 60일 수익률로."""
    name = "신고가근접"

    def _ranked(self, d):
        cap = getattr(self, "cap", math.inf)
        memo = self.ctx.setdefault("_memo", {})
        mk = (self.name, cap, d)
        if mk in memo:
            return memo[mk]
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i) or s.c[i] > cap:
                continue
            hi, m = s.ind["hi120"][i], s.ind["mom60"][i]
            if hi == hi and m == m and m > 0:
                out.append((t, s.c[i] / hi + 1e-3 * m))
        memo[mk] = sorted(out, key=lambda x: -x[1])
        return memo[mk]


class Surge(Strategy):
    """거래대금 급증 돌파: 종가가 20일 신고가 + 거래대금이 20일 평균의 m 배 이상 + 60일선 위 → 다음 날 시가 매수.
    종가가 5일선/10일선 아래로 마감하면 그날 종가 매도, 최대 15거래일."""
    name = "거래대금돌파"

    def candidates(self, d):
        memo = self.ctx.setdefault("_memo", {})
        mk = (self.name, self.p["mult"], d)
        if mk not in memo:
            memo[mk] = self._cands(d)
        return memo[mk]

    def _cands(self, d):
        out = []
        for t in self.members:
            s = self.data.get(t)
            i = s.idx.get(d) if s else None
            if i is None or i < 125 or not self._liquid(s, i):
                continue
            base = s.ind["val20"][i - 1]
            if not (base == base and base > 0):
                continue
            ratio = s.val[i] / base
            if s.c[i] > s.ind["hh20"][i] and ratio >= self.p["mult"] and s.c[i] > s.ind["ma60"][i]:
                out.append((t, ratio))
        return sorted(out, key=lambda x: -x[1])

    def check_exit(self, pos, s, i):
        if s.c[i] < s.ind[self.p["exit"]][i]:
            return ("close", f"{self.p['exit']} 이탈")
        if pos.days >= 15:
            return ("close", "만료")
        return None


class Pull(r2.Pullback):
    """주도주 눌림 (2차) — 주도주 범위 top 을 격자로."""
    name = "주도주눌림"

    def candidates(self, d):
        top = r2.momentum_rank(self.ctx, self.members, d, self.p["top"])
        out = []
        for t in sorted(top):
            s = self.data[t]
            i = s.idx.get(d)
            if i is None or not self._liquid(s, i):
                continue
            if s.ind["rsi2"][i] < 25:
                out.append((t, -s.ind["rsi2"][i]))
        return sorted(out, key=lambda x: -x[1])

    def check_exit(self, pos, s, i):
        if s.c[i] > s.ind["ma5"][i]:
            return ("close", "5일선 회복")
        if pos.days >= 10:
            return ("close", "만료")
        return None


UNIVS = ("KOSPI100", "KOSPI200+KOSDAQ150")
# 전 종목(ALL) 이 스냅샷에 있으면 같은 20개 설정을 ALL 에도 돌린다 (5년 데이터 받기 전에 정해 둔 규칙)


def families(univs=UNIVS) -> list:
    """사전 고정한 격자 (유니버스당 20개)."""
    g = []
    for u in univs:
        for L in (20, 60, 120):
            for r in (5, 20):
                for reg in (False, True):
                    g.append(("모멘텀", Mom, u, {"L": L, "r": r, "regime": reg}))
        for r in (5, 20):
            g.append(("신고가근접", NearHigh, u, {"r": r, "regime": False}))
        for mult in (2, 3):
            for ex in ("ma5", "ma10"):
                g.append(("거래대금돌파", Surge, u, {"mult": mult, "exit": ex}))
        for top in (10, 20):
            g.append(("주도주눌림", Pull, u, {"top": top, "rsi": 25, "exit": "ma5"}))
    return g


def month_dist(cls, ctx, members, p, cal, starts, k, cap) -> dict:
    rets, ntr = [], []
    for a in starts:
        z = cal[cal.index(a) + MONTH - 1]
        st = MaxPrice(cls(ctx, members, **p), cap)
        st.inner.k = k
        r = r2.run(st, cal, a, z, k)
        rets.append(r["liq"] - 1)
        ntr.append(len(r["trades"]))
    return summarize(rets, {"trades": fmean(ntr), "win": float("nan")})


def bench_dist(data, members, cal, starts) -> dict:
    return summarize([bench(data, members, a, cal[cal.index(a) + MONTH - 1])["total"] for a in starts],
                     {"trades": 0, "win": float("nan")})


HDR = ["| 전략 | 유니버스 | 슬롯 | 구간 | 평균 | 중앙값 | 플러스 | **+10%↑** | +20%↑ | **−10%↓** | 하위10% / 상위10% | 최악 / 최고 |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]


def line(lab, u, k, tag, m):
    return (f"| {lab} | {u} | {k} | {tag} | {pct(m['mean'])} | {pct(m['median'])} | {m['p_pos'] * 100:.0f}% | "
            f"**{m['p10u'] * 100:.0f}%** | {m['p20u'] * 100:.0f}% | **{m['p10d'] * 100:.0f}%** | "
            f"{pct(m['q10'])} / {pct(m['q90'])} | {pct(m['min'])} / {pct(m['max'])} |")


def label(cls, p):
    return cls.name + "(" + ", ".join(f"{k}={v}" for k, v in p.items() if k not in ("rsi", "exit") or cls is not Pull) + ")"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    data, meta = load()
    enrich4(data)
    cal = calendar(data)
    end = cal[-1]
    U = meta["univ"]
    univs = UNIVS + (("ALL",) if U.get("ALL") else ())
    ctxs = {u: make_ctx(data, U[u], cal) for u in univs}

    def starts_in(a, z, step):
        ss = [d for d in cal if a <= d <= z]
        ss = [d for d in ss if cal.index(d) + MONTH - 1 < len(cal)]
        return ss[::step]

    # 첫 신호는 125봉 이후라, 데이터 시작 + 130거래일부터 구간을 연다
    first_ok = cal[min(130, len(cal) - 1)]
    is_starts = starts_in(max(IS_START, first_ok), IS_END, 3)
    oos_starts = starts_in(OOS_START, end, 1)
    hold_starts = starts_in(first_ok, HOLDOUT_END, 1) if first_ok < "20230601" else []

    SLOT = (2, 300_000)
    L = ["# 한 달 기준 전략 탐색 (4차)", "",
         f"- 데이터 {cal[0]} ~ {end} ({len(data)}종목). 한 달 = {MONTH}거래일, 빈 계좌로 시작해 마지막 날 종가에 전부 판다",
         f"- 선택: IS 시작 구간 {is_starts[0]} ~ {is_starts[-1]} (3일 간격 {len(is_starts)}개)에서 계열별 한 달 평균 최고 (2슬롯·30만원 이하, −10%↓ 25% 초과 탈락)",
         f"- OOS: {oos_starts[0]} ~ {oos_starts[-1]} 시작 {len(oos_starts)}개" +
         (f" / 홀드아웃: {hold_starts[0]} ~ {hold_starts[-1]} 시작 {len(hold_starts)}개 (선택에 안 씀)" if hold_starts else " / 홀드아웃: 데이터 없음 (5년 스냅샷 필요)"),
         ""]

    # ① IS 격자
    rows = []
    for fam, cls, u, p in families(univs):
        m = month_dist(cls, ctxs[u], U[u], p, cal, is_starts, *SLOT)
        rows.append((fam, cls, u, p, m))
        print(f"IS {fam:6s} {u:18s} {p} mean={pct(m['mean'])} p10u={m['p10u']:.2f} p10d={m['p10d']:.2f}", flush=True)
    L += ["## ① IS 격자 (2슬롯 · 30만원 이하)", ""] + HDR
    L.append(line("동일가중 보유", "KOSPI100", "-", "IS", bench_dist(data, U["KOSPI100"], cal, is_starts)))
    for fam, cls, u, p, m in rows:
        L.append(line(label(cls, p), u, 2, "IS", m))

    best = {}
    for fam, cls, u, p, m in rows:
        if m["p10d"] > 0.25:
            continue
        if fam not in best or m["mean"] > best[fam][4]["mean"]:
            best[fam] = (fam, cls, u, p, m)

    # ② 선택 → OOS / 홀드아웃, 1·2·3슬롯
    L += ["", "## ② 계열별 IS 최선 → OOS · 홀드아웃", ""] + HDR
    periods = [("IS", is_starts), ("OOS", oos_starts)] + ([("홀드아웃", hold_starts)] if hold_starts else [])
    for tag, ss in periods:
        L.append(line("동일가중 보유", "KOSPI100", "-", tag, bench_dist(data, U["KOSPI100"], cal, ss)))
    for fam, cls, u, p, _ in best.values():
        for k, cap in ((1, 600_000), (2, 300_000), (3, 200_000)):
            for tag, ss in periods:
                m = month_dist(cls, ctxs[u], U[u], p, cal, ss, k, cap)
                L.append(line(label(cls, p), u, k, tag, m))
                print(tag, k, label(cls, p), u, pct(m["mean"]), flush=True)

    # ③ 생존편향 점검: 유니버스별 같은 규칙, 상위 10건을 뺀 나머지 거래의 합
    L += ["", "## ③ 생존편향 점검 — 연속 운용(IS 시작 ~ 끝, 2슬롯·30만원)에서 이익이 어디서 났나", "",
          "| 설정 | 유니버스 | 거래 | 누적 | KOSDAQ 비중 | 상위 10건 기여 (로그합) | 나머지 거래 로그합 |", "|---|---|---|---|---|---|---|"]
    kq = U["KOSDAQ150"]
    for fam, cls, u, p, _ in best.values():
        for uu in dict.fromkeys((u, "KOSPI100")):
            st = MaxPrice(cls(ctxs[uu], U[uu], **p), 300_000)
            st.inner.k = 2
            rr = r2.run(st, cal, is_starts[0], end, 2)
            tr = sorted(rr["trades"], key=lambda t: -t.net)
            lg = [math.log1p(t.net) for t in tr]
            L.append(f"| {label(cls, p)} | {uu} | {len(tr)} | {pct(rr['liq'] - 1)} | "
                     f"{sum(t.t in kq for t in tr) / max(len(tr), 1) * 100:.0f}% | {sum(lg[:10]):+.2f} | {sum(lg[10:]):+.2f} |")

    # ④ KOSPI100 한정 재선택 (③에서 KOSDAQ 포함 유니버스가 생존편향에 크게 기댄다는 게 드러나서 추가한 단계.
    #    선택 규칙은 ②와 같고 유니버스만 KOSPI100 으로 제한)
    best100 = {}
    for fam, cls, u, p, m in rows:
        if u != "KOSPI100" or m["p10d"] > 0.25:
            continue
        if fam not in best100 or m["mean"] > best100[fam][4]["mean"]:
            best100[fam] = (fam, cls, u, p, m)
    L += ["", "## ④ KOSPI100 한정 재선택 → OOS · 홀드아웃 (③ 결과를 보고 추가한 단계)", ""] + HDR
    for fam, cls, u, p, _ in best100.values():
        for k, cap in ((1, 600_000), (2, 300_000), (3, 200_000)):
            for tg, ss in periods:
                m = month_dist(cls, ctxs[u], U[u], p, cal, ss, k, cap)
                L.append(line(label(cls, p), u, k, tg, m))
                print("K100", tg, k, label(cls, p), pct(m["mean"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    tag = "5y" if hold_starts else "3y"
    path = os.path.join(OUT, f"research4_{tag}_{end}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
