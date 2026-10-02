"""
backtest/lab/us/us_engine.py — 한 달(20거래일) 창 시뮬레이터 (한국판 tail_policy 와 같은 규칙, 미국 데이터용)

진입 = 신호일 종가 (buy_at='close') 또는 다음 날 시가 ('open'). 청산 = 보유 hold 일 뒤 시가 / 손절·추적(종가 판정 → 다음 날 시가) / 목표 락(종가 NAV ≥ +lock → 다음 날 시가 전량, 월말까지 현금).
폭락 전환(mix): 폭락일 종가에 전량 매도 → 급락 후보 k개 종가 매수 → 6일째 시가 매도.
비용: 편도 BUYC · SELLC (기본 0.30%). 창 끝엔 종가 전량 매도.
"""
from __future__ import annotations

import numpy as np

W = 20


def make_runner(D, cands: dict, crash_list=None, buyc=0.003, sellc=0.003, cap=1e8):
    o, c, cf, tr, T = D.o, D.c, D.cf, D.tr, D.T
    crash = D.crash

    def run_month(s, cand, k=1, hold=10, stop=0.10, trail=0.10, lock=0.30, mix=False, buy_at='close', signal_only=False, max_entries=None):
        e = s + W - 1
        cash, pos, sell_open, pend_buy, locked, ov = cap, {}, set(), [], False, False
        entries = 0  # max_entries: 한 달 매수 횟수 상한 (손절 뒤 재진입 제한용)
        peak_nav = cap

        def nav(t): return cash + sum(p[0] * cf[t, j] * (1 - sellc) for j, p in pos.items())

        def sell(j, t, px):
            nonlocal cash
            if not tr[t, j] or not px > 0: return False
            cash += pos.pop(j)[0] * px * (1 - sellc); return True

        def buy(j, t, px, kind, budget, hold_):
            nonlocal cash, entries
            if j in pos or not tr[t, j] or not px > 0: return False
            if max_entries is not None and entries >= max_entries: return False
            q = np.floor(min(budget, cash) / (px * (1 + buyc)))
            if q < 1: return False
            cash -= q * px * (1 + buyc); pos[j] = [q, px, px, t + hold_, kind]; entries += 1; return True

        for t in range(s, e + 1):
            for j in list(sell_open): sell(j, t, o[t, j])
            sell_open = set()
            if locked:
                for j in list(pos): sell(j, t, o[t, j])
                continue
            for j in [j for j, p in pos.items() if p[3] == t]: sell(j, t, o[t, j])
            for j, kind, hold_ in pend_buy: buy(j, t, o[t, j], kind, nav(t - 1) / k, hold_)
            pend_buy = []
            crash_now = mix and crash[t] and t < e
            if crash_now and not ov:
                for j in list(pos): sell(j, t, c[t, j])
                nv = nav(t)
                for j in (crash_list[t] if crash_list else [])[:k]: buy(j, t, c[t, j], 'A', nv / k, 6)
                ov = True
            else:
                if ov and not any(p[4] == 'A' for p in pos.values()): ov = False
                for j, p in pos.items():
                    p[2] = max(p[2], cf[t, j])
                    if p[4] != 'A' and ((stop and cf[t, j] <= p[1] * (1 - stop)) or (trail and cf[t, j] <= p[2] * (1 - trail))): sell_open.add(j)
                if not ov and t < e:
                    free = k - len(pos) - len(pend_buy)
                    picks = [j for j in cands[cand][t] if j not in pos][:max(free, 0)]
                    if buy_at == 'close':
                        nv = nav(t)
                        for j in picks: buy(j, t, c[t, j], cand, nv / k, hold + 1)
                    else:
                        pend_buy += [(j, cand, hold) for j in picks]
            nv = nav(t); peak_nav = max(peak_nav, nv)
            if lock and nv >= cap * (1 + lock): locked = True
        for j in list(pos): sell(j, e, c[e, j])
        return cash / cap - 1, peak_nav / cap - 1

    return run_month


def stats(r, pk=None):
    out = f'{(r >= .2).mean() * 100:4.1f} {(r >= .3).mean() * 100:4.1f}'
    if pk is not None: out += f' {(pk >= .3).mean() * 100:4.1f}'
    return out + f' {(r <= -.2).mean() * 100:4.1f} {(r <= -.3).mean() * 100:4.1f} {np.median(r) * 100:+5.1f} {r.mean() * 100:+5.1f}'


def evaluate(run_month, periods, **kw):
    out = {}
    for pn, (a, b) in periods.items():
        res = np.array([run_month(s, **kw) for s in range(a, b)])
        out[pn] = (res[:, 0], res[:, 1])
    return out
