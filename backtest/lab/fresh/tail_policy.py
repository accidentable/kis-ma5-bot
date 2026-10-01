"""
한 달 게임 정책 층 — 집중 모멘텀 + 폭락 전환 위에 목표 락 · 손절 · 추적 · '뒤처지면 집중' 을 격자로 (대회용 리서치 Phase C)

바탕 전략 (Phase B 에서 남은 것)
  D  모멘텀 집중: 20일 수익률 상위 10% 중 높은 순 k개, 최대 10일 보유 뒤 교체
  E  고변동 모멘텀: 60일 변동성 상위 10% ∩ 20일 수익률 상위 절반, 높은 순 k개
  I  D + T5 폭락일엔 폭락 급락주(가격지표 필터) k개로 전환, 5일 뒤 복귀
정책 격자 (미리 정함)
  k      1 / 2
  lock   +20% / +30% / +40% 찍으면(종가 NAV) 다음 날 시가 전량 매도, 월말까지 현금
  stop   −7% / −10% (종가 판정 → 다음 날 시가)
  trail  없음 / 최고 종가 −10%
  boost  없음 / 11일째부터 NAV < +5% 면 k=1 로 집중 (뒤처지면 분산 포기)
지표   월 +20%↑ / +30%↑ / −20%↓ / −30%↓ 확률, 중앙, 평균. 선택 기준: 2020~25 +30% 확률 최대, 단 2011~19 에서도 같은 방향이고 −30% 확률이 10% 를 넘지 않는 것.
"""
import numpy as np, warnings, time, sys, itertools, pickle; warnings.filterwarnings('ignore')
from backtest.lab.fresh import tail_strategies as TS
from backtest.lab import panic2_common as C

F, T, o, c, cf, tr, T5, CAND, PER = TS.F, TS.T, TS.o, TS.c, TS.cf, TS.tr, TS.T5, TS.CAND, TS.PER
BUYC, SELLC, CAP, W = TS.BUYC, TS.SELLC, TS.CAP, TS.W


def run_month(s, cand, k, lock, stop, trail, boost, mix):
    e = s + W - 1
    cash, pos, sell_open, locked, ov, kk = CAP, {}, set(), False, False, k

    def nav(t): return cash + sum(p[0] * cf[t, j] * (1 - SELLC) for j, p in pos.items())

    def sell(j, t, px):
        nonlocal cash
        if not tr[t, j] or not px > 0: return False
        cash += pos.pop(j)[0] * px * (1 - SELLC); return True

    def buy(j, t, px, kind, budget, hold):
        nonlocal cash
        if j in pos or not tr[t, j] or not px > 0: return False
        q = np.floor(min(budget, cash) / (px * (1 + BUYC)))
        if q < 1: return False
        cash -= q * px * (1 + BUYC); pos[j] = [q, px, px, t + hold, kind]; return True

    for t in range(s, e + 1):
        for j in list(sell_open): sell(j, t, o[t, j])
        sell_open = set()
        if locked:
            for j in list(pos): sell(j, t, o[t, j])
            continue
        for j in [j for j, p in pos.items() if p[3] == t]: sell(j, t, o[t, j])
        if boost and t - s >= 10 and nav(t - 1) < CAP * 1.05 and kk > 1:
            kk = 1                                                   # 집중: 가장 강한 하나만 남기고 시가에 판다 (정지 종목은 못 팔면 그대로)
            keep = max(pos, key=lambda j: cf[t - 1, j] / pos[j][1]) if pos else None
            for j in [j for j in list(pos) if j != keep]: sell(j, t, o[t, j])
        crash_now = mix and T5[t] and t < e
        if crash_now and not ov:
            for j in list(pos): sell(j, t, c[t, j])
            nv = nav(t)
            for j in CAND['A'][t][:kk]: buy(j, t, c[t, j], 'A', nv / kk, t + 6)
            ov = True
        else:
            if ov and not any(p[4] == 'A' for p in pos.values()): ov = False
            for j, p in pos.items():
                p[2] = max(p[2], cf[t, j])
                if (stop and cf[t, j] <= p[1] * (1 - stop)) or (trail and cf[t, j] <= p[2] * (1 - trail)): sell_open.add(j)
            if not ov and t < e:
                free = kk - len(pos)
                nv = nav(t)
                for j in [j for j in CAND[cand][t] if j not in pos][:max(free, 0)]: buy(j, t, c[t, j], cand, nv / kk, min(t + 11, e + 1))
        if lock and nav(t) >= CAP * (1 + lock): locked = True
    for j in list(pos): sell(j, e, c[e, j])
    return cash / CAP - 1


def evaluate(cfg):
    out = {}
    for pn, (a, b) in PER.items():
        out[pn] = np.array([run_month(s, *cfg) for s in range(a, b)])
    return out


def fmt(r): return f'{(r >= .2).mean() * 100:4.1f} {(r >= .3).mean() * 100:4.1f} {(r <= -.2).mean() * 100:4.1f} {(r <= -.3).mean() * 100:4.1f} {np.median(r) * 100:+5.1f} {r.mean() * 100:+5.1f}'


if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('정책 격자 — 20일 창, 1억, 비용 포함. 각 기간 열: +20%↑ · +30%↑ · −20%↓ · −30%↓ 확률(%) · 중앙 · 평균')
    P(f'  {"설정":<34} | {"2011~2019":^34} | {"2020~2025":^34} | {"2026":^34}')
    RES = {}
    grid = list(itertools.product(('D', 'E', 'I'), (1, 2), (0.2, 0.3, 0.4), (0.07, 0.10), (None, 0.10), (False, True)))
    for i, (base, k, lock, stop, trail, boost) in enumerate(grid):
        cand = 'D' if base in ('D', 'I') else 'E'; mix = base == 'I'
        label = f'{base} k{k} 락{lock * 100:.0f} 손절{stop * 100:.0f}{" 추적10" if trail else "      "}{" 집중" if boost else "     "}'
        r = evaluate((cand, k, lock, stop, trail, boost, mix)); RES[label] = r
        P(f'  {label:<34} | ' + ' | '.join(fmt(r[pn]) for pn in PER))
        if i % 24 == 23: print(f'  … {i + 1}/{len(grid)} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    P('\n■ 2020~2025 +30% 확률 상위 15 (2011~19 +30% 확률 · −30% 확률 함께)')
    rank = sorted(RES.items(), key=lambda kv: -(kv[1]['2020~2025'] >= .3).mean())
    for label, r in rank[:15]:
        P(f'  {label:<34} 2020~25 +30% {(r["2020~2025"] >= .3).mean() * 100:4.1f}% −30% {(r["2020~2025"] <= -.3).mean() * 100:4.1f}% | 2011~19 +30% {(r["2011~2019"] >= .3).mean() * 100:4.1f}% −30% {(r["2011~2019"] <= -.3).mean() * 100:4.1f}% | 2026 +30% {(r["2026"] >= .3).mean() * 100:4.1f}%')
    pickle.dump(RES, open('/data/lab/tail_policy.pkl', 'wb'))
    open('backtest/results/lab_tail_policy.md', 'w', encoding='utf-8').write(
        '# 한 달 게임 정책 격자 (backtest/lab/fresh/tail_policy.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
