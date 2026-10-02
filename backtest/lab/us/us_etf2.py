"""
미국 — 레버리지 ETF 단일 보유 + 락/손절 변형 (SOXL · TQQQ · LABU · CONL · NVDL), 그리고 'SOXL vs TQQQ 중 20일 모멘텀 높은 쪽'.
20일 창, 종가 매수, 시가 매도, 비용 편도 0.30%. 열: +20%↑ · +30%↑ · −20%↓ · −30%↓ · 중앙 · 평균 (%)
"""
import numpy as np, warnings, time, sys, itertools; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, lists
from backtest.lab.us.us_engine import make_runner, evaluate, stats

t0 = time.time(); D = US(); T, N = D.T, D.N; tr = D.tr
have = set(D.tickers)
def one(t_):
    m = np.zeros((T, N), bool); m[:, D.col(t_)] = tr[:, D.col(t_)]; return [np.where(m[t])[0].tolist() for t in range(T)]
def pair(*names):
    m = np.zeros((T, N), bool)
    for t_ in names: m[:, D.col(t_)] = tr[:, D.col(t_)]
    return lists(m & np.isfinite(D.ret20), D.ret20)
CANDS = {t_: one(t_) for t_ in ('SOXL', 'TQQQ', 'LABU', 'CONL', 'NVDL', 'TNA') if t_ in have}
CANDS['SOXL/TQQQ'] = pair('SOXL', 'TQQQ'); CANDS['SOXL/TQQQ/LABU/TNA'] = pair('SOXL', 'TQQQ', 'LABU', 'TNA')
PER = D.periods(); run = make_runner(D, CANDS)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P(f'  {"설정":<34} | {"2011~2019":^36} | {"2020~2025":^36} | {"2026":^36}')
RES = {}
for cand, lock, stop, trail in itertools.product(list(CANDS), (None, 0.2, 0.3, 0.4), (None, 0.10, 0.15), (None, 0.10)):
    if cand in ('CONL', 'NVDL') and (stop == 0.15 or lock == 0.2): continue
    label = f'{cand:<18} 락{str(int(lock * 100)) if lock else "없":<2} 손절{str(int(stop * 100)) if stop else "없":<2} 추적{"10" if trail else "없"}'
    hold = 10 if '/' in cand else 25
    res = evaluate(run, PER, cand=cand, k=1, hold=hold, stop=stop, trail=trail, lock=lock)
    RES[label] = res; P(f'  {label:<34} | ' + ' | '.join(stats(res[pn][0]) for pn in PER))
P('\n■ 2020~2025 +30% 확률 상위 12 (−30% 확률 함께)')
for label, res in sorted(RES.items(), key=lambda kv: -(kv[1]['2020~2025'][0] >= .3).mean())[:12]:
    P(f'  {label:<34} 2020~25 +30% {(res["2020~2025"][0] >= .3).mean() * 100:4.1f}% −30% {(res["2020~2025"][0] <= -.3).mean() * 100:4.1f}% | 2011~19 +30% {(res["2011~2019"][0] >= .3).mean() * 100:4.1f}% −30% {(res["2011~2019"][0] <= -.3).mean() * 100:4.1f}% | 2026 +30% {(res["2026"][0] >= .3).mean() * 100:4.1f}%')
open('backtest/results/us_etf2.md', 'w', encoding='utf-8').write('# 미국 레버리지 ETF 단일 보유 변형 (backtest/lab/us/us_etf2.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
