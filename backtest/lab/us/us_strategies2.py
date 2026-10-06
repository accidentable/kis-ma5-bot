"""
미국 — 기회 목록에서 두드러진 입구를 전략으로 (us_strategies 보완): 하루 −20% 쇼크 반등 · +20% 급등 추격 · 고변동 · 20일 하위 반전
전부 종가 매수 · 시가 매도 · 락 30 · 보유 10 · 손절/추적 변형. 비용 편도 0.30%.
  H20  하루 −20%↓ 종목 (거래 터짐 무관), 더 빠진 순          C20  +20%↑ 급등 마감, 더 오른 순
  V    60일 변동성 상위 10% 중 20일 수익률 높은 순            R    20일 수익률 하위 10% 중 낮은 순 (반전)
  H10  평소 날 −10%↓ (거래 3배↑ 제외), 보유 10 (us_strategies 의 H 는 보유 5)
"""
import numpy as np, warnings, time, sys, itertools; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, lists, pct_mask
from backtest.lab.us.us_engine import make_runner, evaluate, stats

t0 = time.time(); D = US(); r1, U = D.r1, D.U
E = U & np.isfinite(r1)
CANDS = {
    'H20': lists(E & (r1 <= -0.20), r1, desc=False),
    'C20': lists(E & (r1 >= 0.20), r1),
    'V': lists(pct_mask(D.vol60, E, 0.9, 1.0), D.ret20),
    'R': lists(pct_mask(D.ret20, E, 0.0, 0.1), D.ret20, desc=False),
    'H10': lists(E & ~D.crash[:, None] & (r1 <= -0.10) & ~(np.nan_to_num(D.vr, nan=0) >= 3), r1, desc=False),
}
PER = D.periods(); run = make_runner(D, CANDS)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P('미국 보완 전략 — 락 30 · 보유 10. 열: +20%↑ · +30%↑ · −20%↓ · −30%↓ · 중앙 · 평균 (%)')
P(f'  {"설정":<30} | {"2011~2019":^36} | {"2020~2025":^36} | {"2026":^36}')
RES = {}
for cand, k, stop, trail in itertools.product(('H20', 'C20', 'V', 'R', 'H10'), (1, 2), (None, 0.10), (None, 0.10)):
    label = f'{cand:<4} k{k} 손절{"10" if stop else "없"} 추적{"10" if trail else "없"}'
    res = evaluate(run, PER, cand=cand, k=k, hold=10, stop=stop, trail=trail, lock=0.30)
    RES[label] = res; P(f'  {label:<30} | ' + ' | '.join(stats(res[pn][0]) for pn in PER))
P('\n■ 2020~2025 +30% 확률 상위 10')
for label, res in sorted(RES.items(), key=lambda kv: -(kv[1]['2020~2025'][0] >= .3).mean())[:10]:
    P(f'  {label:<30} 2020~25 +30% {(res["2020~2025"][0] >= .3).mean() * 100:4.1f}% −30% {(res["2020~2025"][0] <= -.3).mean() * 100:4.1f}% | 2011~19 +30% {(res["2011~2019"][0] >= .3).mean() * 100:4.1f}% | 2026 +30% {(res["2026"][0] >= .3).mean() * 100:4.1f}%')
open('backtest/results/us_tail_strategies2.md', 'w', encoding='utf-8').write('# 미국 보완 전략 (backtest/lab/us/us_strategies2.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
