"""
미국 — 레버리지 ETF 보유 + 락 에서 손절 뒤 재진입을 막으면? (us_etf2 보완)
us_etf2 는 손절(종가 판단 → 다음 날 시가 매도) 뒤 같은 ETF 를 그날 종가에 다시 사서 휩소가 난다.
여기서는 재진입 0회(손절 뒤 월말까지 현금) · 1회 · 무제한을 비교. 보유 25 = 창 안에서 회전 없음. 비용 편도 0.30%.
열: +20%↑ · +30%↑ · −20%↓ · −30%↓ · 중앙 · 평균 (%)
"""
import numpy as np, warnings, time, sys, itertools; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US
from backtest.lab.us.us_engine import make_runner, evaluate, stats

t0 = time.time(); D = US(); T, N = D.T, D.N; tr = D.tr
def one(t_):
    m = np.zeros((T, N), bool); m[:, D.col(t_)] = tr[:, D.col(t_)]; return [np.where(m[t])[0].tolist() for t in range(T)]
CANDS = {t_: one(t_) for t_ in ('SOXL', 'TQQQ', 'NVDL') if t_ in set(D.tickers)}
PER = D.periods(); run = make_runner(D, CANDS)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P(f'  {"설정":<34} | {"2011~2019":^36} | {"2020~2025":^36} | {"2026":^36}')
for cand in CANDS:
    for lock, stop, maxe in itertools.product((0.3, 0.2), (None, 0.10, 0.15, 0.20), (None, 1, 2)):
        if stop is None and maxe is not None: continue
        re = '무제한' if maxe is None else f'{maxe - 1}회'
        label = f'{cand:<5} 락{int(lock * 100)} 손절{str(int(stop * 100)) if stop else "없":<2} 재진입{re}'
        res = evaluate(run, PER, cand=cand, k=1, hold=25, stop=stop, trail=None, lock=lock, max_entries=maxe)
        P(f'  {label:<34} | ' + ' | '.join(stats(res[pn][0]) for pn in PER))
    P()
open('backtest/results/us_etf3.md', 'w', encoding='utf-8').write('# 미국 레버리지 ETF 손절 뒤 재진입 제한 (backtest/lab/us/us_etf3.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
