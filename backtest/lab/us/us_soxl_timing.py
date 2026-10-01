"""
미국 — SOXL 진입 타이밍. 20일 창 시작 뒤 조건이 '처음 성립하는 날' 종가에 한 번 매수 (락 30 · 손절 15 뒤 월말까지 현금 / 손절 없음).
조건이 창 안에서 한 번도 안 서면 그 달은 현금(0%). 창은 매 거래일 시작(겹침) → 가능한 모든 '대회 달' 의 평균.
열: 거래한 달 % | 거래한 달 중 +20%↑ · +30%↑ · −20%↓ · −30%↓ · 중앙 · 평균 | 전체 달 기준 +30%↑ · −30%↓ (%)
"""
import numpy as np, pandas as pd, warnings, time, sys; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US
from backtest.lab.us.us_engine import make_runner

t0 = time.time(); D = US(); T = D.T; j = D.col('SOXL')
c = D.c[:, j].astype(float); ok = D.tr[:, j] & np.isfinite(c); r1 = D.r1[:, j]
S = pd.Series(c)
def lag_ret(n): r = np.full(T, np.nan); r[n:] = c[n:] / c[:-n] - 1; return r
ret5, ret20 = lag_ret(5), lag_ret(20)
sma50, sma200 = S.rolling(50).mean().values, S.rolling(200).mean().values
hi20, hi60, lo20 = S.rolling(20).max().values, S.rolling(60).max().values, S.rolling(20).min().values
dd60 = c / hi60 - 1
CONDS = {
    '항상 (첫날 매수)': np.ones(T, bool),
    '전날 하루 −7%↓': r1 <= -0.07, '전날 하루 +7%↑': r1 >= 0.07,
    '5일 −5%↓': ret5 <= -0.05, '5일 −10%↓': ret5 <= -0.10, '5일 −15%↓': ret5 <= -0.15, '5일 +10%↑': ret5 >= 0.10,
    '20일 −20%↓': ret20 <= -0.20, '20일 +20%↑': ret20 >= 0.20,
    '60일 고점 대비 −30%↓': dd60 <= -0.30, '20일 신저가': c <= lo20, '20일 신고가': c >= hi20,
    '50일선 위': c > sma50, '50일선 아래': c < sma50, '200일선 위': c > sma200, '200일선 아래': c < sma200,
}
PER = D.periods()
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
def block(title, **kw):
    P(f'\n■ {title}')
    P(f'  {"진입 조건":<22} | ' + ' | '.join(f'{pn:^52}' for pn in PER))
    P(f'  {"":<22} | ' + ' | '.join(f'{"거래%":>5} {"+20":>5} {"+30":>5} {"-20":>5} {"-30":>5} {"중앙":>5} {"평균":>5} | {"+30":>4} {"-30":>4}' for _ in PER))
    for name, cond in CONDS.items():
        cands = {'X': [[j] if (cond[t] and ok[t]) else [] for t in range(T)]}
        run = make_runner(D, cands); cells = []
        for pn, (a, b) in PER.items():
            r = np.array([run(s, 'X', k=1, hold=25, trail=None, lock=0.3, **kw)[0] for s in range(a, b)])
            tr_ = r != 0; x = r[tr_]
            if tr_.sum() < 20: cells.append(f'{"표본 부족":^52}'); continue
            cells.append(f'{tr_.mean() * 100:5.1f} {(x >= .2).mean() * 100:5.1f} {(x >= .3).mean() * 100:5.1f} {(x <= -.2).mean() * 100:5.1f} {(x <= -.3).mean() * 100:5.1f} {np.median(x) * 100:+5.1f} {x.mean() * 100:+5.1f} | {(r >= .3).mean() * 100:4.1f} {(r <= -.3).mean() * 100:4.1f}')
        P(f'  {name:<22} | ' + ' | '.join(cells))
block('락 30 · 손절 15 → 월말까지 현금 (추천 규칙)', stop=0.15, max_entries=1)
block('락 30 · 손절 없음', stop=None)
open('backtest/results/us_soxl_timing.md', 'w', encoding='utf-8').write('# SOXL 진입 타이밍 (backtest/lab/us/us_soxl_timing.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
