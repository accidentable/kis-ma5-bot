"""
미국 — 밤(종가 → 다음 날 시가) / 낮(시가 → 종가) 수익 분해. 한국판 overnight_split 과 같은 틀. 비용 전.
묶음: 대상 전체 / 시총 500 / 20일 모멘텀 상위 10% / 폭락일 급락주 / 평소 급락 −10% / 고변동 상위 10% / 레버리지 ETF(TQQQ·SOXL·UPRO·TNA) / SPY·QQQ
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, lead, pct_mask

D = US(); T, N = D.T, D.N
o, c, cf, r1, U, tr = D.o, D.c, D.cf, D.r1, D.U, D.tr
E = U & np.isfinite(r1)
oo, cc = np.where(tr, o, np.nan), np.where(tr, c, np.nan)
night = [lead(oo, k) / (cc if k == 1 else lead(cc, k - 1)) - 1 for k in range(1, 6)]
day = [lead(cc, k) / lead(oo, k) - 1 for k in range(1, 6)]
def cols(*ts):
    m = np.zeros((T, N), bool)
    for t_ in ts:
        if t_ in set(D.tickers): m[:, D.col(t_)] = tr[:, D.col(t_)]
    return m
GROUPS = {
    '전체 (대상 평균)': E, '시총 상위 500': E & (D.caprank <= 500)[None, :], '20일 모멘텀 상위 10%': E & pct_mask(D.ret20, E, 0.9, 1.0),
    '폭락일 급락 −7%↓': E & D.crash[:, None] & (r1 <= -0.07), '평소 급락 −10%↓': E & ~D.crash[:, None] & (r1 <= -0.10),
    '60일 변동성 상위 10%': E & pct_mask(D.vol60, E, 0.9, 1.0), '3x ETF (TQQQ·SOXL·UPRO·TNA)': cols('TQQQ', 'SOXL', 'UPRO', 'TNA'), 'SPY·QQQ': cols('SPY', 'QQQ'),
}
PER = D.periods(8)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P('미국 밤/낮 분해 (비용 전, %) — 밤k = D+k−1 종가 → D+k 시가, 낮k = D+k 시가 → D+k 종가')
def avg(mask, arr, a, b):
    m = mask.copy(); m[:a] = False; m[b:] = False; v = arr[m]; v = v[np.isfinite(v)]
    return v.mean() * 100 if len(v) >= 15 else np.nan
for pn, (a, b) in PER.items():
    P(f'\n[{pn}]'); P(f'  {"묶음":<26}{"건수":>8} | ' + ' '.join(f'밤{k} 낮{k}' for k in range(1, 6)) + ' | 밤합 낮합 | 종가→D+5종가 종가→D+6시가 시가→D+5종가 시가→D+6시가')
    for gn, gm in GROUPS.items():
        m = gm.copy(); m[:a] = False; m[b:] = False
        nk = [avg(gm, night[k], a, b) for k in range(5)]; dk = [avg(gm, day[k], a, b) for k in range(5)]
        tot = [avg(gm, lead(cc, 5) / cc - 1, a, b), avg(gm, lead(oo, 6) / cc - 1, a, b), avg(gm, lead(cc, 5) / lead(oo, 1) - 1, a, b), avg(gm, lead(oo, 6) / lead(oo, 1) - 1, a, b)]
        P(f'  {gn:<26}{m.sum():8d} | ' + ' '.join(f'{nk[k]:+5.2f} {dk[k]:+5.2f}' for k in range(5)) + f' | {np.nansum(nk):+5.2f} {np.nansum(dk):+5.2f} | ' + ' '.join(f'{x:+6.2f}' for x in tot))
open('backtest/results/us_overnight_split.md', 'w', encoding='utf-8').write('# 미국 밤/낮 분해 (backtest/lab/us/us_overnight.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
