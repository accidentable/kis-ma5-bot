"""
큰 범주로 훑기 — 지표는 '건당 평균 순수익' 하나만 (비용 0.31% 차감). 코스피 시총 100 + 코스닥 시총 150.
  표 1  오늘(D) 등락 구간 × 매수 시점 (D 종가 · D+1 시가 · D+2 시가), 보유 1 · 3 · 5 · 10일
  표 2  어제(D−1) 등락 × 오늘(D) 등락 격자, D+1 시가 매수 · 5일 보유
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab.panic3 import move_timing as M

U, r1, prev = M.U, M.r1, M.prev
EDGES = [-1, -0.10, -0.05, -0.03, 0, 0.03, 0.05, 0.10, 1]
LAB = ['−10%↓', '−10~−5', '−5~−3', '−3~0', '0~+3', '+3~+5', '+5~+10', '+10%↑']
PER = dict(M.PER); PER['전체'] = (PER['2011~2019'][0], PER['2020~'][1])


def bucket(x):
    return [(x > lo) & (x <= hi) if lo > -1 else (x <= hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def avg(mask, R, a, b):
    mm = mask.copy(); mm[:a] = False; mm[b:] = False
    v = R[mm]; v = v[np.isfinite(v)]
    return (v.mean(), len(v)) if len(v) >= 30 else (np.nan, len(v))


ok = U & np.isfinite(r1) & (np.abs(r1) < 0.295)      # 상·하한가는 빼고 (종가에 못 사거나 악재)
lines = []
P = lines.append
P('표 1. 오늘 등락 × 매수 시점 — 건당 평균 순수익 %')
for pn, (a, b) in PER.items():
    P(f'\n[{pn}]')
    P(f'{"오늘":<9}{"건수":>8} | ' + ' | '.join(f'{e:^23}' for e in ('D 종가', 'D+1 시가', 'D+2 시가')))
    P(f'{"":<9}{"":>8} | ' + ' | '.join('  1일   3일   5일  10일' for _ in range(3)))
    for lab, bm in zip(LAB, bucket(r1)):
        m = ok & bm
        cells, n = [], 0
        for e in ('D 종가', 'D+1 시가', 'D+2 시가'):
            vs = []
            for h in (1, 3, 5, 10):
                v, n0 = avg(m, M.RET[e, h], a, b); vs.append(v); n = max(n, n0)
            cells.append(' '.join(f'{v * 100:+5.2f}' for v in vs))
        P(f'{lab:<9}{n:>8,} | ' + ' | '.join(cells))

P('\n\n표 2. 어제 × 오늘 등락 — D+1 시가 매수 · 5일 보유 평균 순수익 % (괄호 = 건수)')
for pn, (a, b) in PER.items():
    P(f'\n[{pn}]  행 = 어제, 열 = 오늘')
    P(f'{"":<9}' + ''.join(f'{l:>15}' for l in LAB))
    for lp, bp in zip(LAB, bucket(prev)):
        row = f'{lp:<9}'
        for lt, bt in zip(LAB, bucket(r1)):
            v, n = avg(ok & bp & bt & (np.abs(prev) < 0.295), M.RET['D+1 시가', 5], a, b)
            row += f'{"-" if np.isnan(v) else f"{v * 100:+.2f}":>8}({n:>5,})'
        P(row)
txt = '\n'.join(lines)
print('\n\n' + txt)
open('backtest/results/lab_move_grid.md', 'w').write('# 큰 범주 격자 (move_grid.py)\n\n```\n' + txt + '\n```\n')
