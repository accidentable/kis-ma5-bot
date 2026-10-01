"""
미국 — 한 달 안에 +30% 가는 기회는 얼마나 있고 어떤 입구에서 시작하나 (한국판 tail_inventory 와 같은 틀)

대상 미국 상장 보통주 시총 $2B↑ (현재 기준 → 생존 편향), 종가 $5↑. 비용 전. 20거래일 창.
입구 신호는 한국판과 같되 상·하한가 대신 '하루 −20%↓ / +20%↑' 를 극단 구간으로 둔다.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from numpy.lib.stride_tricks import sliding_window_view as swv
from backtest.lab.us.common import US, lag, lead, pct_mask

D = US(); T, N = D.T, D.N; W = 20
cf, c, r1, U = D.cf, D.c, D.r1, D.U
pad = np.vstack([cf, np.full((W, N), np.nan)])
fwd = swv(pad, W + 1, axis=0)[:, :, 1:]
fmax = np.nanmax(fwd, axis=2) / cf - 1; fmin = np.nanmin(fwd, axis=2) / cf - 1; fend = fwd[:, :, -1] / cf - 1
idx = np.cumprod(1 + np.nan_to_num(D.mkt)); m20 = idx / lag(idx[:, None], 20)[:, 0] - 1
ups = np.zeros((T, N), int); up = np.nan_to_num(r1) > 0
for t in range(1, T): ups[t] = np.where(up[t], ups[t - 1] + 1, 0)
E = U & np.isfinite(r1)
top500 = D.caprank <= 500
SIG = {
    '전체 (대상 전 종목)': E,
    '시총 상위 500 (S&P500 급)': E & top500,
    '시총 501~ (중형주)': E & ~top500,
    '폭락일 급락 −7%↓': E & D.crash[:, None] & (r1 <= -0.07),
    '평소 날 급락 −10%↓': E & ~D.crash[:, None] & (r1 <= -0.10),
    '하루 −20%↓ (실적 쇼크 급)': E & (r1 <= -0.20),
    '20일 신고가 돌파 & 거래 2배↑': E & (c > D.hi20p) & (D.vr >= 2),
    '52주 고가 근접 (≥98%)': E & (c / D.hi250 >= 0.98),
    '+10~20% 급등 마감': E & (r1 >= 0.10) & (r1 < 0.20),
    '+20%↑ 급등 마감': E & (r1 >= 0.20),
    '20일 수익률 상위 10%': E & pct_mask(D.ret20, E, 0.9, 1.0),
    '20일 수익률 하위 10%': E & pct_mask(D.ret20, E, 0.0, 0.1),
    '60일 변동성 상위 10%': E & pct_mask(D.vol60, E, 0.9, 1.0),
    '거래대금 3배↑ (방향 무관)': E & (D.vr >= 3),
    '3일 연속 상승': E & (ups >= 3),
    '시장 강세 (20일 > +5%)': E & (m20 > 0.05)[:, None],
    '시장 약세 (20일 < −5%)': E & (m20 < -0.05)[:, None],
}
PER = D.periods(W + 1)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P(f'미국 한 달(20거래일) 꼬리 기회 — 대상 {int(np.nanmedian(U.sum(1))):,}종목/일, 종가 매수, 비용 전. 열: 건수 | 최고가 +20%↑ / +30%↑ / +50%↑ | 20일째 +30%↑ | 최저가 −20%↓ | 20일째 평균')
for pn, (a, b) in PER.items():
    P(f'\n[{pn}]')
    for sn, m in SIG.items():
        mm = m.copy(); mm[:a] = False; mm[b:] = False
        v = fmax[mm]; ok = np.isfinite(v); v = v[ok]; e = fend[mm][ok]; mn = fmin[mm][ok]
        if len(v) < 30: P(f'  {sn:<28} 표본 부족 ({len(v)})'); continue
        P(f'  {sn:<28} {len(v):8,d} | {(v >= .2).mean() * 100:5.1f}% {(v >= .3).mean() * 100:5.1f}% {(v >= .5).mean() * 100:5.1f}% | {(e >= .3).mean() * 100:5.1f}% | {(mn <= -.2).mean() * 100:5.1f}% | {e.mean() * 100:+5.2f}%')
P('\n■ 매달(20일 창) 대상 중 20일 내 최고 종가 +30%↑ 종목 수 — 평균 / 중앙 / 0개 달 / 5개↑ / 20개↑ ; 20일째 +30%↑ 평균')
for pn, (a, b) in PER.items():
    cnt = np.array([(E[s] & (fmax[s] >= .3)).sum() for s in range(a, b)]); cnt_e = np.array([(E[s] & (fend[s] >= .3)).sum() for s in range(a, b)])
    P(f'  {pn}: 평균 {cnt.mean():5.1f} 중앙 {np.median(cnt):4.0f} | 0개 {(cnt == 0).mean() * 100:4.1f}% 5개↑ {(cnt >= 5).mean() * 100:4.0f}% 20개↑ {(cnt >= 20).mean() * 100:4.0f}% | 20일째 +30%↑ 평균 {cnt_e.mean():5.1f}개 (대상 {E[a:b].sum(1).mean():.0f})')
P('\n■ 폭락일(시장 −3% & 3σ) 수: ' + ', '.join(f'{pn} {D.crash[a:b].sum()}일' for pn, (a, b) in PER.items()) + f' | 시장 하루 −3%↓ 일수: ' + ', '.join(f'{pn} {(D.mkt[a:b] <= -0.03).sum()}' for pn, (a, b) in PER.items()))
open('backtest/results/us_tail_inventory.md', 'w', encoding='utf-8').write('# 미국 한 달 꼬리 기회 (backtest/lab/us/us_inventory.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
