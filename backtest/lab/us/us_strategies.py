"""
미국 — 한 달 +20/+30% 확률로 전략 비교 (한국판 Phase B+C). 비용 편도 0.30% 기본, 0.10% 민감도.

전략 (종가 매수, 시가 매도)
  A 폭락 올인 (폭락일에만 −7%↓ 급락주, 거래 터짐 제외, 5일)  C 급등 추격 (+10~25%)  D 모멘텀 집중 (20일 상위 10%)  E 고변동 모멘텀
  F 돌파 추격 (20일 신고가 & 거래 2배)  H 평소 급락 (−10%↓, 5일)  L 대형주 모멘텀 (시총 500 안)  M 중형주 모멘텀 (시총 501~)
  I 혼합 = D + 폭락 전환
정책 격자: D · E · M · I × k 1/2 × 락 20/30/40 × 손절 7/10 × 추적 없음/10 (보유 10일 고정)
"""
import numpy as np, warnings, time, sys, itertools; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, BUYC, lists, pct_mask
from backtest.lab.us.us_engine import make_runner, evaluate, stats

t0 = time.time()
D = US(); T, N = D.T, D.N
r1, U, c = D.r1, D.U, D.c
E = U & np.isfinite(r1)
mom_top = pct_mask(D.ret20, E, 0.9, 1.0); mom_half = pct_mask(D.ret20, E, 0.5, 1.0); vol_top = pct_mask(D.vol60, E, 0.9, 1.0)
large = (D.caprank <= 500)[None, :]
CANDS = {
    'A': lists(E & D.crash[:, None] & (r1 <= -0.07) & ~(np.nan_to_num(D.vr, nan=0) >= 3), r1, desc=False),
    'C': lists(E & (r1 >= 0.10) & (r1 < 0.25), r1),
    'D': lists(mom_top, D.ret20), 'E': lists(vol_top & mom_half, D.ret20),
    'F': lists(E & (c > D.hi20p) & (D.vr >= 2), D.vr),
    'H': lists(E & ~D.crash[:, None] & (r1 <= -0.10) & ~(np.nan_to_num(D.vr, nan=0) >= 3), r1, desc=False),
    'L': lists(mom_top & large, D.ret20), 'M': lists(mom_top & ~large, D.ret20),
}
RULES = {'A 폭락 올인': dict(cand='A', hold=5, stop=None, trail=None, mix=False), 'C 급등 추격': dict(cand='C', hold=10, stop=0.07, trail=0.10),
         'D 모멘텀 집중': dict(cand='D', hold=10, stop=0.07, trail=None), 'E 고변동 모멘텀': dict(cand='E', hold=10, stop=0.07, trail=0.12),
         'F 돌파 추격': dict(cand='F', hold=15, stop=None, trail=0.08), 'H 평소 급락': dict(cand='H', hold=5, stop=None, trail=None),
         'L 대형주 모멘텀': dict(cand='L', hold=10, stop=0.07, trail=None), 'M 중형주 모멘텀': dict(cand='M', hold=10, stop=0.07, trail=None)}
PER = D.periods()
print(f'준비 {time.time() - t0:.0f}s, 대상 {int(np.nanmedian(U.sum(1)))}종목/일', flush=True)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P('미국 꼬리 포착 전략 — 20일 창, 비용 편도 0.30%. 각 기간 열: +20%↑ · +30%↑ · 창 안 +30% 찍음 · −20%↓ · −30%↓ · 중앙 · 평균 (%)')
P(f'  {"전략 (k)":<26} | {"2011~2019":^38} | {"2020~2025":^38} | {"2026":^38}')
run = make_runner(D, CANDS, CANDS['A'])
for rn, r in RULES.items():
    for k in (1, 2, 3):
        res = evaluate(run, PER, cand=r['cand'], k=k, hold=r['hold'], stop=r['stop'], trail=r['trail'], lock=None, mix=r.get('mix', False))
        P(f'  {rn:<24} k{k} | ' + ' | '.join(stats(res[pn][0], res[pn][1]) for pn in PER))
    print(f'  … {rn} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
P('\n■ 정책 격자 (보유 10일). 열: +20%↑ · +30%↑ · −20%↓ · −30%↓ · 중앙 · 평균')
RES = {}
for base, k, lock, stop, trail in itertools.product(('D', 'E', 'M', 'I'), (1, 2), (0.2, 0.3, 0.4), (0.07, 0.10), (None, 0.10)):
    cand = 'D' if base in ('D', 'I') else base; mix = base == 'I'
    label = f'{base} k{k} 락{lock * 100:.0f} 손절{stop * 100:.0f}{" 추적10" if trail else "      "}'
    res = evaluate(run, PER, cand=cand, k=k, hold=10, stop=stop, trail=trail, lock=lock, mix=mix)
    RES[label] = res; P(f'  {label:<28} | ' + ' | '.join(stats(res[pn][0]) for pn in PER))
P('\n■ 2020~2025 +30% 확률 상위 12')
for label, res in sorted(RES.items(), key=lambda kv: -(kv[1]['2020~2025'][0] >= .3).mean())[:12]:
    P(f'  {label:<28} 2020~25 +30% {(res["2020~2025"][0] >= .3).mean() * 100:4.1f}% −30% {(res["2020~2025"][0] <= -.3).mean() * 100:4.1f}% | 2011~19 +30% {(res["2011~2019"][0] >= .3).mean() * 100:4.1f}% | 2026 +30% {(res["2026"][0] >= .3).mean() * 100:4.1f}%')
P('\n■ 비용 민감도 — 대표 설정 (D k1 락30 손절10 추적10) 편도 0.30% vs 0.10% vs 0.03%')
for bc in (0.003, 0.001, 0.0003):
    run2 = make_runner(D, CANDS, CANDS['A'], buyc=bc, sellc=bc)
    res = evaluate(run2, PER, cand='D', k=1, hold=10, stop=0.10, trail=0.10, lock=0.30)
    P(f'  편도 {bc * 100:.2f}%  | ' + ' | '.join(stats(res[pn][0]) for pn in PER))
open('backtest/results/us_tail_strategies.md', 'w', encoding='utf-8').write('# 미국 꼬리 포착 전략 + 정책 격자 (backtest/lab/us/us_strategies.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
