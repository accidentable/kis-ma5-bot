"""
미국 — 대표 설정의 달력 달(매달 첫 거래일 시작, 20거래일) 결과: 연도별 +20/+30/−20 달 수, 평균, 최고 달, +30% 달에 산 종목
  R   20일 수익률 하위 10% 반전 1종목 · 락30 · 손절 없음
  E   고변동 모멘텀 1종목 · 락30 · 손절 10
  L   대형주(시총 500) 모멘텀 1종목 · 락30 · 손절 10
  ETF 레버리지 ETF(롱+인버스) 20일 모멘텀 1위 · 락30 · 손절 10 · 추적 10
  SOXL 보유만 (락 30)
"""
import numpy as np, warnings, time, sys; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, lists, pct_mask
from backtest.lab.us.us_engine import make_runner, W

t0 = time.time(); D = US(); T, N = D.T, D.N; r1, U, tr = D.r1, D.U, D.tr
E = U & np.isfinite(r1)
have = set(D.tickers)
LEV = [t for t in D.tickers[D.is_lev]]
def etf_lists(names):
    m = np.zeros((T, N), bool)
    for t_ in names: m[:, D.col(t_)] = tr[:, D.col(t_)]
    return lists(m & np.isfinite(D.ret20), D.ret20)
soxl = np.zeros((T, N), bool); soxl[:, D.col('SOXL')] = tr[:, D.col('SOXL')]
CANDS = {'R': lists(pct_mask(D.ret20, E, 0.0, 0.1), D.ret20, desc=False),
         'E': lists(pct_mask(D.vol60, E, 0.9, 1.0) & pct_mask(D.ret20, E, 0.5, 1.0), D.ret20),
         'L': lists(pct_mask(D.ret20, E, 0.9, 1.0) & (D.caprank <= 500)[None, :], D.ret20),
         'ETF': etf_lists(LEV), 'SOXL': [np.where(soxl[t])[0].tolist() for t in range(T)]}
CFG = {'R 반전 k1 락30': dict(cand='R', k=1, hold=10, stop=None, trail=None, lock=0.3),
       'E 고변동모멘텀 k1 락30': dict(cand='E', k=1, hold=10, stop=0.10, trail=None, lock=0.3),
       'L 대형주모멘텀 k1 락30': dict(cand='L', k=1, hold=10, stop=0.10, trail=None, lock=0.3),
       'ETF 로테이션 락30': dict(cand='ETF', k=1, hold=10, stop=0.10, trail=0.10, lock=0.3),
       'SOXL 보유 락30': dict(cand='SOXL', k=1, hold=25, stop=None, trail=None, lock=0.3)}
# 종목 이름을 남기는 러너
names_bought = {}
def runner_with_names(cfg_name, **kw):
    run = make_runner(D, CANDS)
    return run
months = {}
for t in range(D.didx('20110103'), T - W - 1):
    ym = str(D.dates[t])[:6]
    if ym not in months: months[ym] = t
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
run = make_runner(D, CANDS)
for name, kw in CFG.items():
    rows = {ym: run(s, **kw)[0] for ym, s in months.items()}
    P(f'\n■ {name} — 달력 달 {len(rows)}개: +30% {sum(v >= .3 for v in rows.values())}달 ({sum(v >= .3 for v in rows.values()) / len(rows):.0%}), +20% {sum(v >= .2 for v in rows.values())}달, −20% {sum(v <= -.2 for v in rows.values())}달, −30% {sum(v <= -.3 for v in rows.values())}달')
    P('  연도  달수 +20 +30 −20 −30   평균   최고달')
    for y in range(2011, 2027):
        ys = {ym: v for ym, v in rows.items() if ym.startswith(str(y))}
        if not ys: continue
        r = np.array(list(ys.values())); best = max(ys.items(), key=lambda kv: kv[1])
        P(f'  {y}  {len(r):3d}  {(r >= .2).sum():3d} {(r >= .3).sum():3d} {(r <= -.2).sum():3d} {(r <= -.3).sum():3d}  {r.mean() * 100:+5.1f}%  {best[0]} {best[1] * 100:+.0f}%')
# +30% 달에 산 종목 (R · ETF) — 첫 매수 종목만 기록
P('\n■ +30% 달의 첫 매수 종목 (R 반전 · ETF 로테이션, 최근 20개)')
for name in ('R 반전 k1 락30', 'ETF 로테이션 락30'):
    kw = CFG[name]; hits = []
    for ym, s in months.items():
        r = run(s, **kw)[0]
        if r >= .3:
            first = next((D.tickers[j] for t in range(s, s + W) for j in CANDS[kw['cand']][t][:1]), '?')
            hits.append(f'{ym} {first} {r * 100:+.0f}%')
    P(f'  {name}: ' + ', '.join(hits[-20:]))
open('backtest/results/us_yearly.md', 'w', encoding='utf-8').write('# 미국 대표 설정 연도별 (backtest/lab/us/us_yearly.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
