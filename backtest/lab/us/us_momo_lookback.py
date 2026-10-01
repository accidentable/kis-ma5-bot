"""
미국 모멘텀 집중 — 기준 기간(20·60·120·250일)과 시총 그룹별 비교. k1, 그날 종가 매수, 보유 20, 락 30, 손절 없음 (us_strategies 의 'D 모멘텀 집중' 은 20일 기준·보유 10·손절 10 이었다).
열: +20%↑ · +30%↑ · +50%↑ · −20%↓ · −30%↓ · 중앙 · 평균 (%)
"""
import numpy as np, warnings, time, sys; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, lists, pct_mask, lag
from backtest.lab.us.us_engine import make_runner

t0 = time.time(); D = US(); T, N = D.T, D.N
EU = D.U & np.isfinite(D.r1)
ret120 = D.cf / lag(D.cf, 120) - 1; ret250 = D.cf / lag(D.cf, 250) - 1; ret250x = lag(D.cf, 20) / lag(D.cf, 250) - 1   # 12-1개월
cap500 = (D.caprank <= 500)[None, :]
CANDS = {}
for name, sc in (('20일', D.ret20), ('60일', D.ret60), ('120일', ret120), ('250일', ret250), ('250-20일(12-1개월)', ret250x)):
    m = pct_mask(sc, EU & np.isfinite(sc), 0.9, 1.0)
    CANDS[f'{name} 상위10% 전체'] = lists(m, sc); CANDS[f'{name} 상위10% 시총500'] = lists(m & cap500, sc); CANDS[f'{name} 상위10% 시총501~'] = lists(m & ~cap500, sc)
CANDS['60일 상위10% 변동성 큰 순'] = lists(pct_mask(D.ret60, EU & np.isfinite(D.ret60), 0.9, 1.0), D.vol60)
CANDS['60일 상위 5% (ret60 순)'] = lists(pct_mask(D.ret60, EU & np.isfinite(D.ret60), 0.95, 1.0), D.ret60)
CANDS['60일 상위 10~20% (ret60 순)'] = lists(pct_mask(D.ret60, EU & np.isfinite(D.ret60), 0.8, 0.9), D.ret60)
PER = D.periods(); run = make_runner(D, CANDS)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
def stats(r): return f'{(r >= .2).mean() * 100:5.1f} {(r >= .3).mean() * 100:5.1f} {(r >= .5).mean() * 100:5.1f} {(r <= -.2).mean() * 100:5.1f} {(r <= -.3).mean() * 100:5.1f} {np.median(r) * 100:+5.1f} {r.mean() * 100:+5.1f}'
for title, kw in (('k1 · 보유 20 · 락 30 · 손절 없음', dict(k=1, hold=20, stop=None, trail=None, lock=0.3)), ('k1 · 보유 10 · 락 30 · 손절 10 · 추적 10', dict(k=1, hold=10, stop=0.10, trail=0.10, lock=0.3)),
                  ('k1 · 보유 20 · 락 30 · 손절 15 → 현금', dict(k=1, hold=20, stop=0.15, trail=None, lock=0.3, max_entries=1))):
    P(f'\n■ {title}'); P(f'  {"후보":<28}' + ' | '.join(f'{pn:^45}' for pn in PER)); P(f'  {"":<28}' + ' | '.join(f'{"+20":>5} {"+30":>5} {"+50":>5} {"-20":>5} {"-30":>5} {"중앙":>5} {"평균":>5}' for _ in PER))
    for name in CANDS:
        cells = [stats(np.array([run(s, name, **kw)[0] for s in range(a, b)])) for pn, (a, b) in PER.items()]
        P(f'  {name:<28}' + ' | '.join(cells))
open('backtest/results/us_momo_lookback.md', 'w', encoding='utf-8').write('# 미국 모멘텀 집중 — 기준 기간·시총 비교 (backtest/lab/us/us_momo_lookback.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
