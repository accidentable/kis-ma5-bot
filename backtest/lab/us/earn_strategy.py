"""
실적 반응 뒤 매수 — 대회 시뮬 (20일 창, 매 거래일 시작, 락 30/50, 비용 편도 0.30%).
후보: 반응일 R 에 규칙을 만족한 종목을 그날 종가(또는 R+1 시가)에 산다. k 슬롯, 보유 hold 일, 손절, 락.
열: 거래한 달 % · +20%↑ · +30%↑ · +50%↑ · −20%↓ · −30%↓ · 중앙 · 평균 (전체 달 기준, %)
"""
import numpy as np, pandas as pd, warnings, time, sys, itertools; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US
from backtest.lab.us.us_engine import make_runner

t0 = time.time(); D = US(); T = D.T; tick = {t: i for i, t in enumerate(D.tickers)}
E = pd.read_pickle('/data/lab/us/earnings/events.pkl'); E['j'] = E.sym.map(tick)
E = E[(E.caprank.notna()) & (D.raw[E.r.values, E.j.values] >= 5)]
UP = E.dayret >= .1; DN = E.dayret <= -.1
RULES = {
    'U10 급등 ≥+10% (큰 순)': (UP, 'dayret', True),
    'U15 급등 ≥+15%': (E.dayret >= .15, 'dayret', True),
    'U20 급등 ≥+20%': (E.dayret >= .2, 'dayret', True),
    'U10 & 서프+': (UP & (E.surprise > 0), 'dayret', True),
    'U10 & 강하게 마감(intra>0)': (UP & (E.intra > 0), 'dayret', True),
    'U10 & 52주 고가 −10% 안': (UP & (E.hi250 >= -.1), 'dayret', True),
    'U10 & 거래 5배↑': (UP & (E.vr >= 5), 'dayret', True),
    'U10 & 시총 501~': (UP & (E.caprank > 500), 'dayret', True),
    'U10 & 시총 500 안': (UP & (E.caprank <= 500), 'dayret', True),
    'U10 & 변동성 ≥4%': (UP & (E.vol60 >= .04), 'dayret', True),
    'U10 서프 큰 순': (UP, 'surprise', True),
    'D10 급락 ≤−10% (큰 순)': (DN, 'dayret', False),
    'D15 급락 ≤−15%': (E.dayret <= -.15, 'dayret', False),
    'D20 급락 ≤−20%': (E.dayret <= -.2, 'dayret', False),
    'D10 & 서프+ (셀더뉴스)': (DN & (E.surprise > 0), 'dayret', False),
    'D10 & 낙폭 회복(intra>0)': (DN & (E.intra > 0), 'dayret', False),
    'D10 & 시총 500 안': (DN & (E.caprank <= 500), 'dayret', False),
    'D10 & 시총 501~': (DN & (E.caprank > 500), 'dayret', False),
    'D10 & 52주 고가 −40% 밖': (DN & (E.hi250 < -.4), 'dayret', False),
    'U10 변동성 큰 순': (UP, 'vol60', True),
    'U10 & 60일 −20%↓ (턴어라운드)': (UP & (E.ret60 <= -.2), 'dayret', True),
    'U10 & 52주 고가 −20% 밖': (UP & (E.hi250 < -.2), 'dayret', True),
    'U10 & 직전 분기 −5%↓': (UP & (E.prev_dayret <= -.05), 'dayret', True),
    'D10 & 60일 −20%↓ (투매)': (DN & (E.ret60 <= -.2), 'dayret', False),
    'D10 & 2일 더 −5%↓ → R+2 매수': (DN & (E.c2 <= -.05), 'dayret', False, 2),
    'D10 & 다음날 −3%↓ → R+1 매수': (DN & (E.c1 <= -.03), 'dayret', False, 1),
    'U10 & 2일 −3%↓ 눌림 → R+2 매수': (UP & (E.c2 <= -.03), 'dayret', True, 2),
    'D10 & 60일 −20%↓ ret60 낮은 순': (DN & (E.ret60 <= -.2), 'ret60', False),
    'D10 & 60일 −20%↓ 완만한 낙폭 순': (DN & (E.ret60 <= -.2), 'dayret', True),
    'D10 & 52주 −40% 밖 완만한 낙폭 순': (DN & (E.hi250 < -.4), 'dayret', True),
    'D10~−20% & 60일 −20%↓ (−20% 넘는 폭락 제외)': (DN & (E.dayret > -.2) & (E.ret60 <= -.2), 'dayret', True),
    'U10 & 60일 −20%↓ ret60 낮은 순': (UP & (E.ret60 <= -.2), 'ret60', False),
    'U10 거래 배수 큰 순': (UP, 'vr', True),
}
def cand_list(mask, key, desc, offset=0):
    out = [[] for _ in range(T)]
    sub = E[mask & E[key].notna()].sort_values(key, ascending=not desc)
    for r, j in zip(sub.r.values, sub.j.values):
        if int(r) + offset < T: out[int(r) + offset].append(int(j))
    return out
CANDS = {name: cand_list(*v) for name, v in RULES.items()}
PER = D.periods(); run = make_runner(D, CANDS)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
def stats(r):
    tr_ = r != 0
    return (f'{tr_.mean() * 100:5.1f} {(r >= .2).mean() * 100:5.1f} {(r >= .3).mean() * 100:5.1f} {(r >= .5).mean() * 100:5.1f} '
            f'{(r <= -.2).mean() * 100:5.1f} {(r <= -.3).mean() * 100:5.1f} {np.median(r) * 100:+5.1f} {r.mean() * 100:+5.1f}')
HDR = '  ' + ' | '.join(f'{pn:^52}' for pn in PER); SUB = '  ' + ' | '.join(f'{"거래%":>5} {"+20":>5} {"+30":>5} {"+50":>5} {"-20":>5} {"-30":>5} {"중앙":>5} {"평균":>5}' for _ in PER)
def block(title, names, **kw):
    P(f'\n■ {title}'); P(f'  {"규칙":<30}' + HDR); P(f'  {"":<30}' + SUB)
    for name in names:
        cells = []
        for pn, (a, b) in PER.items():
            r = np.array([run(s, name, **kw)[0] for s in range(a, b)]); cells.append(stats(r))
        P(f'  {name:<30}  ' + ' | '.join(cells))
names = list(RULES)
block('k1 · 보유 20 · 락 30 · 손절 없음 · R 종가 매수', names, k=1, hold=20, stop=None, trail=None, lock=0.3)
block('k1 · 보유 20 · 락 50 · 손절 15 · R 종가 매수', names, k=1, hold=20, stop=0.15, trail=None, lock=0.5)
block('k2 · 보유 20 · 락 30 · 손절 15 · R 종가 매수', names, k=2, hold=20, stop=0.15, trail=None, lock=0.3)
block('k1 · 보유 10 · 락 30 · 손절 10 · 추적 10 · R 종가 매수', names, k=1, hold=10, stop=0.10, trail=0.10, lock=0.3)
block('k1 · 보유 20 · 락 30 · 손절 없음 · R+1 시가 매수', names[:3] + names[11:14], k=1, hold=20, stop=None, trail=None, lock=0.3, buy_at='open')
block('k1 · 보유 20 · 락 30 · 손절 15 → 현금(재진입 없음)', names, k=1, hold=20, stop=0.15, trail=None, lock=0.3, max_entries=1)
SUB3 = [n for n in names if n.startswith('D10') or n.startswith('D15') or n.startswith('D20') or n in ('U10 급등 ≥+10% (큰 순)', 'U10 변동성 큰 순', 'U10 & 60일 −20%↓ (턴어라운드)', 'U10 & 52주 고가 −20% 밖')]
block('k3 · 보유 20 · 락 30 · 손절 없음 · R 종가 매수 (분산)', SUB3, k=3, hold=20, stop=None, trail=None, lock=0.3)
open('backtest/results/us_earnings_strategy.md', 'w', encoding='utf-8').write('# 미국 실적 반응 매수 대회 시뮬 (backtest/lab/us/earn_strategy.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
