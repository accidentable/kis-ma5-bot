"""
모멘텀 종목을 '실적 발표 직전' 에 사서 발표를 넘겨 보유 — 대회 시뮬 (20일 창, 매 거래일 시작, 비용 편도 0.30%).
발표일은 실제론 2~4주 전에 공지되므로 '다음 실적까지 n 거래일' 은 그날 알 수 있는 정보로 본다 (events.pkl 의 실현 날짜 사용).
후보(그날 종가 매수, 점수 높은 순): M60 = 60일 수익률 상위 10%, M20 = 20일 수익률 상위 10%, pre5 = 다음 실적이 1~5 거래일 안.
열: 거래한 달 % · +20%↑ · +30%↑ · +50%↑ · −20%↓ · −30%↓ · 중앙 · 평균 (전체 달 기준, %)
"""
import numpy as np, pandas as pd, warnings, time, sys; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, lists, pct_mask
from backtest.lab.us.us_engine import make_runner

t0 = time.time(); D = US(); T, N = D.T, D.N; tick = {t: i for i, t in enumerate(D.tickers)}
E = pd.read_pickle('/data/lab/us/earnings/events.pkl'); E['j'] = E.sym.map(tick)
days_to = np.full((T, N), 99, np.int16)
for r, j in zip(E.r.values.astype(int), E.j.values.astype(int)):
    lo = max(r - 10, 0); seg = days_to[lo:r, j]; days_to[lo:r, j] = np.minimum(seg, (r - np.arange(lo, r)).astype(np.int16))
pre5 = (days_to >= 1) & (days_to <= 5); pre10 = (days_to >= 1) & (days_to <= 10); post = np.zeros((T, N), bool)
for r, j in zip(E.r.values.astype(int), E.j.values.astype(int)): post[r:min(r + 20, T), j] = True     # 발표 뒤 20일 (실적 직후 제외용)
EU = D.U & np.isfinite(D.r1) & np.isfinite(D.ret60)
M60 = pct_mask(D.ret60, EU, 0.9, 1.0); M20 = pct_mask(D.ret20, EU & np.isfinite(D.ret20), 0.9, 1.0); HV = D.vol60 >= .04
CANDS = {
    'M60 상위10% (기준, 실적 무관)': lists(M60, D.ret60),
    'M60 & 실적 1~5일 안': lists(M60 & pre5, D.ret60),
    'M60 & 실적 1~10일 안': lists(M60 & pre10, D.ret60),
    'M60 & 실적 5일 안 & 변동성≥4%': lists(M60 & pre5 & HV, D.ret60),
    'M60 & 실적 뒤 20일 아님': lists(M60 & ~post, D.ret60),
    'M20 상위10% (기준)': lists(M20, D.ret20),
    'M20 & 실적 1~5일 안': lists(M20 & pre5, D.ret20),
    'ret60≥30% & 실적 5일 안': lists(EU & (D.ret60 >= .3) & pre5, D.ret60),
    '실적 5일 안 (모멘텀 무관, 변동성 큰 순)': lists(EU & pre5, D.vol60),
    '실적 5일 안 & 변동성≥4% (ret60 순)': lists(EU & pre5 & HV, D.ret60),
    '시총500 M60 & 실적 5일 안': lists(M60 & pre5 & (D.caprank <= 500)[None, :], D.ret60),
}
PER = D.periods(); run = make_runner(D, CANDS)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
def stats(r):
    tr_ = r != 0
    return (f'{tr_.mean() * 100:5.1f} {(r >= .2).mean() * 100:5.1f} {(r >= .3).mean() * 100:5.1f} {(r >= .5).mean() * 100:5.1f} '
            f'{(r <= -.2).mean() * 100:5.1f} {(r <= -.3).mean() * 100:5.1f} {np.median(r) * 100:+5.1f} {r.mean() * 100:+5.1f}')
HDR = '  ' + ' | '.join(f'{pn:^52}' for pn in PER); SUB = '  ' + ' | '.join(f'{"거래%":>5} {"+20":>5} {"+30":>5} {"+50":>5} {"-20":>5} {"-30":>5} {"중앙":>5} {"평균":>5}' for _ in PER)
def block(title, **kw):
    P(f'\n■ {title}'); P(f'  {"후보":<34}' + HDR); P(f'  {"":<34}' + SUB)
    for name in CANDS:
        cells = []
        for pn, (a, b) in PER.items():
            r = np.array([run(s, name, **kw)[0] for s in range(a, b)]); cells.append(stats(r))
        P(f'  {name:<34}  ' + ' | '.join(cells))
block('k1 · 보유 20 · 락 30 · 손절 없음', k=1, hold=20, stop=None, trail=None, lock=0.3)
block('k1 · 보유 20 · 락 30 · 손절 10', k=1, hold=20, stop=0.10, trail=None, lock=0.3)
block('k1 · 보유 10 · 락 30 · 손절 10 · 추적 10 (발표 뒤 5일쯤 교체)', k=1, hold=10, stop=0.10, trail=0.10, lock=0.3)
block('k1 · 보유 20 · 락 50 · 손절 15', k=1, hold=20, stop=0.15, trail=None, lock=0.5)
block('k2 · 보유 20 · 락 30 · 손절 10', k=2, hold=20, stop=0.10, trail=None, lock=0.3)
open('backtest/results/us_earnings_momo.md', 'w', encoding='utf-8').write('# 미국 모멘텀 × 실적 직전 매수 대회 시뮬 (backtest/lab/us/earn_momo.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
