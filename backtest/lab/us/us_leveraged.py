"""
미국 — 레버리지 ETF 로 한 달 +30% 노리기 (대회에서 ETF·레버리지가 허용될 때만 의미 있음 — 규정 확인 필요)

① 보유만: 각 ETF 를 20일 창 동안 들고 있을 때 +20/+30% 확률, −20/−30% 확률 (비용 편도 0.30%)
② ETF 모멘텀 로테이션: 레버리지 ETF 중 20일 수익률 1위 1개, 손절 10 · 추적 10 · 보유 10 · 락 30
③ 추세 필터: 기초지수(QQQ·SOXX·SPY·IWM)가 50일선 위면 3x 롱, 아래면 현금 / 아래면 3x 인버스 (락 30)
④ 폭락일 종가에 3x 롱(TQQQ·SOXL·UPRO·TNA) 매수 → 5·10일 뒤 시가 매도
⑤ 단일종목 2x ETF (NVDL·TSLL·CONL·MSTU·AMDL…) 모멘텀 1위 (상장 뒤 구간만)
"""
import numpy as np, warnings, time, sys; warnings.filterwarnings('ignore')
from backtest.lab.us.common import US, lag, lead, lists
from backtest.lab.us.us_engine import make_runner, evaluate, stats, W

t0 = time.time(); D = US(); T, N = D.T, D.N
o, c, cf, tr = D.o, D.c, D.cf, D.tr
have = set(D.tickers)
LONG3 = [t for t in ('TQQQ', 'SOXL', 'UPRO', 'TNA', 'LABU', 'FNGU', 'TECL', 'FAS', 'NUGT', 'UDOW', 'YINN', 'KORU', 'DFEN', 'DPST', 'DRN', 'ERX', 'GUSH', 'HIBL', 'MIDU', 'NAIL', 'PILL', 'RETL', 'TPOR', 'UTSL', 'WANT', 'WEBL', 'CURE', 'DUSL', 'BNKU', 'TMF', 'BOIL', 'UCO', 'UVXY') if t in have]
INV3 = [t for t in ('SQQQ', 'SOXS', 'SPXU', 'TZA', 'LABD', 'TECS', 'FAZ', 'YANG', 'TMV', 'KOLD', 'SCO', 'SVXY') if t in have]
SINGLE = [t for t in ('NVDL', 'TSLL', 'CONL', 'MSTU', 'AMDL', 'NVDX', 'TSLT', 'AAPU', 'GGLL', 'AMZU', 'MSFU', 'NFLU') if t in have]
PER = D.periods()
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P(f'미국 레버리지 ETF — 데이터에 있는 3x 롱 {len(LONG3)}, 인버스 {len(INV3)}, 단일종목 2x {len(SINGLE)}. 비용 편도 0.30%. 열: +20%↑ · +30%↑ · −20%↓ · −30%↓ · 중앙 · 평균 (%)')
# ① 보유만 (종가 매수 → 20일째 종가 매도)
P('\n■ ① 한 달 보유만')
for t_ in LONG3[:12] + INV3[:4] + SINGLE[:6] + ['SPY', 'QQQ']:
    if t_ not in have: continue
    j = D.col(t_); r = (lead(cf[:, j], W) / cf[:, j]) * (1 - 0.003) / (1 + 0.003) - 1
    cells = []
    for pn, (a, b) in PER.items():
        v = r[a:b]; v = v[np.isfinite(v)]
        cells.append(stats(v) if len(v) >= 30 else '표본 부족'.center(36))
    P(f'  {t_:<6} | ' + ' | '.join(cells))
# ② ETF 모멘텀 로테이션
def etf_lists(names):
    m = np.zeros((T, N), bool)
    for t_ in names: m[:, D.col(t_)] = tr[:, D.col(t_)]
    return lists(m & np.isfinite(D.ret20), D.ret20)
CANDS = {'LONG': etf_lists(LONG3), 'ALL': etf_lists(LONG3 + INV3), 'SINGLE': etf_lists(SINGLE) if SINGLE else [[] for _ in range(T)]}
run = make_runner(D, CANDS)
P('\n■ ② 레버리지 ETF 모멘텀 1위 (20일 수익률) · 손절 10 · 추적 10 · 보유 10')
for cand, lock in (('LONG', None), ('LONG', 0.3), ('ALL', None), ('ALL', 0.3), ('SINGLE', None), ('SINGLE', 0.3)):
    if cand == 'SINGLE' and not SINGLE: continue
    res = evaluate(run, PER, cand=cand, k=1, hold=10, stop=0.10, trail=0.10, lock=lock)
    P(f'  {cand:<6} 락{"30" if lock else "없음"} | ' + ' | '.join(stats(res[pn][0]) for pn in PER))
# ③ 추세 필터 — 기초지수 50일선
P('\n■ ③ 추세 필터: 기초지수 50일선 위 → 3x 롱, 아래 → 현금 / 인버스 (월 창, 락 30)')
PAIRS = [('QQQ', 'TQQQ', 'SQQQ'), ('SOXX', 'SOXL', 'SOXS'), ('SPY', 'UPRO', 'SPXU'), ('IWM', 'TNA', 'TZA')]
for base, lng, inv in PAIRS:
    if not all(x in have for x in (base, lng, inv)): continue
    jb, jl, ji = D.col(base), D.col(lng), D.col(inv)
    ma50 = lag(np.vstack([np.full((49, 1), np.nan), np.convolve(np.nan_to_num(cf[:, jb]), np.ones(50) / 50, mode='valid')[:, None]]), 1)[:, 0]
    above = cf[:, jb] > ma50
    for mode in ('현금', '인버스'):
        cand = np.zeros((T, N), bool); cand[above, jl] = True
        if mode == '인버스': cand[~above & np.isfinite(ma50), ji] = True
        run3 = make_runner(D, {'X': [np.where(cand[t])[0].tolist() for t in range(T)]})
        res = evaluate(run3, PER, cand='X', k=1, hold=5, stop=0.10, trail=0.10, lock=0.30)
        P(f'  {lng}/{inv} {mode:<3} | ' + ' | '.join(stats(res[pn][0]) for pn in PER))
# ④ 폭락일 3x 롱
P('\n■ ④ 폭락일(시장 −3% & 3σ) 종가에 3x 롱 매수 → n일 뒤 시가 매도 (건별, 비용 포함)')
for t_ in ('TQQQ', 'SOXL', 'UPRO', 'TNA'):
    if t_ not in have: continue
    j = D.col(t_)
    for n in (5, 10):
        r = lead(o[:, j], n + 1) / cf[:, j] * (1 - 0.003) / (1 + 0.003) - 1
        cells = []
        for pn, (a, b) in PER.items():
            m = D.crash.copy(); m[:a] = False; m[b:] = False; v = r[m]; v = v[np.isfinite(v)]
            cells.append(f'{len(v):3d}건 평균 {v.mean() * 100:+6.1f}% 중앙 {np.median(v) * 100:+6.1f}% +20%↑ {(v >= .2).mean() * 100:3.0f}% −20%↓ {(v <= -.2).mean() * 100:3.0f}%' if len(v) >= 3 else '표본 부족'.center(48))
        P(f'  {t_:<5} {n:>2}일 | ' + ' | '.join(cells))
open('backtest/results/us_leveraged.md', 'w', encoding='utf-8').write('# 미국 레버리지 ETF 로 한 달 +30% (backtest/lab/us/us_leveraged.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
