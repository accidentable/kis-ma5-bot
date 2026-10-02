"""
밤(종가 → 다음 날 시가) 과 낮(시가 → 종가) 수익 분해 — 언제 사고 언제 팔지를 일봉으로 알 수 있는 유일한 단서

대상  코스피 시총 100 + 코스닥 시총 150 (전날 순위, 주가 1,000원↑, 상·하한가 · 최근 20일 정지/하한가 제외)
묶음  전체(대상 평균) / 급락주(−10~−20%) / 급락주 & 시장 −2%↓ / 조용한 대형주 상위 20 (회전율↓ 변동성↓ 시총↑ 신고가근접↑)
     / 거래 터진 종목 (vr ≥ 3) / 회전율 상위 10%
구간  신호일 D 이후 5거래일의 밤 · 낮을 각각 평균 (비용 전). 밤1 = D 종가 → D+1 시가, 낮1 = D+1 시가 → D+1 종가, …
     + 누적: 'D 종가 매수' 와 'D+1 시가 매수' 로 5일 들고 'D+5 종가' vs 'D+6 시가' 에 파는 네 조합의 총수익 (비용 전)
기간  2011~2019 / 2020~2025 / 2026
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, c = F.o.astype(float), F.c.astype(float); tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out
def lead(x, k): out = np.full_like(x, np.nan); out[:-k] = x[k:]; return out


pc = lag(np.where(tr, c, np.nan), 1); r1 = c / pc - 1
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
E = U & np.isfinite(r1) & ~F.limdown.astype(bool) & ~F.limup.astype(bool) & ~F.bad20.astype(bool)
oo, cc = np.where(tr, o, np.nan), np.where(tr, c, np.nan)
night = [lead(oo, k) / (cc if k == 1 else lead(cc, k - 1)) - 1 for k in range(1, 7)]      # 밤k = D+k−1 종가 → D+k 시가
day = [lead(cc, k) / lead(oo, k) - 1 for k in range(1, 7)]                                  # 낮k = D+k 시가 → D+k 종가
vr, turn1, vol60, cap, hi250 = (getattr(F, k).astype(float) for k in ('vr', 'turn1', 'vol60', 'cap', 'hi250'))


def pct_rank(x, mask):
    out = np.full((T, N), np.nan)
    for t in range(T):
        idx = np.where(mask[t] & np.isfinite(x[t]))[0]
        if len(idx) < 20: continue
        r = np.empty(len(idx)); r[np.argsort(x[t, idx])] = np.arange(len(idx)); out[t, idx] = r / (len(idx) - 1)
    return out


quality = (pct_rank(-turn1, E) + pct_rank(-lag(vol60, 1), E) + pct_rank(cap, E) + pct_rank(c / hi250, E)) / 4
q20 = np.zeros((T, N), bool)
for t in range(T):
    idx = np.where(E[t] & np.isfinite(quality[t]) & (F.max20[t] < 0.10))[0]
    if len(idx) >= 20: q20[t, idx[np.argsort(-quality[t, idx])[:20]]] = True
dip = E & (r1 <= -0.10) & (r1 > -0.20)
GROUPS = {'전체 (대상 평균)': E, '급락주 −10~−20%': dip, '급락주 & 시장 −2%↓': dip & (mkt <= -0.02)[:, None],
          '조용한 대형주 상위 20': q20, '거래 터진 종목 vr≥3': E & (vr >= 3), '회전율 상위 10%': E & (pct_rank(turn1, E) >= 0.9)}
PER = {'2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')), '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
       '2026': (C.didx(F, '20260101'), T - 8)}
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))


def avg(mask, arr, a, b):
    m = mask.copy(); m[:a] = False; m[b:] = False
    v = arr[m]; v = v[np.isfinite(v)]
    return v.mean() * 100 if len(v) >= 15 else np.nan


P('밤/낮 수익 분해 (비용 전, %) — 밤k = D+k−1 종가 → D+k 시가, 낮k = D+k 시가 → D+k 종가')
for pn, (a, b) in PER.items():
    P(f'\n[{pn}]')
    P(f'  {"묶음":<22}{"건수":>7} | ' + ' '.join(f'밤{k} 낮{k}' for k in range(1, 6)) + ' | 밤합 낮합 | 종가→D+5종가 종가→D+6시가 시가→D+5종가 시가→D+6시가')
    for gn, gm in GROUPS.items():
        m = gm.copy(); m[:a] = False; m[b:] = False
        n_ = m.sum()
        nk = [avg(gm, night[k], a, b) for k in range(5)]; dk = [avg(gm, day[k], a, b) for k in range(5)]
        tot = {
            'c5c': avg(gm, lead(cc, 5) / cc - 1, a, b), 'c6o': avg(gm, lead(oo, 6) / cc - 1, a, b),
            'o5c': avg(gm, lead(cc, 5) / lead(oo, 1) - 1, a, b), 'o6o': avg(gm, lead(oo, 6) / lead(oo, 1) - 1, a, b)}
        P(f'  {gn:<22}{n_:7d} | ' + ' '.join(f'{nk[k]:+5.2f} {dk[k]:+5.2f}' for k in range(5))
          + f' | {np.nansum(nk):+5.2f} {np.nansum(dk):+5.2f} | {tot["c5c"]:+6.2f} {tot["c6o"]:+6.2f} {tot["o5c"]:+6.2f} {tot["o6o"]:+6.2f}')

open('backtest/results/lab_overnight_split.md', 'w', encoding='utf-8').write(
    '# 밤/낮 수익 분해 (backtest/lab/fresh/overnight_split.py)\n\n정의는 스크립트 머리말 참고. 비용 전.\n\n```\n' + '\n'.join(lines) + '\n```\n')
