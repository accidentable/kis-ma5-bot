"""
"시장 폭락일엔 −10% 급락주를 무조건 사라" 가 맞나 — 꼬리 위험 · 독립 사건 수 · 시장 낙폭별

신호   코스피 시총 100 + 코스닥 시총 150 · 그날 −10~−20% (하한가 제외) · 그날 대상 평균 등락 −2% 이하
매수   D+1 시가, 5거래일 보유 (D+6 종가), 비용 왕복 0.31%
본 것  건별 분포 (10·25·50·75·90 백분위, 최악) · 날짜별 평균 (같은 날 신호 한 묶음) · 날짜별로 −5% · −10% 넘게 깨진 비율
       '그날 아무 종목이나' = 같은 날 대상 전 종목을 D+1 시가에 샀을 때 → 급락주를 골라서 더 번 몫
       독립 사건 = 신호일이 5거래일 넘게 떨어지면 새 사건으로 센 수
       시장 낙폭별 −2~−3 / −3~−5 / −5~−8 / −8% 이하
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, c = F.o.astype(float), F.c.astype(float); tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
BUYC, SELLC = 0.00065, 0.00265


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out
def lead(x, k): out = np.full_like(x, np.nan); out[:-k] = x[k:]; return out


cf = ffill(np.where(tr, c, np.nan)); pc = lag(cf, 1); r1 = c / pc - 1
lim = np.where(F.dates < '20150615', 0.15, 0.30)[:, None]; limdown = r1 <= -(lim - 0.005)
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
dip = U & np.isfinite(r1) & (r1 <= -0.10) & (r1 > -0.20) & ~limdown
ent = lead(np.where(tr, o, np.nan), 1)
R5 = lead(cf, 6) / ent * (1 - SELLC) / (1 + BUYC) - 1
bench = np.nanmean(np.where(U, R5, np.nan), axis=1)
PER = {'2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')),
       '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
       '2026': (C.didx(F, '20260101'), T - 12)}


def ds(mask, a, b):
    m = mask.copy(); m[:a] = False; m[b:] = False
    tt, jj = np.where(m); v = R5[tt, jj]; ok = np.isfinite(v); tt, v = tt[ok], v[ok]
    d = defaultdict(list)
    for t, x in zip(tt, v): d[t].append(x)
    days = sorted(d); dm = np.array([np.mean(d[t]) for t in days]); cnt = [len(d[t]) for t in days]
    ep = 1 + sum(1 for i in range(1, len(days)) if days[i] - days[i - 1] > 5) if days else 0
    return v, days, dm, cnt, ep


lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P('−10~−20% (하한가 제외) & 그날 시장 −2%↓ → D+1 시가 매수 · 5일 보유')
for pn, (a, b) in PER.items():
    v, days, dm, cnt, ep = ds(dip & (mkt <= -0.02)[:, None], a, b)
    bm = np.array([bench[t] for t in days])
    P(f'\n[{pn}] {len(v)}건 · 신호일 {len(days)}일 · 독립 사건 약 {ep}번')
    P(f'  건별: 평균 {v.mean() * 100:+.2f}% 승률 {(v > 0).mean() * 100:.0f}% | 10% {np.percentile(v, 10) * 100:+.1f} 25% {np.percentile(v, 25) * 100:+.1f} '
      f'중앙 {np.median(v) * 100:+.1f} 75% {np.percentile(v, 75) * 100:+.1f} 90% {np.percentile(v, 90) * 100:+.1f} | 최악 {v.min() * 100:+.1f}%')
    P(f'  날짜별: 평균 {dm.mean() * 100:+.2f}% 플러스인 날 {(dm > 0).mean() * 100:.0f}% | 그날 아무 종목이나 샀으면 {np.nanmean(bm) * 100:+.2f}% '
      f'→ 급락주 골라서 더 번 몫 {np.nanmean(dm - bm) * 100:+.2f}%p')
    P(f'  날짜별로 −10% 넘게 깨진 날 {(dm <= -0.10).mean() * 100:.0f}% · −5% 넘게 깨진 날 {(dm <= -0.05).mean() * 100:.0f}%')
    P('  최악의 날: ' + ', '.join(f'{F.dates[days[i]]} {dm[i] * 100:+.0f}% ({cnt[i]}건)' for i in np.argsort(dm)[:5]))
    P('  최고의 날: ' + ', '.join(f'{F.dates[days[i]]} {dm[i] * 100:+.0f}% ({cnt[i]}건)' for i in np.argsort(-dm)[:5]))
    P('  시장 낙폭별:')
    for lo, hi, nm in ((-0.03, -0.02, '−2~−3%'), (-0.05, -0.03, '−3~−5%'), (-0.08, -0.05, '−5~−8%'), (-9, -0.08, '−8% 이하')):
        v2, d2, dm2, _, _ = ds(dip & ((mkt > lo) & (mkt <= hi))[:, None], a, b)
        if len(v2) < 10: P(f'    {nm:<8} 표본 부족 ({len(v2)}건)'); continue
        P(f'    {nm:<8} {len(v2):4d}건 건별 {v2.mean() * 100:+6.2f}% 승 {(v2 > 0).mean() * 100:3.0f}% | 날짜별 {dm2.mean() * 100:+6.2f}% ({len(d2)}일, 플러스인 날 {(dm2 > 0).mean() * 100:.0f}%)')

open('backtest/results/lab_crash_dip_check.md', 'w', encoding='utf-8').write(
    '# 시장 폭락일 급락주 매수 — 꼬리 위험 점검 (backtest/lab/fresh/crash_dip_check.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
