"""
한 달 안에 +30% 가는 기회는 얼마나 있고, 어떤 모습으로 시작하나 — 꼬리 기회 목록 (대회용 리서치 Phase A)

대상   코스피 시총 200 + 코스닥 시총 150 (전날 순위, 상장폐지 포함, 주가 1,000원↑, 거래됨) ≈ 대회 지수 종목
창     어느 날 종가에 사서 20거래일 안에 (1) 최고 종가가 +20 / +30 / +50% 를 넘는 확률 (2) 20일째 종가가 +30% 넘는 확률 (3) 최저 종가가 −20% 아래로 가는 확률
       최고 종가 기준이라 '그 가격에 팔 수 있었다' 는 뜻은 아니다 — 기회의 존재 여부만 본다. 비용 전.
입구 신호 (그날 종가까지 정보, 결과 보기 전에 정함)
  전체 / 폭락일(시장 −3% & −3σ) 급락 −7%↓ / 평소 날 급락 −10%↓ / 20일 신고가 돌파 & 거래 2배↑ / 52주 고가 근접 (종가 ≥ 250일 최고 × 0.98)
  / 상한가 마감 / +10~20% 급등 마감 / 20일 수익률 상위 10% / 하위 10% / 60일 베타 상위 10% / 거래대금 20일 평균 3배↑ / 3일 연속 상승
  / 시장 강세 (시장 20일 수익률 > +5%) / 시장 약세 (< −5%) / 60일 변동성 상위 10% (고변동 종목)
또   매달(20일 창)마다 대상 중 '종가→20일 내 최고 종가 +30%↑' 종목 수의 분포 — 기회가 아예 없는 달이 있나
기간 2011~2019 / 2020~2025 / 2026
"""
import numpy as np, warnings, time; warnings.filterwarnings('ignore')
from numpy.lib.stride_tricks import sliding_window_view as swv
from backtest.lab import panic2_feat as PF, panic2_common as C

t0 = time.time()
F = PF.load(); T, N = F.T, F.N
o, h, l, c = (getattr(F, k).astype(float) for k in 'ohlc'); tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 200)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
W = 20


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


cf = np.where(tr, c, np.nan)
for t in range(1, T):
    m = np.isnan(cf[t]); cf[t, m] = cf[t - 1, m]
pc = lag(cf, 1); r1 = c / pc - 1
# 앞으로 20일 최고 · 최저 · 20일째 종가 (t+1..t+20)
pad = np.vstack([cf, np.full((W, N), np.nan)])
fwd = swv(pad, W + 1, axis=0)[:, :, 1:]                    # [t, j, k] = cf[t+1+k]
fmax = np.nanmax(fwd, axis=2) / cf - 1
fmin = np.nanmin(fwd, axis=2) / cf - 1
fend = fwd[:, :, -1] / cf - 1
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
idx = np.cumprod(1 + np.nan_to_num(mkt)); m20 = idx / lag(idx[:, None], 20)[:, 0] - 1
sig20 = np.full(T, np.nan)
for t in range(21, T): sig20[t] = np.nanstd(mkt[t - 20:t])
T5 = (mkt <= -0.03) & (mkt <= -3 * sig20)
limup, limdown, bad20 = F.limup.astype(bool), F.limdown.astype(bool), F.bad20.astype(bool)
ret20, hi250, vr, bstar, vol60, max20 = (getattr(F, k).astype(float) for k in ('ret20', 'hi250', 'vr', 'bstar', 'vol60', 'max20'))
pad20 = np.vstack([np.full((19, N), np.nan), cf]); hi20_prev = lag(np.nanmax(swv(pad20, 20, axis=0), axis=-1), 1)
up = np.nan_to_num(r1) > 0; ups = np.zeros((T, N), int)
for t in range(1, T): ups[t] = np.where(up[t], ups[t - 1] + 1, 0)


def pct_mask(x, lo, hi):
    out = np.zeros((T, N), bool)
    for t in range(T):
        ii = np.where(U[t] & np.isfinite(x[t]))[0]
        if len(ii) < 20: continue
        order = ii[np.argsort(x[t, ii])]; n = len(order); out[t, order[int(n * lo):int(n * hi)]] = True
    return out


E = U & np.isfinite(r1) & ~bad20
SIG = {
    '전체 (대상 전 종목)': E,
    '폭락일 급락 −7%↓': E & T5[:, None] & (r1 <= -0.07) & ~limdown,
    '평소 날 급락 −10%↓': E & ~T5[:, None] & (r1 <= -0.10) & ~limdown,
    '20일 신고가 돌파 & 거래 2배↑': E & (c > hi20_prev) & (vr >= 2),
    '52주 고가 근접 (≥98%)': E & (c / hi250 >= 0.98),
    '상한가 마감': U & limup,
    '+10~20% 급등 마감': E & (r1 >= 0.10) & (r1 < 0.20) & ~limup,
    '20일 수익률 상위 10%': E & pct_mask(ret20, 0.9, 1.0),
    '20일 수익률 하위 10%': E & pct_mask(ret20, 0.0, 0.1),
    '베타 상위 10%': E & pct_mask(bstar, 0.9, 1.0),
    '60일 변동성 상위 10%': E & pct_mask(lag(vol60, 1), 0.9, 1.0),
    '거래대금 3배↑ (방향 무관)': E & (vr >= 3),
    '3일 연속 상승': E & (ups >= 3),
    '시장 강세 (20일 > +5%)': E & (m20 > 0.05)[:, None],
    '시장 약세 (20일 < −5%)': E & (m20 < -0.05)[:, None],
    '시장 약세 & 20일 수익률 상위 10%': E & (m20 < -0.05)[:, None] & pct_mask(ret20, 0.9, 1.0),
}
PER = {'2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')), '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
       '2026': (C.didx(F, '20260101'), T - W - 1)}
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P('한 달(20거래일) 꼬리 기회 목록 — 종가 매수 기준, 비용 전. 열: 건수 | 최고가 +20%↑ / +30%↑ / +50%↑ 확률 | 20일째 +30%↑ | 최저가 −20%↓ | 20일째 평균')
for pn, (a, b) in PER.items():
    P(f'\n[{pn}]')
    for sn, m in SIG.items():
        mm = m.copy(); mm[:a] = False; mm[b:] = False
        v = fmax[mm]; ok = np.isfinite(v); v = v[ok]; e = fend[mm][ok]; mn = fmin[mm][ok]
        if len(v) < 30: P(f'  {sn:<28} 표본 부족 ({len(v)})'); continue
        P(f'  {sn:<28} {len(v):7,d} | {(v >= .2).mean() * 100:5.1f}% {(v >= .3).mean() * 100:5.1f}% {(v >= .5).mean() * 100:5.1f}% | {(e >= .3).mean() * 100:5.1f}% | {(mn <= -.2).mean() * 100:5.1f}% | {e.mean() * 100:+5.2f}%')
P('\n■ 매달(20일 창, 시작일 하루씩) 대상 중 20일 내 최고 종가 +30%↑ 종목 수 — 평균 / 중앙 / 0개인 달 / 5개↑ 달 / 20개↑ 달 ; 20일째 종가 +30%↑ 종목 수 평균')
for pn, (a, b) in PER.items():
    cnt = np.array([(E[s] & (fmax[s] >= .3)).sum() for s in range(a, b)]); cnt_e = np.array([(E[s] & (fend[s] >= .3)).sum() for s in range(a, b)])
    P(f'  {pn}: 평균 {cnt.mean():5.1f} 중앙 {np.median(cnt):4.0f} | 0개 {(cnt == 0).mean() * 100:4.1f}% 5개↑ {(cnt >= 5).mean() * 100:4.0f}% 20개↑ {(cnt >= 20).mean() * 100:4.0f}% | 20일째 +30%↑ 평균 {cnt_e.mean():5.1f}개 (대상 {E[a:b].sum(1).mean():.0f}종목 중)')
P('\n■ 시장 상태별: 그 달 시장(대상 동일가중) 20일 수익률 구간 → +30% 기회 종목 수 평균, 10개↑ 달 비율')
mf = np.array([np.prod(1 + np.nan_to_num(mkt[s + 1:s + W + 1])) - 1 for s in range(T - W - 1)])
for lo, hi, nm in ((-9, -0.08, '시장 −8%↓'), (-0.08, -0.03, '−8~−3%'), (-0.03, 0.03, '−3~+3%'), (0.03, 0.08, '+3~+8%'), (0.08, 9, '+8%↑')):
    ss = [s for s in range(C.didx(F, '20110103'), T - W - 1) if lo < mf[s] <= hi]
    cnt = np.array([(E[s] & (fmax[s] >= .3)).sum() for s in ss])
    P(f'  {nm:<10} {len(ss):4d}달 | +30% 기회 평균 {cnt.mean():5.1f}개 · 10개↑ 달 {(cnt >= 10).mean() * 100:3.0f}% · 0개 달 {(cnt == 0).mean() * 100:3.0f}%')
open('backtest/results/lab_tail_inventory.md', 'w', encoding='utf-8').write(
    '# 한 달 꼬리 기회 목록 (backtest/lab/fresh/tail_inventory.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s')
