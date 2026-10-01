"""
급등 · 급락 종목을 언제 사야 하나 — 이벤트 스터디 (코스피 시총 100 + 코스닥 시총 150, 전날 순위, 상장폐지 포함)

신호 (D = 신호일, 등락률 = 종가 / 전날 종가 − 1)
  하루   D 등락률이 −3~−5 · −5~−10 · −10% 이하(하한가 제외) · 하한가 / +3~+5 · +5~+10 · +10% 이상(상한가 제외) · 상한가
  이틀   D−1, D 모두 −5% 이하 · 모두 +5% 이상
매수 시점: D 종가 · D+1 시가 · D+1 종가 · D+2 시가 · D+2 종가
보유: 매수 뒤 N 거래일 (종가 매수 → N일 뒤 종가, 시가 매수 → N일 뒤 시가 매도), N = 1 · 3 · 5 · 10
지표: 건당 순수익 (비용 0.31% 차감), 초과수익 (같은 날 같은 방식으로 유니버스 전 종목을 샀을 때 대비), 승률,
      t = 날짜별 평균의 t값 (같은 날 여러 건이 몰리는 걸 한 건으로 묶음)
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, c = F.o.astype(float), F.c.astype(float)
pc = np.vstack([np.full((1, N), np.nan), c[:-1]])
r1 = c / pc - 1
U = F.traded & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
COST = 0.00015 * 2 + 0.002 + 0.0005 * 2


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


cf = ffill(np.where(F.traded, c, np.nan))
of = np.where(F.traded, o, cf)                     # 거래정지일 시가 = 마지막 종가


def shift(x, k):
    out = np.full_like(x, np.nan)
    if k < T: out[:T - k] = x[k:]
    return out


# 매수 시점: (이름, 신호일로부터 며칠 뒤, 가격 종류)
ENTRIES = [('D 종가', 0, 'c'), ('D+1 시가', 1, 'o'), ('D+1 종가', 1, 'c'), ('D+2 시가', 2, 'o'), ('D+2 종가', 2, 'c')]
HOLDS = (1, 3, 5, 10)
RET, EXC = {}, {}
for name, k, kind in ENTRIES:
    raw_px = (o if kind == 'o' else c)
    ent = np.where(shift(F.traded, k).astype(bool), shift(raw_px, k), np.nan)   # 매수일에 거래돼야 함
    exitpx = of if kind == 'o' else cf
    for n in HOLDS:
        r = shift(exitpx, k + n) / ent - 1 - COST
        bench = np.nanmean(np.where(U, r, np.nan), axis=1, keepdims=True)
        RET[name, n], EXC[name, n] = r, r - bench

prev = np.vstack([np.full((1, N), np.nan), r1[:-1]])
LIM = 0.295
SIG = {
    '−3~−5%': (r1 <= -0.03) & (r1 > -0.05),
    '−5~−10%': (r1 <= -0.05) & (r1 > -0.10),
    '−10% 이하': (r1 <= -0.10) & (r1 > -LIM),
    '하한가': r1 <= -LIM,
    '이틀 연속 −5%↓': (r1 <= -0.05) & (prev <= -0.05) & (r1 > -LIM),
    '+3~+5%': (r1 >= 0.03) & (r1 < 0.05),
    '+5~+10%': (r1 >= 0.05) & (r1 < 0.10),
    '+10% 이상': (r1 >= 0.10) & (r1 < LIM),
    '상한가': r1 >= LIM,
    '이틀 연속 +5%↑': (r1 >= 0.05) & (prev >= 0.05) & (r1 < LIM),
}
PER = {'2011~2019': (C.didx(F, '20110101'), C.didx(F, '20200101')), '2020~': (C.didx(F, '20200101'), T - 15)}


def stats(mask, a, b, M):
    mm = mask.copy(); mm[:a] = False; mm[b:] = False
    v = M[mm]; ok = np.isfinite(v)
    if ok.sum() < 30: return None
    tt, jj = np.where(mm)
    v = v[ok]; tt = tt[ok]
    days = np.unique(tt)
    daily = np.array([M[t][mm[t] & np.isfinite(M[t])].mean() for t in days])
    tval = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 2 else 0
    return v.mean(), np.median(v), (v > 0).mean(), len(v), tval, len(days)


rows = []
for sname, s in SIG.items():
    m = s & U & np.isfinite(r1)
    print(f'\n■ {sname}')
    for pn, (a, b) in PER.items():
        mm = m.copy(); mm[:a] = False; mm[b:] = False
        nd = (b - a); cnt = mm.sum(); dd = mm[a:b].any(1).sum()
        print(f'  [{pn}] {cnt:,}건 · 하루 평균 {cnt / nd:.1f}건 · 신호 있는 날 {dd / nd * 100:.0f}%')
        print(f'    {"매수":<9}' + ''.join(f'| {n}일 보유: 순수익 / 초과 / 승률 / t       ' for n in HOLDS))
        for name, _, _ in ENTRIES:
            line = f'    {name:<9}'
            for n in HOLDS:
                st = stats(m, a, b, RET[name, n]); se = stats(m, a, b, EXC[name, n])
                if st is None: line += '| -' + ' ' * 38; continue
                line += f'| {st[0] * 100:+6.2f}% {se[0] * 100:+6.2f}% {st[2] * 100:3.0f}% {se[4]:+5.1f}    '
                rows.append(dict(sig=sname, per=pn, entry=name, hold=n, ret=st[0], exc=se[0], med=st[1], win=st[2],
                                 n=st[3], t_exc=se[4], days=st[5]))
            print(line, flush=True)

import json, os
os.makedirs('backtest/results', exist_ok=True)
json.dump(rows, open('backtest/results/lab_move_timing.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
