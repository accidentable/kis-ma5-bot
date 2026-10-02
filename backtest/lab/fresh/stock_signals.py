"""
종목 자체 신호 14개 — 시장과 상관없이 단기(1~10일)에 먹히는 게 있나 (이벤트 스터디)

대상   코스피 시총 100 + 코스닥 시총 150 (전날 순위, 상장폐지 포함, 주가 1,000원↑)
공통 제외  그날 상한가 · 하한가 마감, 최근 20일 안에 거래정지 · 하한가가 있었던 종목 (관리 · 경고 종목 대용)
매수   D+1 시가 (봇이 09:05 에 사는 것과 같음) — 기본. D 종가도 5일 보유만 같이 본다
보유   매수한 날 종가 기준 N 거래일 뒤 종가 (N = 1 · 3 · 5 · 10). 비용 왕복 0.31%
시장을 빼는 법  '초과' = 같은 날 대상 전 종목을 같은 방식으로 샀을 때 대비. 날짜별로 묶어서(같은 날 신호 = 한 묶음) 평균 · t 를 낸다.
                 그리고 신호일 시장 상태(조용 |±1%| · 하락 −1%↓ · 상승 +1%↑)로 나눠 시장과 무관한지 본다

신호 (전부 그 종목 데이터만으로 정의. 결과 보기 전에 정했다)
  되돌림 (약세 뒤 매수)
    R1  종목 충격        z ≤ −2.5  (z = 베타 뺀 잔차 수익 / 그 종목의 60일 잔차 변동폭 — 시장 탓이 아닌, 평소의 2.5배 넘는 하락)
    R2  R1 & 조용한 거래   거래대금이 20일 평균의 1.5배 미만 (뉴스 없는 하락 → 되돌림 기대)
    R3  R1 & 거래 폭발     거래대금이 20일 평균의 3배 이상 (뉴스 있는 하락 → 되돌림 안 될 것. 대조군)
    R4  R1 & 저가 마감     종가 위치(IBS) ≤ 0.2
    R5  R1 & 고가 회복     종가 위치(IBS) ≥ 0.6 (장중 반전)
    R6  연속 하락 3일↑     그동안 잔차 누적 ≤ −8%
    R7  20일선 아래 −2ATR  (종가 − 20일선) / ATR14 ≤ −2 (자기 변동성 대비 많이 눌림)
    R8  RSI(2) ≤ 10       고전적 단기 과매도
    R9  갭다운 후 반전      시가 갭 ≤ −3% 이고 시가 → 종가 ≥ +2%
    R10 상승 추세 속 눌림   60일 수익률 > 0 이면서 5일 수익률이 그날 대상 중 하위 10%
  추세 (강세 뒤 매수)
    M1  종목 강세 & 조용    z ≥ +2.5, 거래대금 < 20일 평균 × 1.5
    M2  20일 신고가 돌파    종가 > 직전 20일 최고 종가, 거래대금 ≥ 20일 평균 × 2
    M3  갭업 후 강세        시가 갭 ≥ +3% 이고 시가 → 종가 ≥ +2%
    M4  5일 연속 상승
통과 기준 (미리 정함)  D+1 시가 · 5일 보유의 날짜별 초과수익이 세 기간 모두 > 0 이고, 2011~2019 · 2020~2025 둘 다 t ≥ 2,
                       그리고 한 달(20거래일)에 살 수 있는 신호가 6건 이상인 달이 두 기간 모두 80% 이상
기간   2011~2019 / 2020~2025 / 2026
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view as swv
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, h, l, c = (getattr(F, k).astype(float) for k in 'ohlc')
tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
BUYC, SELLC = 0.00065, 0.00265
HOLDS = (1, 3, 5, 10); MONTH = 20


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out
def lead(x, k): out = np.full_like(x, np.nan); out[:-k] = x[k:]; return out
def net(entry, exitpx): return exitpx / entry * (1 - SELLC) / (1 + BUYC) - 1


cf = ffill(np.where(tr, c, np.nan)); pc = lag(cf, 1); r1 = c / pc - 1
limdown, limup = F.limdown.astype(bool), F.limup.astype(bool)
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
E = U & np.isfinite(r1) & ~limdown & ~limup & ~F.bad20.astype(bool)

z, vr, ibs, gap, intra = (getattr(F, k).astype(float) for k in ('z', 'vr', 'ibs', 'gap', 'intra'))
nd, icum, ddatr, rsi2 = F.down_days.astype(int), F.icum.astype(float), F.dd_atr.astype(float), F.rsi2.astype(float)
ret5, ret60 = F.ret5.astype(float), F.ret60.astype(float)
pad = np.vstack([np.full((19, N), np.nan), cf]); hi20_prev = lag(np.nanmax(swv(pad, 20, axis=0), axis=-1), 1)
up = np.nan_to_num(r1) > 0; ups = np.zeros((T, N), int)
for t in range(1, T): ups[t] = np.where(up[t], ups[t - 1] + 1, 0)
# 5일 수익률 그날 대상 중 하위 10%
low10 = np.zeros((T, N), bool)
for t in range(T):
    idx = np.where(U[t] & np.isfinite(ret5[t]))[0]
    if len(idx) < 20: continue
    k = max(1, len(idx) // 10); low10[t, idx[np.argsort(ret5[t, idx])[:k]]] = True

R1 = E & (z <= -2.5)
SIGS = {
    'R1 종목 충격 z≤−2.5': R1,
    'R2 R1 & 조용 (vr<1.5)': R1 & (vr < 1.5),
    'R3 R1 & 거래폭발 (vr≥3)': R1 & (vr >= 3),
    'R4 R1 & 저가마감 ibs≤.2': R1 & (ibs <= 0.2),
    'R5 R1 & 고가회복 ibs≥.6': R1 & (ibs >= 0.6),
    'R6 3일↑ 연속하락 잔차≤−8%': E & (nd >= 3) & (icum <= -0.08),
    'R7 20일선 −2ATR 아래': E & (ddatr <= -2),
    'R8 RSI2 ≤ 10': E & (rsi2 <= 10),
    'R9 갭다운−3% 후 반전+2%': E & (gap <= -0.03) & (intra >= 0.02),
    'R10 상승추세 속 눌림': E & (ret60 > 0) & low10,
    'M1 종목 강세 z≥2.5 & 조용': E & (z >= 2.5) & (vr < 1.5),
    'M2 20일 신고가 & vr≥2': E & (c > hi20_prev) & (vr >= 2),
    'M3 갭업+3% 후 강세+2%': E & (gap >= 0.03) & (intra >= 0.02),
    'M4 5일 연속 상승': E & (ups >= 5),
}


def rets(k, px):
    ent = np.where(tr, px, np.nan); ent = lead(ent, k) if k else ent
    return {n: net(ent, lead(cf, k + n)) for n in HOLDS}


RET = {'D+1 시가': rets(1, o), 'D 종가': rets(0, c)}
BENCH = {e: {n: np.nanmean(np.where(U, R[n], np.nan), axis=1) for n in HOLDS} for e, R in RET.items()}
PER = {'2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')),
       '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
       '2026': (C.didx(F, '20260101'), T - 12)}


def pm(m, a, b): mm = m.copy(); mm[:a] = False; mm[b:] = False; return mm


def stats(mask, R, B):
    tt, jj = np.where(mask); v = R[tt, jj]; ok = np.isfinite(v); tt, v = tt[ok], v[ok]
    if len(v) < 15: return None
    ex = v - B[tt]
    d = defaultdict(list); dx = defaultdict(list)
    for t, x, y in zip(tt, v, ex): d[t].append(x); dx[t].append(y)
    dm = np.array([np.mean(x) for x in d.values()]); de = np.array([np.mean(x) for x in dx.values()])
    tv = de.mean() / (de.std(ddof=1) / np.sqrt(len(de))) if len(de) > 2 else np.nan
    return dict(n=len(v), days=len(dm), mean=v.mean(), med=np.median(v), win=(v > 0).mean(), dmean=dm.mean(), dexc=de.mean(), t=tv)


def freq(mask, a, b):
    cnt = mask.sum(1).astype(float); cs = np.concatenate([[0], np.cumsum(cnt)])
    return np.array([cs[s + MONTH - 1] - cs[s - 1] for s in range(max(a, 1), min(b, T - MONTH))])


lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P('종목 자체 신호 14개 — D+1 시가 매수 · 비용 왕복 0.31% · 초과 = 같은 날 대상 전 종목 대비 (날짜별 묶음)')

P('\n■ 1. 신호 빈도 — 건수 · 신호일 · 한 달(20거래일) 평균 건수 · 6건↑ 달 비율')
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for sn, m in SIGS.items():
        mm = pm(m, a, b); fq = freq(m, a, b)
        P(f'    {sn:<26} {mm.sum():6,d}건 {mm.any(1).sum():5d}일 | 한 달 {fq.mean():6.1f}건 · 6건↑ {(fq >= 6).mean() * 100:4.0f}%')

P('\n■ 2. 수익 — D+1 시가 매수 1 · 3 · 5 · 10일, D 종가 매수 5일')
P(f'    {"신호":<26}{"매수·보유":<12} | {"건수":>6} | {"평균":>7} {"중앙":>7} {"승률":>4} | {"날짜평균":>7} {"날짜초과":>7} {"t":>5}')
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for sn, m in SIGS.items():
        for ent, n in (('D+1 시가', 1), ('D+1 시가', 3), ('D+1 시가', 5), ('D+1 시가', 10), ('D 종가', 5)):
            s = stats(pm(m, a, b), RET[ent][n], BENCH[ent][n])
            lab_ = f'{ent} {n:>2}일'
            if s is None: P(f'    {sn:<26}{lab_:<12} | 표본 부족'); continue
            P(f'    {sn:<26}{lab_:<12} | {s["n"]:6d} | {s["mean"] * 100:+6.2f}% {s["med"] * 100:+6.2f}% {s["win"] * 100:3.0f}% '
              f'| {s["dmean"] * 100:+6.2f}% {s["dexc"] * 100:+6.2f}% {s["t"]:+5.1f}')

P('\n■ 3. 시장 상태별 — D+1 시가 매수 · 5일 (건수 · 평균 · 날짜초과 · t). 시장과 무관하면 세 칸이 비슷해야 한다')
MK = [('조용 |±1%|', np.abs(mkt) < 0.01), ('하락 −1%↓', mkt <= -0.01), ('상승 +1%↑', mkt >= 0.01)]
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for sn, m in SIGS.items():
        line = f'    {sn:<26}'
        for mn, mm in MK:
            s = stats(pm(m & mm[:, None], a, b), RET['D+1 시가'][5], BENCH['D+1 시가'][5])
            line += f'| {mn} ' + ('표본 부족            ' if s is None else f'{s["n"]:5d}건 {s["mean"] * 100:+6.2f}% 초과 {s["dexc"] * 100:+5.2f} t{s["t"]:+4.1f} ')
        P(line)

P('\n■ 4. 통과 판정 (D+1 시가 · 5일: 날짜별 초과 > 0 세 기간 모두, t ≥ 2 두 기간, 6건↑ 달 ≥ 80% 두 기간)')
for sn, m in SIGS.items():
    row = {}
    for pn, (a, b) in PER.items():
        s = stats(pm(m, a, b), RET['D+1 시가'][5], BENCH['D+1 시가'][5])
        row[pn] = (s['dexc'], s['t']) if s else (np.nan, np.nan)
    f1 = (freq(m, *PER['2011~2019']) >= 6).mean(); f2 = (freq(m, *PER['2020~2025']) >= 6).mean()
    ok_exc = all(np.nan_to_num(row[p][0], nan=-1) > 0 for p in PER)
    ok_t = all(np.nan_to_num(row[p][1], nan=-9) >= 2 for p in ('2011~2019', '2020~2025'))
    ok_f = f1 >= 0.8 and f2 >= 0.8
    P(f'  {sn:<26} 초과 ' + ' '.join(f'{row[p][0] * 100:+5.2f}(t{row[p][1]:+4.1f})' for p in PER)
      + f' | 6건↑ {f1 * 100:3.0f}% {f2 * 100:3.0f}% | ' + ('통과' if ok_exc and ok_t and ok_f else
                                                       '탈락: ' + ', '.join(x for x, ok in (('초과', ok_exc), ('t', ok_t), ('빈도', ok_f)) if not ok)))

open('backtest/results/lab_stock_signals.md', 'w', encoding='utf-8').write(
    '# 종목 자체 신호 14개 — 단기 이벤트 스터디 (backtest/lab/fresh/stock_signals.py)\n\n정의 · 통과 기준은 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
