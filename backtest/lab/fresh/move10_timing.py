"""
하루 ±10% 움직인 종목 — 언제 사나 (하락 · 상승을 같은 틀로) + 한 달에 6번 넘게 거래할 만큼 신호가 나오나

대상   코스피 시총 100 + 코스닥 시총 150 (전날 시총 순위, 상장폐지 포함, 주가 1,000원↑)
신호   하락  D 종가 −10% 이하 −20% 초과, 하한가 제외   (drop10_timing.py 와 같음)
       상승  D 종가 +10% 이상 +20% 미만, 상한가 제외   (상한가는 종가에 사기 어려워서 크기별 표에서 따로 본다)

매수 시점
  D 장중 ±10%   장중에 전날 종가 ±10% 에 닿으면 그 가격에 산다 (시가부터 넘어 있으면 시가).
                신호 = '장중에 닿음' 만으로 정한다. 그날 종가(하한가 · −20% 아래 마감 등)는 모르는 시점이라 쓰지 않는다.
  D 종가 · D+1 시가 · D+1 종가 · D+2 시가
  상승 쪽은 상한가에 묶인 가격으로는 못 산다고 본다: 시가가 상한가면 그 시가 매수, 종가가 상한가면 그 종가 매수는 뺀다.
다음 날(D+1) 움직임 (D 종가 대비)
  같은 방향 5%   하락 쪽 = D+1 에 −5% 더 빠짐, 상승 쪽 = D+1 에 +5% 더 오름 (장중 닿을 때 · D+1 종가 · D+2 시가)
  반대 방향 5%   하락 쪽 = D+1 에 +5% 반등, 상승 쪽 = D+1 에 −5% 눌림 (D+1 종가에 산다)
  나머지         D+1 이 ±5% 안 (D+1 종가에 산다)
보유   매수한 날 종가 기준 N 거래일 뒤 종가 (N = 1 · 3 · 5 · 10). 비용 왕복 0.31%
지표   평균 · 중앙 · 승률 · 날짜평균(같은 날 신호를 한 묶음) · t · 초과(같은 날 대상 전 종목을 같은 방식으로 산 것 대비)
       장중 매수의 '초과' 기준은 그날 시가 매수라 다른 행과 바로 비교하면 안 된다
시장   대상 전 종목의 그날 평균 등락 (±30% 자름): −2%↓ / −2~+2% / +2%↑
빈도   20거래일 한 달 동안 살 수 있는 신호 수 (첫날 아침에 아는 전날 신호부터 마지막 전날 신호까지)
기간   2011~2019 / 2020~2025 / 2026
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, h, l, c = (getattr(F, k).astype(float) for k in 'ohlc')
tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
BUYC, SELLC = 0.00015 + 0.0005, 0.00015 + 0.002 + 0.0005
HOLDS = (1, 3, 5, 10)
MONTH = 20


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


def lag(x, k):
    out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


def lead(x, k):
    out = np.full_like(x, np.nan); out[:-k] = x[k:]; return out


def net(entry, exitpx):
    return exitpx / entry * (1 - SELLC) / (1 + BUYC) - 1


cf = ffill(np.where(tr, c, np.nan))
pc = lag(cf, 1)
r1 = c / pc - 1
lim = np.where(F.dates < '20150615', 0.15, 0.30)[:, None]
limdown = r1 <= -(lim - 0.005)
limup = r1 >= lim - 0.005
open_limup = o >= pc * (1 + lim - 0.005)
mkt = np.nanmean(np.where(U, np.clip(r1, -0.3, 0.3), np.nan), axis=1)
d1 = lead(cf, 1) / cf - 1                          # D+1 종가 / D 종가
tr1 = lead(tr.astype(float), 1) > 0


def rets(k, px, ok):
    ent = np.where(ok, px, np.nan)
    ent = lead(ent, k) if k else ent
    return {n: net(ent, lead(cf, k + n)) for n in HOLDS}


def bench_of(R):
    return {n: np.nanmean(np.where(U, R[n], np.nan), axis=1) for n in HOLDS}


BENCH = {'시가': bench_of(rets(0, o, tr)), '종가': bench_of(rets(0, c, tr))}
BENCH['D 종가'] = BENCH['종가']
BENCH['D+1 시가'] = bench_of(rets(1, o, tr)); BENCH['D+1 종가'] = bench_of(rets(1, c, tr)); BENCH['D+2 시가'] = bench_of(rets(2, o, tr))
BENCH['D 장중'] = BENCH['시가']; BENCH['D+1 장중'] = BENCH['D+1 시가']


def side_data(side):
    """side = −1 (하락) · +1 (상승). 반환: 신호 · 매수 시점별 수익 · 장중 매수 신호"""
    if side < 0:
        sig = U & np.isfinite(r1) & (r1 <= -0.10) & (r1 > -0.20) & ~limdown
        ok_open, ok_close = tr, tr
    else:
        sig = U & np.isfinite(r1) & (r1 >= 0.10) & (r1 < 0.20) & ~limup
        ok_open, ok_close = tr & ~open_limup, tr & ~limup
    R = {'D 종가': rets(0, c, ok_close), 'D+1 시가': rets(1, o, ok_open), 'D+1 종가': rets(1, c, ok_close),
         'D+2 시가': rets(2, o, ok_open)}
    # D 장중 ±10% (전날 종가 기준)
    thr = pc * (1 + 0.10 * side)
    if side < 0:
        hit = tr & np.isfinite(pc) & (l <= thr); px = np.where(o <= thr, o, thr)
    else:
        hit = tr & np.isfinite(pc) & (h >= thr) & ~open_limup; px = np.where(o >= thr, o, thr)
    R['D 장중'] = {n: net(np.where(hit, px, np.nan), lead(cf, n)) for n in HOLDS}
    touch_sig = U & hit
    # D+1 장중 같은 방향 5% (D 종가 기준)
    o1, l1, h1 = lead(o, 1), lead(l, 1), lead(h, 1)
    thr1 = cf * (1 + 0.05 * side)
    if side < 0:
        hit1 = tr1 & (l1 <= thr1); px1 = np.where(o1 <= thr1, o1, thr1)
    else:
        hit1 = tr1 & (h1 >= thr1) & ~(lead(open_limup.astype(float), 1) > 0); px1 = np.where(o1 >= thr1, o1, thr1)
    R['D+1 장중'] = {n: net(np.where(hit1, px1, np.nan), lead(cf, 1 + n)) for n in HOLDS}
    return sig, touch_sig, hit1, R


PER = {
    '2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')),
    '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
    '2026': (C.didx(F, '20260101'), T - 12),
}


def pm(m, a, b):
    mm = m.copy(); mm[:a] = False; mm[b:] = False; return mm


def stats(mask, R, B):
    tt, jj = np.where(mask)
    v = R[tt, jj]; ok = np.isfinite(v)
    tt, v = tt[ok], v[ok]
    if len(v) < 15: return None
    d = defaultdict(list)
    for t, x in zip(tt, v): d[t].append(x)
    dm = np.array([np.mean(x) for x in d.values()])
    tv = dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 2 else np.nan
    return dict(n=len(v), days=len(dm), mean=v.mean(), med=np.median(v), win=(v > 0).mean(), dmean=dm.mean(), t=tv,
                exc=np.nanmean(v - B[tt]))


lines = []
P = lambda s='': (lines.append(s), print(s, flush=True))
HDR = f'    {"매수":<16}{"보유":>4} | {"건수":>5} {"신호일":>5} | {"평균":>7} {"중앙":>7} {"승률":>5} | {"날짜평균":>7} {"t":>5} | {"초과":>7}'


def fmt(name, n, s):
    if s is None: return f'    {name:<16}{n:>3}일 | 표본 부족'
    return (f'    {name:<16}{n:>3}일 | {s["n"]:5d} {s["days"]:5d} | {s["mean"] * 100:+6.2f}% {s["med"] * 100:+6.2f}% '
            f'{s["win"] * 100:4.0f}% | {s["dmean"] * 100:+6.2f}% {s["t"]:+5.1f} | {s["exc"] * 100:+6.2f}%')


def short(s):
    return '표본 부족' if s is None else f'{s["n"]:4d}건 {s["mean"] * 100:+6.2f}% (날짜 {s["dmean"] * 100:+5.2f}, 초과 {s["exc"] * 100:+5.2f})'


def table(title, rows, R):
    P(f'\n■ {title}')
    for pn, (a, b) in PER.items():
        P(f'  [{pn}]'); P(HDR)
        for name, mask, ent, bench in rows:
            for n in HOLDS:
                P(fmt(name, n, stats(pm(mask, a, b), R[ent][n], BENCH[bench][n])))


def freq(mask, a, b):
    """시작일 s 의 한 달(s..s+19)에 살 수 있는 신호 수 = 신호일 s−1 .. s+18"""
    cnt = mask.sum(1).astype(float)
    cs = np.concatenate([[0], np.cumsum(cnt)])
    n = np.array([cs[s + MONTH - 1] - cs[s - 1] for s in range(max(a, 1), min(b, T - MONTH))])
    return n


SIDES = {'하락': -1, '상승': +1}
DATA = {nm: side_data(sd) for nm, sd in SIDES.items()}

for nm, sd in SIDES.items():
    sig, touch_sig, hit1, R = DATA[nm]
    same = '−5% 더 빠짐' if sd < 0 else '+5% 더 오름'
    opp = '+5% 반등' if sd < 0 else '−5% 눌림'
    lbl = '−10~−20% (하한가 제외)' if sd < 0 else '+10~+20% (상한가 제외)'
    P(f'\n\n━━━━━━━━━━ {nm}: 하루 {lbl} ━━━━━━━━━━')
    for pn, (a, b) in PER.items():
        m = pm(sig, a, b)
        P(f'  {pn}: 신호 {m.sum():,}건 · 신호 있는 날 {m.any(1).sum()}일 / {b - a}일 · 하루 최대 {m.sum(1).max()}건')

    table(f'{nm} 1. 당일에 살까, 다음 날 살까', [
        (f'D 장중 {"−" if sd < 0 else "+"}10%', touch_sig, 'D 장중', 'D 장중'),
        ('D 종가', sig, 'D 종가', 'D 종가'), ('D+1 시가', sig, 'D+1 시가', 'D+1 시가'), ('D+1 종가', sig, 'D+1 종가', 'D+1 종가')], R)

    P(f'\n■ {nm} 1-1. 크기별 · 그날 시장별 — 5일 보유 (평균 / 날짜평균 / 초과, 건수)')
    a_ = np.abs(r1)
    ext = (U & np.isfinite(r1) & (np.sign(r1) == sd) & (a_ >= 0.20) & ~(limdown if sd < 0 else limup))
    limm = U & (limdown if sd < 0 else limup)
    SUB = [('10~12%', sig & (a_ < 0.12)), ('12~15%', sig & (a_ >= 0.12) & (a_ < 0.15)), ('15~20%', sig & (a_ >= 0.15)),
           ('20%~가격제한 전', ext), ('하한가' if sd < 0 else '상한가 (종가 매수 불가)', limm),
           ('시장 −2%↓', sig & (mkt <= -0.02)[:, None]), ('시장 −2~+2%', sig & (np.abs(mkt) < 0.02)[:, None]),
           ('시장 +2%↑', sig & (mkt >= 0.02)[:, None])]
    for pn, (a, b) in PER.items():
        P(f'  [{pn}]')
        for sn, sm in SUB:
            P(f'    {sn:<18}' + ''.join(f'| {ent} {short(stats(pm(sm, a, b), R[ent][5], BENCH[ent][5]))} '
                                         for ent in ('D 종가', 'D+1 시가', 'D+1 종가')))

    SAME = sig & tr1 & ((d1 <= -0.05) if sd < 0 else (d1 >= 0.05))
    SAME_NL = SAME & ~((lead(limdown.astype(float), 1) > 0) if sd < 0 else (lead(limup.astype(float), 1) > 0))
    OPP = sig & tr1 & ((d1 >= 0.05) if sd < 0 else (d1 <= -0.05))
    REST = sig & tr1 & (np.abs(d1) < 0.05)
    for pn, (a, b) in PER.items():
        P(f'  {pn}: D+1 장중 {same[:3]} 닿음 {pm(sig & hit1, a, b).sum():,}건 · D+1 종가 {same} {pm(SAME, a, b).sum():,}건 '
          f'(가격제한 마감 {pm(SAME & ~SAME_NL, a, b).sum()}건) · {opp} {pm(OPP, a, b).sum():,}건 · 나머지 {pm(REST, a, b).sum():,}건')
    table(f'{nm} 2. 다음 날(D+1) 움직임에 따라 (D 종가 대비)', [
        (f'D+1 장중 {same[:3]}', sig & hit1, 'D+1 장중', 'D+1 장중'),
        (f'D+1 종가 {same[:3]}', SAME_NL, 'D+1 종가', 'D+1 종가'),
        ('└ D+2 시가', SAME, 'D+2 시가', 'D+2 시가'),
        (f'D+1 종가 {opp}', OPP, 'D+1 종가', 'D+1 종가'),
        ('(비교) 나머지', REST, 'D+1 종가', 'D+1 종가')], R)

    P(f'\n■ {nm} 3. 연도별 — 5일 보유 (건수 · 평균 · 초과)')
    years = sorted({d[:4] for d in F.dates if '2011' <= d[:4]})
    for name, mask, ent in (('D+1 시가 (전체)', sig, 'D+1 시가'), (f'D+1 종가 {same}', SAME_NL, 'D+1 종가')):
        P(f'  {name}')
        cells = []
        for y in years:
            a, b = C.didx(F, f'{y}0101'), min(C.didx(F, f'{int(y) + 1}0101'), T - 12)
            s = stats(pm(mask, a, b), R[ent][5], BENCH[ent][5])
            cells.append(f'{y} ' + ('-' if s is None else f'{s["n"]}건 {s["mean"] * 100:+.1f}% ({s["exc"] * 100:+.1f})'))
        for i in range(0, len(cells), 4): P('    ' + ' | '.join(cells[i:i + 4]))

P('\n\n━━━━━━━━━━ 빈도: 한 달(20거래일)에 살 수 있는 신호 수 — 평균 / 중앙 / 6건↑ 달 / 1건↑ 달 ━━━━━━━━━━')
dn, up = DATA['하락'][0], DATA['상승'][0]
FQ = [('하락 전체', dn), ('하락 · 시장 −2%↓', dn & (mkt <= -0.02)[:, None]), ('하락 · 시장 −2% 위', dn & (mkt > -0.02)[:, None]),
      ('상승 전체', up), ('상승 · 시장 +2%↑', up & (mkt >= 0.02)[:, None]), ('상승 · 시장 +2% 아래', up & (mkt < 0.02)[:, None]),
      ('하락 + 상승 합', dn | up)]
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for fn, fm in FQ:
        n = freq(fm, a, b)
        P(f'    {fn:<18} 평균 {n.mean():6.1f}건  중앙 {np.median(n):5.0f}건  6건↑ {(n >= 6).mean() * 100:4.0f}%  1건↑ {(n >= 1).mean() * 100:4.0f}%')

txt = '\n'.join(lines)
open('backtest/results/lab_move10_timing.md', 'w', encoding='utf-8').write(
    '# 하루 ±10% 움직인 종목 — 언제 사나 · 한 달 신호 수 (backtest/lab/fresh/move10_timing.py)\n\n'
    '정의는 스크립트 머리말 참고. 날짜평균 = 같은 날 신호를 한 묶음으로 본 평균, 초과 = 같은 날 대상 전 종목 대비.\n\n```\n'
    + txt + '\n```\n')
