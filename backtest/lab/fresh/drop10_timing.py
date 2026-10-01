"""
하루 −10~−20% 빠진 종목, 언제 사야 하나 + 다음 날 −5% 더 빠지면 사기 (drop5_timing.py 와 같은 틀)

대상   코스피 시총 100 + 코스닥 시총 150 (전날 시총 순위, 상장폐지 포함, 주가 1,000원↑)
신호   D 종가가 전날 종가 대비 −10% 이하 −20% 초과, 하한가 마감 제외 (2015-06-15 전 가격제한 15% 시절 포함)

1. 매수 시점   D 장중 −10% (저가가 −10% 에 닿으면 그 가격, 시가부터 아래면 시가. 신호 = '장중 −10% 닿음' 만으로 정한다 —
               그날 종가(하한가 · −20% 아래 마감)는 사는 시점엔 모르는 정보라 쓰지 않는다)
               D 종가 · D+1 시가 · D+1 종가
2. 다음 날 −5% 추가 하락 (D 종가 대비)
     D+1 장중 −5%   D+1 저가가 D 종가 −5% 에 닿으면 그 가격 (시가부터 아래면 시가). 닿은 종목 전부 (종가가 회복했어도)
     D+1 종가        D+1 종가가 D 종가 −5% 이하일 때 그 종가
     D+2 시가        위와 같은 종목을 다음 날 시가
   비교: D+1 이 −5% 까지는 안 빠진 종목을 D+1 종가에 산 경우
보유   매수한 날 종가 기준 N 거래일 뒤 종가 (N = 1 · 3 · 5 · 10). 비용 왕복 0.31%
지표   평균 · 중앙 · 승률 · 날짜평균(같은 날 신호를 한 묶음) · t · 초과(같은 날 대상 전 종목을 같은 방식으로 산 것 대비)
       장중 매수의 '초과' 기준은 D(또는 D+1) 시가 매수라 다른 행과 바로 비교하면 안 된다
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
mkt = np.nanmean(np.where(U, np.clip(r1, -0.3, 0.3), np.nan), axis=1)       # 대상 전 종목 그날 평균 등락

band = np.isfinite(r1) & (r1 > -0.20) & ~limdown
SIG = U & band & (r1 <= -0.10)
SIG_TOUCH = U & np.isfinite(pc) & (l <= pc * 0.90)
touch_px = np.where(o <= pc * 0.90, o, pc * 0.90)

# 보유 N 일 수익 (신호일 t 에 맞춘 T × N). k = 매수일 오프셋, px = 그날 매수 가격 (T × N, 매수일 기준)
def rets(k, px, ok=None):
    ok = tr if ok is None else ok
    ent = np.where(ok, px, np.nan)
    ent = lead(ent, k) if k else ent
    return {n: net(ent, lead(cf, k + n)) for n in HOLDS}


RET = {
    'D 장중 −10%': rets(0, touch_px, SIG_TOUCH),
    'D 종가': rets(0, c),
    'D+1 시가': rets(1, o),
    'D+1 종가': rets(1, c),
    'D+2 시가': rets(2, o),
}
# D+1 장중 −5% (D 종가 대비): D+1 저가가 D 종가 × 0.95 에 닿으면
o1, l1, c1, tr1 = lead(o, 1), lead(l, 1), lead(c, 1), lead(tr.astype(float), 1) > 0
touch1 = tr1 & (l1 <= cf * 0.95)
px1 = np.where(o1 <= cf * 0.95, o1, cf * 0.95)
RET['D+1 장중 −5%'] = {n: net(np.where(touch1, px1, np.nan), lead(cf, 1 + n)) for n in HOLDS}


def bench_of(R):
    return {n: np.nanmean(np.where(U, R[n], np.nan), axis=1) for n in HOLDS}


BENCH = {k: bench_of(v) for k, v in RET.items() if '장중' not in k}
BENCH['D 장중 −10%'] = bench_of(rets(0, o))
BENCH['D+1 장중 −5%'] = bench_of(rets(1, o))

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
HDR = f'    {"매수":<12}{"보유":>4} | {"건수":>5} {"신호일":>5} | {"평균":>7} {"중앙":>7} {"승률":>5} | {"날짜평균":>7} {"t":>5} | {"초과":>7}'


def fmt(name, n, s):
    if s is None: return f'    {name:<12}{n:>3}일 | 표본 부족'
    return (f'    {name:<12}{n:>3}일 | {s["n"]:5d} {s["days"]:5d} | {s["mean"] * 100:+6.2f}% {s["med"] * 100:+6.2f}% '
            f'{s["win"] * 100:4.0f}% | {s["dmean"] * 100:+6.2f}% {s["t"]:+5.1f} | {s["exc"] * 100:+6.2f}%')


def table(title, rows):
    """rows: [(이름, 마스크, 매수 시점 이름)]"""
    P(f'\n■ {title}')
    for pn, (a, b) in PER.items():
        P(f'  [{pn}]'); P(HDR)
        for name, mask, ent in rows:
            for n in HOLDS:
                P(fmt(name, n, stats(pm(mask, a, b), RET[ent][n], BENCH[ent][n])))


P('하루 −10~−20% 하락 종목 (하한가 제외) — 코스피 시총 100 + 코스닥 시총 150, 비용 왕복 0.31%')
for pn, (a, b) in PER.items():
    m = pm(SIG, a, b)
    P(f'  {pn}: 신호 {m.sum():,}건 · 신호 있는 날 {m.any(1).sum()}일 / {b - a}일 · 하루 최대 {m.sum(1).max()}건')

table('1. 당일에 살까, 다음 날 살까',
      [('D 장중 −10%', SIG_TOUCH, 'D 장중 −10%'), ('D 종가', SIG, 'D 종가'), ('D+1 시가', SIG, 'D+1 시가'), ('D+1 종가', SIG, 'D+1 종가')])

P('\n■ 1-1. 낙폭 크기별 · 그날 시장 — 5일 보유 (평균 / 날짜평균 / 초과, 건수)')
SUB = [('−10~−12%', r1 > -0.12), ('−12~−15%', (r1 <= -0.12) & (r1 > -0.15)), ('−15~−20%', r1 <= -0.15),
       ('시장 −2%↓ 같이 빠진 날', (mkt <= -0.02)[:, None] & np.ones((1, N), bool)),
       ('시장 −2% 위 (혼자 빠짐)', (mkt > -0.02)[:, None] & np.ones((1, N), bool))]
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for sn, sm in SUB:
        line = f'    {sn:<20}'
        for ent in ('D 종가', 'D+1 시가', 'D+1 종가'):
            s = stats(pm(SIG & sm, a, b), RET[ent][5], BENCH[ent][5])
            line += f'| {ent} ' + ('표본 부족 ' if s is None else
                                    f'{s["n"]:4d}건 {s["mean"] * 100:+6.2f}% (날짜 {s["dmean"] * 100:+5.2f}, 초과 {s["exc"] * 100:+5.2f}) ')
        P(line)

d1 = lead(cf, 1) / cf - 1
D1_DN5 = SIG & tr1 & (d1 <= -0.05)
D1_LIM = lead(limdown.astype(float), 1) > 0
D1_TOUCH = SIG & touch1
D1_REST = SIG & tr1 & (d1 > -0.05)
P('\n■ 2. 다음 날(D+1) −5% 더 빠지면 사기 (D 종가 대비)')
for pn, (a, b) in PER.items():
    P(f'  {pn}: D+1 장중 −5% 닿음 {pm(D1_TOUCH, a, b).sum():,}건 · D+1 종가 −5%↓ {pm(D1_DN5, a, b).sum():,}건 '
      f'(그중 D+1 하한가 {pm(D1_DN5 & D1_LIM, a, b).sum()}건) · D+1 −5% 위 {pm(D1_REST, a, b).sum():,}건')
table('2. D+1 −5% 추가 하락 매수 vs 안 빠진 종목',
      [('D+1 장중 −5%', D1_TOUCH, 'D+1 장중 −5%'),
       ('D+1 종가−5%↓', D1_DN5, 'D+1 종가'),
       ('└ 하한가 빼고', D1_DN5 & ~D1_LIM, 'D+1 종가'),
       ('D+2 시가', D1_DN5, 'D+2 시가'),
       ('(비교) 안빠짐', D1_REST, 'D+1 종가')])

P('\n■ 3. 연도별 — 5일 보유 (건수 · 평균 · 초과). 몇 해에 몰렸는지 확인')
years = sorted({d[:4] for d in F.dates if '2011' <= d[:4]})
for name, mask, ent in (('D+1 시가 (전체)', SIG, 'D+1 시가'), ('D+1 종가−5%↓ 하한가 뺌', D1_DN5 & ~D1_LIM, 'D+1 종가')):
    P(f'  {name}')
    cells = []
    for y in years:
        a, b = C.didx(F, f'{y}0101'), min(C.didx(F, f'{int(y) + 1}0101'), T - 12)
        s = stats(pm(mask, a, b), RET[ent][5], BENCH[ent][5])
        cells.append(f'{y} ' + ('-' if s is None else f'{s["n"]}건 {s["mean"] * 100:+.1f}% ({s["exc"] * 100:+.1f})'))
    for i in range(0, len(cells), 4): P('    ' + ' | '.join(cells[i:i + 4]))

txt = '\n'.join(lines)
open('backtest/results/lab_drop10_timing.md', 'w', encoding='utf-8').write(
    '# 하루 −10~−20% 하락 종목 — 언제 사나 · 다음 날 −5% 더 빠지면 (backtest/lab/fresh/drop10_timing.py)\n\n'
    '정의는 스크립트 머리말 참고. 날짜평균 = 같은 날 신호를 한 묶음으로 본 평균, 초과 = 같은 날 대상 전 종목 대비.\n\n```\n'
    + txt + '\n```\n')
