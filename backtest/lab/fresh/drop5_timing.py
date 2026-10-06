"""
하루 −5% 빠진 종목, 언제 사야 하나 — 당일 vs 다음 날, 그리고 다음 날 방향별

대상   코스피 시총 100 + 코스닥 시총 150 (전날 시총 순위, 상장폐지 포함, 주가 1,000원↑)
신호   D 종가가 전날 종가 대비 −5% 이하 (D = 신호일). 같은 종목이 며칠 연속 −5% 면 날마다 따로 센다.

매수 시점
  D 종가          장 마감 직전에 −5% 인 걸 보고 종가에 산다
  D 장중 −5%      장중 저가가 전날 종가 −5% 에 닿으면 그 가격에 산다 (시가부터 −5% 아래면 시가). 신호 = '장중에 −5% 닿음'
                  (종가로 판정하는 위 신호와 대상이 조금 다르다 — 닿았다가 회복한 종목도 들어간다)
  D+1 시가 · D+1 종가 · D+2 시가

보유   매수한 날 종가 기준 N 거래일 뒤 종가에 판다 (N = 1 · 3 · 5 · 10). 시가 매수도 같은 기준이라
       'D+1 시가 · 5일' 은 D+1 시가에 사서 D+6 종가에 판다.
비용   왕복 0.31% (매수 0.065%: 수수료 0.015% + 슬리피지 0.05%, 매도 0.245%: + 세금 0.2%)
지표   건별 평균 · 중앙값 · 승률
       날짜평균  같은 날 신호를 한 묶음으로 평균 낸 뒤 날짜끼리 평균 (폭락일 하루에 신호 100개가 몰려도 하루로 센다)
       t         날짜평균의 t 값
       초과      같은 날 같은 방식으로 대상 전 종목을 샀을 때보다 얼마나 나았나 (시장 흐름을 뺀 값)
기간   2011~2019 / 2020~2025 / 2026 (2026 은 시장이 하루 ±5~10% 움직인 날이 잦아 따로 본다)
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


cf = ffill(np.where(tr, c, np.nan))                 # 정지일 = 마지막 종가 (정지 중엔 그 가격으로 청산)
pc = lag(cf, 1)
r1 = c / pc - 1
lim = np.where(F.dates < '20150615', 0.15, 0.30)[:, None]
limdown = r1 <= -(lim - 0.005)

SIG = U & np.isfinite(r1) & (r1 <= -0.05)
touch_px = np.where(o <= pc * 0.95, o, pc * 0.95)
SIG_TOUCH = U & np.isfinite(pc) & (l <= pc * 0.95)


def net(entry, exitpx):
    return exitpx / entry * (1 - SELLC) / (1 + BUYC) - 1


# 매수 시점별 (매수일 오프셋, 매수 가격 배열) → 보유 N 일 수익 (신호일 t 에 맞춰 정렬된 T × N 배열)
def entry_returns(k, px):
    buy_ok = lead(tr.astype(float), k) > 0 if k else tr
    ent = np.where(buy_ok, lead(px, k) if k else px, np.nan)
    out = {}
    for n in HOLDS:
        out[n] = net(ent, lead(cf, k + n))
    return out


ENTRIES = {
    'D 종가': (0, c),
    'D+1 시가': (1, o),
    'D+1 종가': (1, c),
    'D+2 시가': (2, o),
}
RET = {name: entry_returns(k, px) for name, (k, px) in ENTRIES.items()}
RET['D 장중 −5%'] = {n: net(np.where(SIG_TOUCH, touch_px, np.nan), lead(cf, n)) for n in HOLDS}
# 비교 기준: 같은 날 같은 방식으로 대상 전 종목 (장중 매수는 D 시가 매수로 근사)
BENCH = {}
for name, R in RET.items():
    if name == 'D 장중 −5%':
        R0 = {n: net(np.where(tr, o, np.nan), lead(cf, n)) for n in HOLDS}
    else:
        R0 = R
    BENCH[name] = {n: np.nanmean(np.where(U, R0[n], np.nan), axis=1, keepdims=True) for n in HOLDS}

PER = {
    '2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')),
    '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
    '2026': (C.didx(F, '20260101'), T - 12),
}


def stats(mask, R, B):
    """mask: T × N bool. 반환 dict 또는 None."""
    tt, jj = np.where(mask)
    v = R[tt, jj]; ok = np.isfinite(v)
    tt, jj, v = tt[ok], jj[ok], v[ok]
    if len(v) < 20: return None
    ex = v - B[tt, 0]
    d = defaultdict(list)
    for t, x in zip(tt, v): d[t].append(x)
    dm = np.array([np.mean(x) for x in d.values()])
    tv = dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 2 else np.nan
    return dict(n=len(v), days=len(dm), mean=v.mean(), med=np.median(v), win=(v > 0).mean(), dmean=dm.mean(), t=tv,
                exc=np.nanmean(ex))


def period_mask(m, a, b):
    mm = m.copy(); mm[:a] = False; mm[b:] = False; return mm


lines = []
P = lambda s='': (lines.append(s), print(s, flush=True))


def table(title, mask_fn, names):
    P(f'\n■ {title}')
    for pn, (a, b) in PER.items():
        P(f'  [{pn}]')
        P(f'    {"매수":<11}{"보유":>4} | {"건수":>6} {"신호일":>5} | {"평균":>7} {"중앙":>7} {"승률":>5} | {"날짜평균":>7} {"t":>5} | {"초과":>7}')
        for name in names:
            m = period_mask(mask_fn(name), a, b)
            for n in HOLDS:
                s = stats(m, RET[name][n], BENCH[name][n])
                if s is None:
                    P(f'    {name:<11}{n:>3}일 | 표본 부족'); continue
                P(f'    {name:<11}{n:>3}일 | {s["n"]:6d} {s["days"]:5d} | {s["mean"] * 100:+6.2f}% {s["med"] * 100:+6.2f}% '
                  f'{s["win"] * 100:4.0f}% | {s["dmean"] * 100:+6.2f}% {s["t"]:+5.1f} | {s["exc"] * 100:+6.2f}%')


P('하루 −5% 하락 종목 — 매수 시점 비교 (코스피 시총 100 + 코스닥 시총 150, 비용 왕복 0.31%)')
for pn, (a, b) in PER.items():
    m = period_mask(SIG, a, b)
    P(f'  {pn}: 신호 {m.sum():,}건 · 신호 있는 날 {m.any(1).sum()}일 / {b - a}일 · 하루 평균 {m.sum() / (b - a):.1f}건 · '
      f'그중 하한가 {period_mask(SIG & limdown, a, b).sum()}건')

# 1. 당일 vs 다음 날
table('1. 당일에 살까, 다음 날 살까 (D 종가 −5% 이하 전부)',
      lambda name: SIG_TOUCH if name == 'D 장중 −5%' else SIG,
      ['D 장중 −5%', 'D 종가', 'D+1 시가', 'D+1 종가'])

# 1-1. 낙폭 크기별 (D 종가 vs D+1 시가 vs D+1 종가, 5일 보유만)
P('\n■ 1-1. 낙폭 크기별 — 5일 보유 평균 / 날짜평균 / 초과')
BUCKETS = [('−5~−7%', (r1 <= -0.05) & (r1 > -0.07)), ('−7~−10%', (r1 <= -0.07) & (r1 > -0.10)),
           ('−10% 이하 (하한가 제외)', (r1 <= -0.10) & ~limdown), ('하한가', limdown)]
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for bn, bm in BUCKETS:
        line = f'    {bn:<18}'
        for name in ('D 종가', 'D+1 시가', 'D+1 종가'):
            s = stats(period_mask(SIG & bm, a, b), RET[name][5], BENCH[name][5])
            line += f'| {name} ' + ('표본 부족' if s is None else
                                     f'{s["n"]:5d}건 {s["mean"] * 100:+6.2f}% (날짜 {s["dmean"] * 100:+5.2f}%, 초과 {s["exc"] * 100:+5.2f}%) ')
        P(line)

# 2. 다음 날 방향별
d1 = lead(cf, 1) / cf - 1                          # D+1 종가 / D 종가
UP = SIG & (d1 >= 0); DN = SIG & (d1 < 0)
P('\n■ 2. 다음 날(D+1) 하락 · 상승에 따라 — D+1 종가 또는 D+2 시가에 산다 (D+1 종가에 방향을 보고 결정)')
for pn, (a, b) in PER.items():
    P(f'  {pn}: D+1 하락 {period_mask(DN, a, b).sum():,}건 · 상승(보합 포함) {period_mask(UP, a, b).sum():,}건')
for lab_, M in (('D+1 하락 (D+1 종가 < D 종가)', DN), ('D+1 상승 (D+1 종가 ≥ D 종가)', UP)):
    table(f'2. {lab_}', lambda name, M=M: M, ['D+1 종가', 'D+2 시가'])

P('\n■ 2-1. D+1 등락 크기별 — D+1 종가 매수 · 5일 보유 (평균 / 날짜평균 / 초과, 건수)')
D1B = [('−5% 이하', d1 <= -0.05), ('−5~−2%', (d1 > -0.05) & (d1 <= -0.02)), ('−2~0%', (d1 > -0.02) & (d1 < 0)),
       ('0~+2%', (d1 >= 0) & (d1 < 0.02)), ('+2~+5%', (d1 >= 0.02) & (d1 < 0.05)), ('+5% 이상', d1 >= 0.05)]
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for bn, bm in D1B:
        s = stats(period_mask(SIG & bm, a, b), RET['D+1 종가'][5], BENCH['D+1 종가'][5])
        P(f'    D+1 {bn:<8} ' + ('표본 부족' if s is None else
                                  f'{s["n"]:5d}건 평균 {s["mean"] * 100:+6.2f}% 중앙 {s["med"] * 100:+6.2f}% 승 {s["win"] * 100:3.0f}% '
                                  f'| 날짜평균 {s["dmean"] * 100:+6.2f}% (t {s["t"]:+4.1f}) | 초과 {s["exc"] * 100:+6.2f}%'))

txt = '\n'.join(lines)
open('backtest/results/lab_drop5_timing.md', 'w', encoding='utf-8').write(
    '# 하루 −5% 하락 종목 — 언제 사나 (backtest/lab/fresh/drop5_timing.py)\n\n'
    '정의는 스크립트 머리말 참고. 날짜평균 = 같은 날 신호를 한 묶음으로 본 평균, 초과 = 같은 날 대상 전 종목 대비.\n\n```\n'
    + txt + '\n```\n')
