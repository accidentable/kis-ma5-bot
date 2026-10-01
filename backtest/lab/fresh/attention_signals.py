"""
인기 종목 — 관심(거래) 이 몰린 종목을 사면 어떻게 되나 (stock_signals.py 와 같은 틀)

'인기' 를 과거 데이터로 잴 수 있는 대용 지표
  거래대금 순위   그날 전 종목(상장 전체) 중 거래대금 순위
  거래대금 급증   그날 거래대금 / 직전 20일 평균 (vr)
  회전율          직전 20일 평균 거래대금 / 시가총액 (전날 기준) — 늘 많이 거래되는 종목
  최근 인기        직전 20일 안에 거래대금 50위 안에 든 적 있음 (attn20)
못 재는 것: 검색량 · 종목토론방 · 앱 조회수 (과거 기록이 없다. 네이버 데이터랩 검색 추이는 API 키가 있으면 2016~ 받을 수 있다)

대상 · 제외 · 매수 · 보유 · 지표는 stock_signals.py 와 같다 (코스피 시총 100 + 코스닥 시총 150, 상·하한가 · 최근 20일 정지/하한가 제외,
D+1 시가 매수, 비용 왕복 0.31%, 초과 = 같은 날 대상 전 종목 대비, 날짜별 묶음). 보유에 20일을 더했다 (인기는 한 달 단위 효과가 있을 수 있어서).

신호 (결과 보기 전에 정함)
  A1  거래대금 전체 10위 안
  A2  A1 & 상승 마감            A3  A1 & 하락 마감
  A4  거래대금 50위 안 & 처음    (오늘 50위 안인데 직전 20일엔 없었음 — 새로 관심이 붙은 날)
  A5  거래대금 급증 vr ≥ 3 (방향 무관)
  A6  A5 & 상승 마감            A7  A5 & 하락 마감
  A8  회전율 상위 10% (대상 안)   A9  회전율 하위 30% (대상 안, 조용한 종목)
  A10 최근 인기 (20일 안 50위 안 든 적 있음)   A11 비인기 (든 적 없고 오늘도 50위 밖)
통과 기준  stock_signals.py 와 같음 (D+1 시가 · 5일 날짜별 초과 > 0 세 기간, t ≥ 2 두 기간, 6건↑ 달 ≥ 80% 두 기간)
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, c = F.o.astype(float), F.c.astype(float); tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
BUYC, SELLC = 0.00065, 0.00265
HOLDS = (1, 3, 5, 10, 20); MONTH = 20


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out
def lead(x, k): out = np.full_like(x, np.nan); out[:-k] = x[k:]; return out
def net(entry, exitpx): return exitpx / entry * (1 - SELLC) / (1 + BUYC) - 1


cf = ffill(np.where(tr, c, np.nan)); pc = lag(cf, 1); r1 = c / pc - 1
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
E = U & np.isfinite(r1) & ~F.limdown.astype(bool) & ~F.limup.astype(bool) & ~F.bad20.astype(bool)
val = F.val.astype(float); vr = F.vr.astype(float); turn1 = F.turn1.astype(float); attn20 = F.attn20.astype(bool)
vrank = (-np.nan_to_num(val, nan=-1)).argsort(axis=1).argsort(axis=1) + 1          # 그날 전 종목 거래대금 순위


def pct_mask(x, lo, hi):
    """대상(U) 안에서 x 의 백분위가 [lo, hi) 인 종목 (작을수록 0)"""
    out = np.zeros((T, N), bool)
    for t in range(T):
        idx = np.where(U[t] & np.isfinite(x[t]))[0]
        if len(idx) < 20: continue
        order = idx[np.argsort(x[t, idx])]; n = len(order)
        out[t, order[int(n * lo):int(n * hi)]] = True
    return out


A1 = E & (vrank <= 10); A5 = E & (vr >= 3)
SIGS = {
    'A1 거래대금 10위 안': A1,
    'A2 A1 & 상승 마감': A1 & (r1 > 0),
    'A3 A1 & 하락 마감': A1 & (r1 < 0),
    'A4 50위 안 & 처음 (새 관심)': E & (vrank <= 50) & ~attn20,
    'A5 거래대금 급증 vr≥3': A5,
    'A6 A5 & 상승 마감': A5 & (r1 > 0),
    'A7 A5 & 하락 마감': A5 & (r1 < 0),
    'A8 회전율 상위 10%': E & pct_mask(turn1, 0.9, 1.0),
    'A9 회전율 하위 30% (조용)': E & pct_mask(turn1, 0.0, 0.3),
    'A10 최근 인기 (20일 내 50위)': E & attn20,
    'A11 비인기 (50위 밖 지속)': E & ~attn20 & (vrank > 50),
}


def rets(k, px):
    ent = np.where(tr, px, np.nan); ent = lead(ent, k) if k else ent
    return {n: net(ent, lead(cf, k + n)) for n in HOLDS}


RET = {'D+1 시가': rets(1, o), 'D 종가': rets(0, c)}
BENCH = {e: {n: np.nanmean(np.where(U, R[n], np.nan), axis=1) for n in HOLDS} for e, R in RET.items()}
PER = {'2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')),
       '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
       '2026': (C.didx(F, '20260101'), T - 22)}


def pm(m, a, b): mm = m.copy(); mm[:a] = False; mm[b:] = False; return mm


def stats(mask, R, B):
    tt, jj = np.where(mask); v = R[tt, jj]; ok = np.isfinite(v); tt, v = tt[ok], v[ok]
    if len(v) < 15: return None
    ex = v - B[tt]; d = defaultdict(list); dx = defaultdict(list)
    for t, x, y in zip(tt, v, ex): d[t].append(x); dx[t].append(y)
    dm = np.array([np.mean(x) for x in d.values()]); de = np.array([np.mean(x) for x in dx.values()])
    tv = de.mean() / (de.std(ddof=1) / np.sqrt(len(de))) if len(de) > 2 else np.nan
    return dict(n=len(v), days=len(dm), mean=v.mean(), med=np.median(v), win=(v > 0).mean(), dmean=dm.mean(), dexc=de.mean(), t=tv)


def freq(mask, a, b):
    cnt = mask.sum(1).astype(float); cs = np.concatenate([[0], np.cumsum(cnt)])
    return np.array([cs[s + MONTH - 1] - cs[s - 1] for s in range(max(a, 1), min(b, T - MONTH))])


lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P('인기 종목 대용 지표 11개 — D+1 시가 매수 · 비용 왕복 0.31% · 초과 = 같은 날 대상 전 종목 대비 (날짜별 묶음)')
P('\n■ 1. 신호 빈도 — 건수 · 신호일 · 한 달(20거래일) 평균 건수 · 6건↑ 달 비율')
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for sn, m in SIGS.items():
        mm = pm(m, a, b); fq = freq(m, a, b)
        P(f'    {sn:<26} {mm.sum():6,d}건 {mm.any(1).sum():5d}일 | 한 달 {fq.mean():6.1f}건 · 6건↑ {(fq >= 6).mean() * 100:4.0f}%')
P('\n■ 2. 수익 — D+1 시가 매수 1 · 3 · 5 · 10 · 20일, D 종가 매수 5일')
P(f'    {"신호":<26}{"매수·보유":<12} | {"건수":>6} | {"평균":>7} {"중앙":>7} {"승률":>4} | {"날짜평균":>7} {"날짜초과":>7} {"t":>5}')
for pn, (a, b) in PER.items():
    P(f'  [{pn}]')
    for sn, m in SIGS.items():
        for ent, n in (('D+1 시가', 1), ('D+1 시가', 3), ('D+1 시가', 5), ('D+1 시가', 10), ('D+1 시가', 20), ('D 종가', 5)):
            s = stats(pm(m, a, b), RET[ent][n], BENCH[ent][n]); lab_ = f'{ent} {n:>2}일'
            if s is None: P(f'    {sn:<26}{lab_:<12} | 표본 부족'); continue
            P(f'    {sn:<26}{lab_:<12} | {s["n"]:6d} | {s["mean"] * 100:+6.2f}% {s["med"] * 100:+6.2f}% {s["win"] * 100:3.0f}% '
              f'| {s["dmean"] * 100:+6.2f}% {s["dexc"] * 100:+6.2f}% {s["t"]:+5.1f}')
P('\n■ 3. 통과 판정 (D+1 시가 · 5일) + 20일 초과')
for sn, m in SIGS.items():
    row = {}
    for pn, (a, b) in PER.items():
        s5 = stats(pm(m, a, b), RET['D+1 시가'][5], BENCH['D+1 시가'][5]); s20 = stats(pm(m, a, b), RET['D+1 시가'][20], BENCH['D+1 시가'][20])
        row[pn] = ((s5['dexc'], s5['t']) if s5 else (np.nan, np.nan), s20['dexc'] if s20 else np.nan)
    f1 = (freq(m, *PER['2011~2019']) >= 6).mean(); f2 = (freq(m, *PER['2020~2025']) >= 6).mean()
    ok_exc = all(np.nan_to_num(row[p][0][0], nan=-1) > 0 for p in PER)
    ok_t = all(np.nan_to_num(row[p][0][1], nan=-9) >= 2 for p in ('2011~2019', '2020~2025'))
    ok_f = f1 >= 0.8 and f2 >= 0.8
    P(f'  {sn:<26} 5일 ' + ' '.join(f'{row[p][0][0] * 100:+5.2f}(t{row[p][0][1]:+4.1f})' for p in PER)
      + ' | 20일 ' + ' '.join(f'{row[p][1] * 100:+5.2f}' for p in PER)
      + f' | 6건↑ {f1 * 100:3.0f}% {f2 * 100:3.0f}% | ' + ('통과' if ok_exc and ok_t and ok_f else
                                                       '탈락: ' + ', '.join(x for x, ok in (('초과', ok_exc), ('t', ok_t), ('빈도', ok_f)) if not ok)))
open('backtest/results/lab_attention_signals.md', 'w', encoding='utf-8').write(
    '# 인기 종목 대용 지표 — 단기 이벤트 스터디 (backtest/lab/fresh/attention_signals.py)\n\n정의 · 통과 기준은 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
