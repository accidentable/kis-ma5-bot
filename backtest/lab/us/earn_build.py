"""
실적 이벤트 표 만들기: EDGAR 실적 8-K 날짜(edgar_8k.jsonl) ∪ 나스닥 캘린더(calendar.jsonl) × prices.npz → /data/lab/us/earnings/events.pkl
- 우리 가격 데이터에 있는 종목(ETF 제외)만. 발표일은 EDGAR period_ending 우선, 나스닥 날짜는 EDGAR 와 2일 넘게 다를 때만 추가 (ADR 등 8-K 미제출 종목).
  EPS·서프라이즈는 ±2일 안의 나스닥 행. 같은 종목 20거래일 안 중복은 앞의 것.
- 반응일 R: 발표일 D 와 D+1 중 거래대금 배수(vr)가 큰 날 (과거 캘린더엔 장전/장후 정보가 없다).
- 특징(R 기준): gap(전일 종가→시가), dayret(전일 종가→종가), intra(시가→종가), vr, surprise(%), 추정치 수, 발표 전 20/60일 수익률, 52주 고가 대비, 60일 변동성, 시총 순위, 섹터, 직전 분기 dayret.
- 앞으로: 진입 c[R] · o[R+1] · c[R+1] 기준 +1/2/3/5/10/20일 종가 수익률, 20일 내 최고/최저 종가, SPY 20일 수익률(시장 조정용).
"""
import json, re, sys, time, pathlib, numpy as np, pandas as pd
from backtest.lab.us.common import US

t0 = time.time(); D = US(); T = D.T
tick = {t: i for i, t in enumerate(D.tickers)}
def money(s):
    if not s or s in ('N/A', ''): return np.nan
    m = re.match(r'^\(?\$?(-?[\d,]*\.?\d+)\)?$', s.strip())
    if not m: return np.nan
    v = float(m.group(1).replace(',', '')); return -v if s.strip().startswith('(') else v
# 1) 나스닥 캘린더 → (sym, date) → eps·컨센서스·서프라이즈 (2024-11~ 전부, 그 전은 수집된 만큼)
import datetime as dt, collections
nq = {}
for line in open('/data/lab/us/earnings/calendar.jsonl', encoding='utf-8'):
    rec = json.loads(line); d = rec['date'].replace('-', '')
    for r in rec['rows']:
        sym = r.get('symbol', '')
        if sym not in tick or D.is_etf[tick[sym]]: continue
        eps, fc = money(r.get('eps')), money(r.get('epsForecast'))
        try: sur = float(r.get('surprise'))
        except (TypeError, ValueError): sur = (eps - fc) / abs(fc) * 100 if np.isfinite(eps) and np.isfinite(fc) and fc != 0 else np.nan
        try: nest = int(r.get('noOfEsts'))
        except (TypeError, ValueError): nest = 0
        if (sym, d) not in nq or np.isfinite(eps): nq[(sym, d)] = dict(eps=eps, fc=fc, surprise=sur, nest=nest, fq=r.get('fiscalQuarterEnding', ''))
nq_by_sym = collections.defaultdict(list)
for (sym, d) in nq: nq_by_sym[sym].append(d)
# 2) EDGAR 실적 8-K → sym → 발표일(period_ending, 없으면 file_date) 집합
ed = collections.defaultdict(set)
if pathlib.Path('/data/lab/us/earnings/edgar_8k.jsonl').exists():
    for line in open('/data/lab/us/earnings/edgar_8k.jsonl', encoding='utf-8'):
        rec = json.loads(line)
        for fl in rec['filings']:
            if fl.get('form') != '8-K': continue
            d = (fl.get('period') or fl.get('file_date') or '').replace('-', '')
            if len(d) == 8 and '20100101' <= d <= '20261001': ed[rec['sym']].add(d)
# 3) 이벤트 후보 = EDGAR 날짜 ∪ 나스닥 날짜(EDGAR 날짜와 2일 이내면 제외). 서프라이즈는 ±2일 안의 나스닥 행에서
def shift(d, k): return (dt.date(int(d[:4]), int(d[4:6]), int(d[6:])) + dt.timedelta(k)).strftime('%Y%m%d')
rows = []
for sym in set(ed) | set(nq_by_sym):
    dates = {(d, 'edgar') for d in ed.get(sym, ())}
    for d in nq_by_sym.get(sym, ()):
        if not any(abs((dt.date(int(d[:4]), int(d[4:6]), int(d[6:])) - dt.date(int(e[:4]), int(e[4:6]), int(e[6:]))).days) <= 2 for e, _ in dates): dates.add((d, 'nasdaq'))
    for d, src in sorted(dates):
        m = next((nq[(sym, shift(d, k))] for k in (0, 1, -1, 2, -2) if (sym, shift(d, k)) in nq), None)
        rows.append((sym, d, src, *(m.values() if m else (np.nan, np.nan, np.nan, 0, ''))))
ev = pd.DataFrame(rows, columns=['sym', 'date', 'src', 'eps', 'fc', 'surprise', 'nest', 'fq'])
print(f'EDGAR 종목 {len(ed)}, 나스닥 (종목,날짜) {len(nq)} → 이벤트 후보 {len(ev)} (edgar {(ev.src == "edgar").sum()}, nasdaq만 {(ev.src == "nasdaq").sum()}, 서프라이즈 있음 {ev.surprise.notna().sum()})', flush=True)
# 반응일 · 특징
spy = D.col('SPY') if 'SPY' in tick else None
out = []
for sym, g in ev.groupby('sym'):
    j = tick[sym]; last_r = -99
    for _, e in g.sort_values('date').iterrows():
        tD = D.didx(e.date)
        if tD >= T - 1 or tD < 61: continue
        v0, v1 = D.vr[tD, j], D.vr[tD + 1, j]
        if not np.isfinite(v0) and not np.isfinite(v1): continue
        # 반응일: 그날 종가에 알 수 있는 정보만 — D 의 거래 배수가 2 이상이면 D, 아니면 D+1 이 2 이상일 때 D+1, 둘 다 아니면 큰 쪽(품질 낮음)
        r = tD if np.nan_to_num(v0) >= 2 else (tD + 1 if np.nan_to_num(v1) >= 2 else (tD if np.nan_to_num(v0) >= np.nan_to_num(v1) else tD + 1))
        if r >= T - 2 or not D.tr[r, j] or not D.tr[r - 1, j]: continue
        if r - last_r < 20: continue
        pc = D.cf[r - 1, j]; o, c = D.o[r, j], D.c[r, j]
        if not (pc > 0 and o > 0 and c > 0): continue
        rec = dict(sym=sym, date=e.date, src=e.src, fq=e.fq, eps=e.eps, fc=e.fc, surprise=e.surprise, nest=e.nest, r=r, rdate=D.dates[r], bmo=(r == tD),
                   gap=o / pc - 1, dayret=c / pc - 1, intra=c / o - 1, vr=max(np.nan_to_num(v0), np.nan_to_num(v1)),
                   ret20=D.cf[r - 1, j] / D.cf[r - 21, j] - 1, ret60=D.cf[r - 1, j] / D.cf[r - 61, j] - 1,
                   hi250=D.cf[r - 1, j] / D.hi250[r, j] - 1 if np.isfinite(D.hi250[r, j]) else np.nan,
                   vol60=D.vol60[r, j], caprank=D.caprank[j], cap=D.cap[j], sector=D.sector[j], mkt_r=D.mkt[r],
                   o1=D.o[r + 1, j] / c - 1 if D.tr[r + 1, j] else np.nan, c1=D.cf[r + 1, j] / c - 1,
                   pre5=D.cf[r - 1, j] / D.cf[r - 6, j] - 1, thru=c / D.cf[r - 6, j] - 1, thru20=(D.cf[r + 20, j] / D.cf[r - 6, j] - 1) if r + 20 < T else np.nan)
        for k in (2, 3, 5, 10, 20, 40, 60):
            rec[f'c{k}'] = D.cf[min(r + k, T - 1), j] / c - 1 if r + k < T else np.nan
        seg = D.cf[r + 1:min(r + 21, T), j]
        rec['max20'] = seg.max() / c - 1 if len(seg) == 20 else np.nan; rec['min20'] = seg.min() / c - 1 if len(seg) == 20 else np.nan
        seg5 = D.cf[r + 1:min(r + 6, T), j]; rec['max5'] = seg5.max() / c - 1 if len(seg5) == 5 else np.nan
        if D.tr[r + 1, j]:                                       # R+1 시가 진입 기준 20일
            o1p = D.o[r + 1, j]; rec['o1_c20'] = D.cf[r + 20, j] / o1p - 1 if r + 20 < T else np.nan
            rec['o1_max20'] = seg.max() / o1p - 1 if len(seg) == 20 else np.nan
        for k in (20, 40, 60):
            rec[f'spy{k}'] = D.cf[r + k, spy] / D.cf[r, spy] - 1 if spy is not None and r + k < T else np.nan
            rec[f'uni{k}'] = float(np.prod(1 + np.nan_to_num(D.mkt[r + 1:r + k + 1]))) - 1 if r + k < T else np.nan   # 대상 종목 동일가중 평균 (같은 생존 편향)
        out.append(rec); last_r = r
E = pd.DataFrame(out).sort_values(['sym', 'r']).reset_index(drop=True)
E['prev_dayret'] = E.groupby('sym').dayret.shift(1); E['prev_gap'] = E.groupby('sym').gap.shift(1)
E['year'] = E.rdate.str[:4].astype(int)
E.to_pickle('/data/lab/us/earnings/events.pkl')
print(f'이벤트 {len(E)}건, 종목 {E.sym.nunique()}, {E.rdate.min()}~{E.rdate.max()}, 반응 vr≥2 {(E.vr >= 2).mean():.0%}, 장전 {E.bmo.mean():.0%}, 서프라이즈 있음 {E.surprise.notna().mean():.0%}, {time.time() - t0:.0f}s', flush=True)
print(E.groupby('year').size().to_string())
