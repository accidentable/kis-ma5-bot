"""
EDGAR 전문검색으로 실적 8-K(Item 2.02 'Results of Operations') 제출 목록 수집 → /data/lab/us/earnings/edgar_8k.jsonl (종목당 한 줄)
- 종목→CIK: https://www.sec.gov/files/company_tickers.json (ETF·해외 ADR(6-K 제출)은 없음)
- 검색: https://efts.sec.gov/LATEST/search-index?q="Results of Operations"&forms=8-K&ciks=...&dateRange=custom&startdt=2010-01-01&enddt=2026-10-01&page=n
- period_ending = 8-K 의 보고 기준일(= 실적 발표일), file_date = 제출일. 접수번호(adsh)로 중복 제거.
- SEC 권장 10 req/s 이하: 스레드 4개 · 요청 사이 0.15초. 이미 받은 종목은 건너뜀.
"""
import json, time, sys, pathlib, threading, requests, numpy as np
from concurrent.futures import ThreadPoolExecutor

OUT = pathlib.Path('/data/lab/us/earnings/edgar_8k.jsonl')
H = {'User-Agent': 'kis-lab research yth1131@gmail.com', 'Accept-Encoding': 'gzip, deflate'}
cm = json.load(open('/data/lab/us/earnings/company_tickers.json'))
norm = lambda t: t.replace('.', '-').replace('/', '-').upper()
cik_of = {norm(v['ticker']): int(v['cik_str']) for v in cm.values()}
z = np.load('/data/lab/us/prices.npz'); tickers = z['tickers'].astype(str)
meta = json.load(open('/data/lab/us/meta.json'))
syms = [t for t in tickers if meta.get(t, {}).get('sector', '') != 'ETF' and norm(t) in cik_of]
done = set()
if OUT.exists():
    for line in OUT.open(encoding='utf-8'):
        try: done.add(json.loads(line)['sym'])
        except Exception: pass
todo = [s for s in syms if s not in done]
print(f'대상 {len(syms)} 종목, 받을 것 {len(todo)} (이미 {len(done)})', flush=True)
lock = threading.Lock(); t0 = time.time(); n = [0, 0]
sess = threading.local()
def get(url):
    if not hasattr(sess, 's'): sess.s = requests.Session(); sess.s.headers.update(H)
    for k in range(6):
        try:
            r = sess.s.get(url, timeout=40)
            if r.status_code == 200: return r.json()
            wait = 60 * (k + 1) if r.status_code in (403, 429) else 10 * (k + 1)
            print(f'  HTTP {r.status_code}; {wait}초 대기', flush=True)
        except Exception as e:
            wait = 15 * (k + 1); print(f'  오류 {type(e).__name__}; {wait}초 대기', flush=True)
        time.sleep(wait)
    return None
def fetch(sym):
    cik = cik_of[norm(sym)]; base = (f'https://efts.sec.gov/LATEST/search-index?q=%22Results%20of%20Operations%22&forms=8-K&ciks={cik:010d}'
                                     f'&dateRange=custom&startdt=2010-01-01&enddt=2026-10-01')
    seen, filings, page = set(), [], 1
    while True:
        j = get(base + (f'&page={page}' if page > 1 else ''))
        if j is None: return None
        hits = j.get('hits', {}).get('hits', [])
        for h in hits:
            adsh = h['_id'].split(':')[0]; s = h['_source']
            if adsh in seen: continue
            seen.add(adsh); filings.append({'adsh': adsh, 'file_date': s.get('file_date'), 'period': s.get('period_ending'), 'form': s.get('form')})
        total = j.get('hits', {}).get('total', {}).get('value', 0)
        if len(hits) < 100 or page * 100 >= total or page >= 10: break
        page += 1; time.sleep(0.15)
    time.sleep(0.15)
    return {'sym': sym, 'cik': cik, 'filings': filings}
with OUT.open('a', encoding='utf-8') as f, ThreadPoolExecutor(4) as ex:
    for rec in ex.map(fetch, todo):
        if rec is None: n[1] += 1; continue
        with lock:
            f.write(json.dumps(rec) + '\n'); f.flush(); n[0] += 1
            if n[0] % 100 == 0: print(f'  {n[0]}종목 ({time.time() - t0:.0f}s, 실패 {n[1]})', flush=True)
print(f'끝 {n[0]}종목 실패 {n[1]} {time.time() - t0:.0f}s', flush=True)
