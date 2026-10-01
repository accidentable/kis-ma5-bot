"""
네이버 뉴스 검색(날짜 조건)으로 급락 사건의 '전날~당일' 기사를 모은다 — gnews_collect.py 의 네이버판 (구글은 800건쯤에서 503 으로 막혔다)

    python -m backtest.lab.fresh.nnews_collect            → /data/lab/gnews/events_naver.jsonl (이어받기 가능)

사건 정의 · 순서는 gnews_collect.py 와 같다 (−10~−20% 급락, 폭락일 먼저 → 최근순).
질의  search.naver.com/search.naver?where=news&query="회사명"&pd=3&ds=D−1&de=D&sort=0 (관련도순) — 첫 페이지 10건만 (분류엔 충분)
저장  한 줄에 사건 하나: date code name r1 mkt vr z crash t1 n_items items[{d, t, s}]   (d = 기사 날짜 'YYYY.MM.DD', s = 언론사)
속도  2~3초에 1건. 비정상 응답(200 아님 · 제목 0건인데 '검색결과가 없습니다' 도 아님)이면 10분 쉬고 같은 사건 재시도 (최대 6번).
"""
import json, os, re, html, sys, time, random, urllib.parse
import numpy as np, requests, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C

OUT = '/data/lab/gnews/events_naver.jsonl'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
F = PF.load(); T, N = F.T, F.N
tr = F.traded.astype(bool); c = F.c.astype(float)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
cf = np.where(tr, c, np.nan)
for t in range(1, T):
    m = np.isnan(cf[t]); cf[t, m] = cf[t - 1, m]
pc = np.vstack([np.full((1, N), np.nan), cf[:-1]]); r1 = c / pc - 1
limdown = F.limdown.astype(bool)
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
sig20 = np.full(T, np.nan)
for t in range(21, T): sig20[t] = np.nanstd(mkt[t - 20:t])
crash = (mkt <= -0.03) & (mkt <= -3 * sig20)
dip = U & np.isfinite(r1) & (r1 <= -0.10) & (r1 > -0.20) & ~limdown
vr, z = F.vr.astype(float), F.z.astype(float)
a0 = C.didx(F, '20110103')
ev = [(t, j) for t in range(a0, T - 7) for j in np.where(dip[t])[0]]
ev.sort(key=lambda x: (not crash[x[0]], -x[0]))
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding='utf-8'):
        try: d = json.loads(line); done.add((d['date'], d['code']))
        except Exception: pass
print(f'사건 {len(ev)}건 (폭락일 {sum(crash[t] for t, _ in ev)}건), 이미 받은 것 {len(done)}건', flush=True)
sess = requests.Session()
sess.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
sess.headers['Accept-Language'] = 'ko-KR,ko;q=0.9'
TITLE = re.compile(r'<span class="[^"]*sds-comps-text-type-headline1[^"]*">(.*?)</span>', re.S)
PRESS = re.compile(r'<span class="[^"]*sds-comps-profile-info-title-text[^"]*">(.*?)</span>', re.S)
DATE = re.compile(r'(\d{4}\.\d{2}\.\d{2})\.')


def clean(x): return html.unescape(re.sub(r'<[^>]+>', '', x)).strip()


def parse(s):
    items, prev = [], 0
    for m in TITLE.finditer(s):
        seg = s[prev:m.end()]; prev = m.end()
        ps = PRESS.findall(seg); ds = DATE.findall(seg)
        items.append({'d': ds[-1] if ds else '', 't': clean(m.group(1)), 's': clean(ps[-1]) if ps else ''})
    return items


def fetch(name, d):
    import datetime as dt
    d1 = dt.datetime.strptime(d, '%Y%m%d'); d0 = d1 - dt.timedelta(days=1)
    q = urllib.parse.quote(f'"{name}"')
    url = (f'https://search.naver.com/search.naver?where=news&query={q}&sm=tab_opt&sort=0&photo=0&field=0&pd=3'
           f'&ds={d0:%Y.%m.%d}&de={d1:%Y.%m.%d}&nso=so%3Ar%2Cp%3Afrom{d0:%Y%m%d}to{d1:%Y%m%d}')
    for k in range(6):
        try:
            r = sess.get(url, timeout=25)
        except requests.RequestException as e:
            print(f'  네트워크 오류 {e}; 120초 대기', flush=True); time.sleep(120); continue
        if r.status_code == 200:
            items = parse(r.text)
            if items or '검색결과가 없습니다' in r.text or 'not_found' in r.text or len(r.text) > 150_000:
                return items
        print(f'  HTTP {r.status_code} 길이 {len(r.text)} 제목 0건; 600초 대기 ({k + 1}/6)', flush=True); time.sleep(600)
    return None


t0 = time.time(); n_new = 0
with open(OUT, 'a', encoding='utf-8') as fh:
    for i, (t, j) in enumerate(ev):
        d, code, name = str(F.dates[t]), str(F.codes[j]), str(F.names[j])
        if (d, code) in done: continue
        items = fetch(name, d)
        if items is None:
            print(f'  {d} {name}: 포기', flush=True); continue
        rec = dict(date=d, code=code, name=name, r1=round(float(r1[t, j]), 4), mkt=round(float(mkt[t]), 4),
                   vr=round(float(vr[t, j]), 2) if np.isfinite(vr[t, j]) else None, z=round(float(z[t, j]), 2) if np.isfinite(z[t, j]) else None,
                   crash=bool(crash[t]), t1=bool(mkt[t] <= -0.03), n_items=len(items), items=items)
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush(); n_new += 1
        if n_new % 50 == 0:
            print(f'  {n_new}건 받음 ({i + 1}/{len(ev)} 위치) {time.time() - t0:.0f}s — 마지막 {d} {name} {len(items)}건', flush=True)
        time.sleep(2.0 + random.random() * 1.0)
print(f'끝: 새로 {n_new}건, {time.time() - t0:.0f}s', flush=True)
