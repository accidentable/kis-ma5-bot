"""
구글 뉴스 RSS 로 급락 사건의 '전날~당일' 기사를 모은다 (종목별 · 날짜별 과거 뉴스 대용)

    python -m backtest.lab.fresh.gnews_collect            → /data/lab/gnews/events.jsonl (이어받기 가능)

사건  코스피 시총 100 + 코스닥 시총 150 종목이 하루 −10~−20% (하한가 제외) 빠진 날 (설계 시뮬의 overlay 후보와 같은 정의), 2011~
순서  폭락일(T5: 시장 −3% & −3σ) 사건 먼저, 그다음 나머지를 최근 날짜부터
질의  https://news.google.com/rss/search?q="회사명" after:D−1 before:D&hl=ko&gl=KR&ceid=KR:ko   (구글 날짜 필터는 양끝 포함처럼 동작)
      회사명은 marcap 의 현재 이름이라 이름이 바뀐 회사(기아차→기아 등)는 옛 기사가 덜 잡힌다.
속도  약 1.2초에 1건, 429/5xx 면 60·120·300초 쉬고 재시도. 전체 약 4,700건 → 1시간 반.
저장  한 줄에 사건 하나: date code name r1 mkt vr z crash(T5) t1(시장 −3%) n_items items[{d, t, s}]
"""
import json, os, re, sys, time, random, urllib.parse
import numpy as np, requests, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C

OUT = '/data/lab/gnews/events.jsonl'
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

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
sess = requests.Session(); sess.headers['User-Agent'] = UA


def prev_day(d):           # 달력 기준 하루 전 (주말 포함 — 금요일 저녁 뉴스는 월요일 사건의 '전날' 이 아니지만 단순히 간다)
    import datetime as dt
    return (dt.datetime.strptime(d, '%Y%m%d') - dt.timedelta(days=1)).strftime('%Y-%m-%d')


def fetch(name, d):
    q = f'"{name}" after:{prev_day(d)} before:{d[:4]}-{d[4:6]}-{d[6:]}'
    url = 'https://news.google.com/rss/search?q=' + urllib.parse.quote(q) + '&hl=ko&gl=KR&ceid=KR:ko'
    # 503/429 는 구글이 IP 를 잠시 막은 것 — 15분 쉬고 같은 사건을 다시 시도 (최대 6번 = 90분). 사건을 건너뛰면 표본이 비니 버티는 쪽.
    backoff = [900, 900, 900, 900, 900, 900]
    for k in range(6):
        try:
            r = sess.get(url, timeout=25)
        except requests.RequestException as e:
            print(f'  네트워크 오류 {e}; 120초 대기', flush=True); time.sleep(120); continue
        if r.status_code == 200:
            items = []
            for it in re.findall(r'<item>(.*?)</item>', r.text, re.S):
                t_ = re.search(r'<title>(.*?)</title>', it, re.S); pd = re.search(r'<pubDate>(.*?)</pubDate>', it); s_ = re.search(r'<source[^>]*>(.*?)</source>', it)
                title = re.sub(r'<!\[CDATA\[|\]\]>', '', t_.group(1)).strip() if t_ else ''
                title = title.replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'").replace('&lt;', '<').replace('&gt;', '>')
                items.append({'d': pd.group(1)[5:16] if pd else '', 't': title, 's': s_.group(1) if s_ else ''})
            return items
        print(f'  HTTP {r.status_code}; {backoff[min(k, 3)]}초 대기', flush=True); time.sleep(backoff[min(k, 3)])
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
        time.sleep(3.0 + random.random() * 1.0)        # 1.5초 간격으론 800건쯤에서 503 이 왔다
print(f'끝: 새로 {n_new}건, {time.time() - t0:.0f}s', flush=True)
