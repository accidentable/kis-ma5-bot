"""
OpenDART 로 급락 사건의 '전날~당일' 공시 목록을 모은다 (뉴스 수집과 같은 사건 · 같은 순서)

    python -m backtest.lab.fresh.dart_collect            → /data/lab/dart/events_dart.jsonl (이어받기 가능)

키     ~/kis-lab/.env 의 DART_API_KEY (출력하지 않는다). 회사 코드는 /data/lab/dart/listed_corp.json (corpCode.xml 에서 추출, 종목코드 → corp_code)
질의   list.json?corp_code=&bgn_de=D−1&end_de=D&page_count=100   (접수일 기준. 당일 장 마감 뒤 공시도 들어가므로 '원인' 과 '결과' 가 섞인다)
저장   한 줄에 사건 하나: date code name r1 mkt vr z crash t1 corp_code n_items items[{dt, nm, rm, no}]   (corp_code 없으면 items=None)
속도   0.4초에 1건 (일 한도 안). 오류 응답(status != '000') 이면 60초 쉬고 재시도, 한도 초과(020)면 중단.
"""
import json, os, sys, time, random
import numpy as np, requests, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C

OUT = '/data/lab/dart/events_dart.jsonl'
KEY = dict(l.strip().split('=', 1) for l in open(os.path.expanduser('~/kis-lab/.env'), encoding='utf-8') if '=' in l and not l.startswith('#')).get('DART_API_KEY', '').strip().strip('"').strip("'")
assert len(KEY) == 40, 'DART_API_KEY 가 .env 에 없다'
CORP = json.load(open('/data/lab/dart/listed_corp.json', encoding='utf-8'))
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


def fetch(corp_code, d):
    import datetime as dt
    d1 = dt.datetime.strptime(d, '%Y%m%d'); d0 = d1 - dt.timedelta(days=1)
    for k in range(5):
        try:
            r = sess.get('https://opendart.fss.or.kr/api/list.json', timeout=30,
                         params=dict(crtfc_key=KEY, corp_code=corp_code, bgn_de=d0.strftime('%Y%m%d'), end_de=d, page_no=1, page_count=100))
            j = r.json()
        except Exception as e:
            print(f'  오류 {e}; 60초 대기', flush=True); time.sleep(60); continue
        st = j.get('status')
        if st == '000':
            return [{'dt': x.get('rcept_dt', ''), 'nm': x.get('report_nm', ''), 'rm': x.get('rm', ''), 'no': x.get('rcept_no', '')} for x in j.get('list', [])]
        if st == '013':                                   # 조회된 데이터가 없습니다
            return []
        if st == '020':
            print('  일 한도 초과 — 중단', flush=True); sys.exit(2)
        print(f'  status {st} {j.get("message")}; 60초 대기', flush=True); time.sleep(60)
    return None


t0 = time.time(); n_new = 0; n_nocode = 0
with open(OUT, 'a', encoding='utf-8') as fh:
    for i, (t, j) in enumerate(ev):
        d, code, name = str(F.dates[t]), str(F.codes[j]), str(F.names[j])
        if (d, code) in done: continue
        cc = CORP.get(code, [None])[0]
        if cc is None:
            items = None; n_nocode += 1
        else:
            items = fetch(cc, d)
            if items is None:
                print(f'  {d} {name}: 포기', flush=True); continue
            time.sleep(0.4 + random.random() * 0.2)
        rec = dict(date=d, code=code, name=name, r1=round(float(r1[t, j]), 4), mkt=round(float(mkt[t]), 4),
                   vr=round(float(vr[t, j]), 2) if np.isfinite(vr[t, j]) else None, z=round(float(z[t, j]), 2) if np.isfinite(z[t, j]) else None,
                   crash=bool(crash[t]), t1=bool(mkt[t] <= -0.03), corp_code=cc, n_items=(len(items) if items is not None else None), items=items)
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush(); n_new += 1
        if n_new % 200 == 0:
            print(f'  {n_new}건 ({i + 1}/{len(ev)} 위치) {time.time() - t0:.0f}s — 마지막 {d} {name} {rec["n_items"]}건 · 코드 없음 누적 {n_nocode}', flush=True)
print(f'끝: 새로 {n_new}건 (corp_code 없음 {n_nocode}건), {time.time() - t0:.0f}s', flush=True)
