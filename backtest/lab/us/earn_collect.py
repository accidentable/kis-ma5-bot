"""
나스닥 실적 캘린더 수집: https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD
2010-01-01 ~ 2026-10-01 평일을 최근 날짜부터 받아서 /data/lab/us/earnings/calendar.jsonl 에 하루 한 줄로 저장 (date, rows).
rows: symbol, name, eps(실제), epsForecast(컨센서스), surprise(%), noOfEsts, fiscalQuarterEnding, time(과거엔 전부 'time-not-supplied'), marketCap(현재 기준).
이미 받은 날짜는 건너뛴다(재개 가능). 실패하면 백오프 후 재시도, 6번 실패하면 그 날짜는 건너뛰고 기록.
"""
import json, time, sys, random, datetime as dt, pathlib, requests

OUT = pathlib.Path('/data/lab/us/earnings/calendar.jsonl'); FAIL = pathlib.Path('/data/lab/us/earnings/failed.txt')
H = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36', 'Accept': 'application/json',
     'Accept-Language': 'en-US,en;q=0.9', 'Origin': 'https://www.nasdaq.com', 'Referer': 'https://www.nasdaq.com/market-activity/earnings'}
done = set()
if OUT.exists():
    for line in OUT.open(encoding='utf-8'):
        try: done.add(json.loads(line)['date'])
        except Exception: pass
start, end = dt.date(2010, 1, 1), dt.date(2026, 10, 1)
days = [start + dt.timedelta(i) for i in range((end - start).days + 1)]
days = [d for d in days if d.weekday() < 5 and d.isoformat() not in done][::-1]     # 최근부터
print(f'받을 날짜 {len(days)}개 (이미 {len(done)}개)', flush=True)
s = requests.Session(); s.headers.update(H); t0 = time.time(); n = 0; rows_total = 0
with OUT.open('a', encoding='utf-8') as f:
    for d in days:
        ds = d.isoformat(); rows = None
        for k in range(6):
            try:
                r = s.get(f'https://api.nasdaq.com/api/calendar/earnings?date={ds}', timeout=45)
                if r.status_code == 200:
                    j = r.json(); data = j.get('data') or {}
                    rows = data.get('rows') or []; break
                print(f'  {ds} HTTP {r.status_code}; {30 * (k + 1)}초 대기', flush=True)
            except Exception as e:
                print(f'  {ds} 오류 {type(e).__name__}: {e}; {30 * (k + 1)}초 대기', flush=True)
            time.sleep(60 * (k + 1))
        if rows is None:
            FAIL.open('a').write(ds + '\n'); continue
        f.write(json.dumps({'date': ds, 'rows': rows}, ensure_ascii=False) + '\n'); f.flush()
        n += 1; rows_total += len(rows)
        if n % 100 == 0: print(f'  {n}일 받음 ({ds} 까지) 누적 {rows_total}행 {time.time() - t0:.0f}s', flush=True)
        time.sleep(1.8 + random.random() * 0.8)        # 0.6초 간격으론 480일쯤에서 타임아웃이 계속 났다
print(f'끝 {n}일 {rows_total}행 {time.time() - t0:.0f}s', flush=True)
