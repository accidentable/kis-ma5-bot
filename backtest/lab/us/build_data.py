"""
미국 주식 일봉 데이터 만들기 — 나스닥 스크리너(전 상장주 시총) + 야후 파이낸스(일봉), 키 불필요

    python -m backtest.lab.us.build_data            → /data/lab/us/prices.npz, meta.json

대상   나스닥 스크리너에 올라온 미국 상장 보통주 중 시총 $2B 이상 (우선주·워런트·유닛 제외) + 레버리지 ETF 목록 (LEV 아래)
기간   2010-01-01 ~ 오늘. 수정 종가(Adj Close)로 수익률, Close 로 가격 수준.
주의   지금 상장된 종목만 들어간다 (생존 편향). 상장폐지·합병된 종목이 빠져 있어 모멘텀 · 꼬리 전략 결과가 실제보다 좋게 나온다.
"""
import io, json, os, sys, time, re
import numpy as np, pandas as pd, requests

DIR = '/data/lab/us'; os.makedirs(DIR, exist_ok=True)
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36'
LEV = ['TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'UPRO', 'SPXU', 'TNA', 'TZA', 'LABU', 'LABD', 'FNGU', 'TECL', 'TECS', 'FAS', 'FAZ', 'NUGT', 'JNUG',
       'UDOW', 'YINN', 'YANG', 'KORU', 'NVDL', 'TSLL', 'CONL', 'MSTU', 'AMDL', 'NVDX', 'TSLT', 'AAPU', 'GGLL', 'AMZU', 'MSFU', 'NFLU',
       'DFEN', 'DPST', 'DRN', 'ERX', 'GUSH', 'HIBL', 'MIDU', 'NAIL', 'PILL', 'RETL', 'TPOR', 'UTSL', 'WANT', 'WEBL', 'CURE', 'DUSL', 'BNKU',
       'UVXY', 'SVXY', 'TMF', 'TMV', 'BOIL', 'KOLD', 'UCO', 'SCO', 'QLD', 'SSO', 'ROM', 'USD', 'BIB', 'UBT', 'TBT', 'SPY', 'QQQ', 'IWM', 'SOXX', 'XLK', 'ARKK', 'GLD', 'TLT']


def screener() -> pd.DataFrame:
    r = requests.get('https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true', timeout=60,
                     headers={'User-Agent': UA, 'Accept': 'application/json'})
    rows = r.json()['data']['rows']
    df = pd.DataFrame(rows)
    df['marketCap'] = pd.to_numeric(df['marketCap'].str.replace(r'[$,]', '', regex=True), errors='coerce')
    df['symbol'] = df['symbol'].str.strip()
    ok = df['symbol'].str.match(r'^[A-Z]{1,5}$') & (df['marketCap'] >= 2e9) & ~df['name'].str.contains('Preferred|Warrant|Unit|Depositary|Notes|%', case=False, regex=True)
    df = df[ok].sort_values('marketCap', ascending=False)
    return df[['symbol', 'name', 'marketCap', 'sector', 'industry', 'country', 'ipoyear']].reset_index(drop=True)


def main() -> int:
    import yfinance as yf
    t0 = time.time()
    sc = screener()
    print(f'스크리너: 시총 $2B↑ 보통주 {len(sc)}개 ({time.time() - t0:.0f}s)', flush=True)
    syms = sc['symbol'].tolist() + [s for s in LEV if s not in set(sc['symbol'])]
    frames = {}
    B = 150
    for i in range(0, len(syms), B):
        batch = syms[i:i + B]
        for k in range(3):
            try:
                df = yf.download(batch, start='2010-01-01', auto_adjust=False, progress=False, group_by='ticker', threads=True)
                break
            except Exception as e:
                print(f'  배치 {i} 실패 {e}; 재시도', flush=True); time.sleep(10)
        else:
            continue
        for s in batch:
            try:
                d = df[s].dropna(subset=['Close'])
            except KeyError:
                continue
            if len(d) >= 60: frames[s] = d
        print(f'  {min(i + B, len(syms))}/{len(syms)} 받음, 유효 {len(frames)} ({time.time() - t0:.0f}s)', flush=True)
        time.sleep(1)
    dates = sorted(set().union(*[set(d.index) for d in frames.values()]))
    dates = [d for d in dates if d >= pd.Timestamp('2010-01-01')]
    didx = {d: i for i, d in enumerate(dates)}
    tickers = sorted(frames)
    T, N = len(dates), len(tickers)
    arr = {k: np.full((T, N), np.nan, np.float32) for k in ('open', 'high', 'low', 'close', 'adj', 'volume')}
    for j, s in enumerate(tickers):
        d = frames[s]; d = d[d.index >= pd.Timestamp('2010-01-01')]
        ii = np.array([didx[x] for x in d.index])
        arr['open'][ii, j] = d['Open'].values; arr['high'][ii, j] = d['High'].values; arr['low'][ii, j] = d['Low'].values
        arr['close'][ii, j] = d['Close'].values; arr['adj'][ii, j] = d['Adj Close'].values; arr['volume'][ii, j] = d['Volume'].values
    np.savez_compressed(os.path.join(DIR, 'prices.npz'), dates=np.array([d.strftime('%Y%m%d') for d in dates]), tickers=np.array(tickers), **arr)
    meta = {r['symbol']: {'name': r['name'], 'cap': float(r['marketCap']), 'sector': r['sector'] or '', 'industry': r['industry'] or '', 'ipo': str(r['ipoyear'] or '')}
            for _, r in sc.iterrows()}
    for s in LEV: meta.setdefault(s, {'name': s, 'cap': 0.0, 'sector': 'ETF', 'industry': 'leveraged' if s not in ('SPY', 'QQQ', 'IWM', 'SOXX', 'XLK', 'ARKK', 'GLD', 'TLT') else 'etf', 'ipo': ''})
    json.dump(meta, open(os.path.join(DIR, 'meta.json'), 'w'), ensure_ascii=False)
    print(f'저장: {T}일 × {N}종목, {time.time() - t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
