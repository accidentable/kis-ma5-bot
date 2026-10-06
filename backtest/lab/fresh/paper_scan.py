"""
종이 계좌 스캔 — 대회 모드 규칙(core/contest.py)을 실제 시세로 매일 돌려 '오늘이면 뭘 했을지' 만 기록한다. 주문 없음, 봇 안 건드림, KIS 키 불필요.

    python -m backtest.lab.fresh.paper_scan            # 장 마감 뒤 (cron 15:40 · 16:10 월~금)

데이터  대상 = 한투 종목마스터(공개 파일) 코스피 시총 200 + 코스닥 시총 150 (관리·정지·경고 제외)
        일봉 = 네이버 금융 siseJson (날짜·시가·고가·저가·종가·거래량). 거래대금은 종가 × 거래량으로 근사 (거래 터짐 판정용)
규칙    config 의 CT_* 기본값 그대로 (1종목 · 20일 모멘텀 상위 10% 1위 · 손절 10 · 추적 10 · 보유 10일 · 락 30% · 폭락 전환 5일). 조건용 1주 거래는 안 한다.
체결    매수 = 그날 종가 (비용 0.065%), 매도 = 다음 날 시가 (비용 0.245%). 폭락 전환 매도는 그날 종가.
기록    /data/lab/paper/state.json (종이 계좌), /data/lab/paper/log.md (사람이 읽는 일지), /data/lab/paper/daily.csv (표),
        /data/lab/paper/candles_YYYYMMDD.json (그날 받은 일봉 캐시)
같은 날짜를 두 번 돌려도 한 번만 반영한다 (휴장일·데이터 지연이면 '새 데이터 없음').
"""
from __future__ import annotations

import csv, json, os, sys, time, random, re
from datetime import date, datetime, timedelta

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config                                                         # noqa: E402
from core import universe                                             # noqa: E402
from core.contest import Snap, crash_candidates, crash_signal, exit_reasons, lock_hit, rank_momentum, split_universe   # noqa: E402

DIR = '/data/lab/paper'
STATE = os.path.join(DIR, 'state.json'); LOG = os.path.join(DIR, 'log.md'); CSV = os.path.join(DIR, 'daily.csv')
CAP = 1e8; BUYC, SELLC = 0.00065, 0.00265
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36'
os.makedirs(DIR, exist_ok=True)


def fetch_candles(ticker: str, sess: requests.Session, days: int = 120) -> list[dict]:
    end = date.today(); start = end - timedelta(days=days)
    url = (f'https://api.finance.naver.com/siseJson.naver?symbol={ticker}&requestType=1&startTime={start:%Y%m%d}&endTime={end:%Y%m%d}&timeframe=day')
    for k in range(3):
        try:
            r = sess.get(url, timeout=20)
            if r.status_code == 200:
                rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', r.text)
                return [{'date': d, 'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c), 'volume': float(v)} for d, o, h, l, c, v in rows]
        except requests.RequestException:
            pass
        time.sleep(2 + k * 3)
    return []


def load_candles(uni: list[dict]) -> dict:
    today = date.today().strftime('%Y%m%d'); path = os.path.join(DIR, f'candles_{today}.json')
    if os.path.exists(path):
        try:
            data = json.load(open(path, encoding='utf-8'))
            if len(data) >= len(uni) * 0.8: return data
        except Exception:
            pass
    sess = requests.Session(); sess.headers['User-Agent'] = UA
    data = {}
    for i, s in enumerate(uni):
        c = fetch_candles(s['ticker'], sess)
        if c: data[s['ticker']] = c
        time.sleep(0.2 + random.random() * 0.2)
        if i % 100 == 99: print(f'  일봉 {i + 1}/{len(uni)}', flush=True)
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    for f in sorted(os.listdir(DIR)):                       # 캐시는 최근 5개만
        if f.startswith('candles_') and f < f'candles_{(date.today() - timedelta(days=7)):%Y%m%d}.json': os.remove(os.path.join(DIR, f))
    return data


def load_state() -> dict:
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding='utf-8'))
    return {'start': '', 'cash': CAP, 'positions': [], 'pending': [], 'anchor': CAP, 'month': '', 'locked': False, 'locked_date': '',
            'history': [], 'done': [], 'dates': []}


def save_state(st: dict) -> None:
    json.dump(st, open(STATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def main() -> int:
    uni = split_universe(universe.get_large_universe(config.CT_UNIVERSE_PULL), config.CT_KOSPI_N, config.CT_KOSDAQ_N)
    cand = load_candles(uni)
    # 기준일 = 가장 많은 종목이 가진 최신 날짜
    last = {}
    for t, c in cand.items():
        if c: last[c[-1]['date']] = last.get(c[-1]['date'], 0) + 1
    if not last:
        print('일봉 없음'); return 1
    asof = max(last, key=lambda d: (last[d], d))
    if last[asof] < len(uni) * 0.6:
        print(f'기준일 {asof} 데이터가 {last[asof]}종목뿐 — 중단'); return 1
    st = load_state()
    if asof in st['done']:
        print(f'{asof} 이미 반영됨 — 새 데이터 없음'); return 0
    if not st['start']: st['start'] = asof
    # 거래일 번호 (삼성전자 일봉 날짜 목록)
    dates = sorted({c['date'] for c in cand.get('005930', [])} | set(st['dates']))
    st['dates'] = dates[-120:]
    idx = {d: i for i, d in enumerate(dates)}
    tidx = idx[asof]
    px = {}; snaps = []
    for s in uni:
        c = cand.get(s['ticker']) or []
        if not c or c[-1]['date'] != asof or len(c) < 3: continue
        today, prev = c[-1], c[-2]
        px[s['ticker']] = today
        closes = [x['close'] for x in c[:-1]]; vals = [x['close'] * x['volume'] for x in c[:-1]]
        snaps.append(Snap(ticker=s['ticker'], name=s['name'], market=s['market'], price=today['close'], prev_close=prev['close'],
                          change_pct=(today['close'] / prev['close'] - 1) * 100 if prev['close'] else 0, upper_limit=prev['close'] * 1.295,
                          lower_limit=prev['close'] * 0.705, value=today['close'] * today['volume'], closes=closes, values=vals))
    name_of = {s.ticker: s.name for s in snaps}

    def nav_at(key='close'):
        return st['cash'] + sum(p['qty'] * px[p['ticker']][key] * (1 - SELLC) for p in st['positions'] if p['ticker'] in px)

    log = [f'## {asof[:4]}-{asof[4:6]}-{asof[6:]}  (대상 {len(snaps)}종목)']
    acts = []
    # ① 월초 리셋
    mon = asof[:6]
    if st['month'] != mon:
        st['month'] = mon; st['anchor'] = nav_at('open') if st['positions'] else st['cash']; st['locked'] = False; st['locked_date'] = ''
        acts.append(f'📅 새 달 기준 순자산 {st["anchor"]:,.0f}원')
    # ② 오늘 시가에 어제 정한 매도
    for pe in st['pending']:
        p = next((p for p in st['positions'] if p['ticker'] == pe['ticker']), None)
        if not p: continue
        if p['ticker'] not in px:
            acts.append(f'⚠️ {p["name"]} 시세 없음 — 매도 보류'); continue
        o = px[p['ticker']]['open']; proceeds = p['qty'] * o * (1 - SELLC)
        st['cash'] += proceeds; st['positions'].remove(p)
        ret = o / p['entry_price'] * (1 - SELLC) / (1 + BUYC) - 1
        st['history'].append({**p, 'exit_date': asof, 'exit_price': o, 'ret': ret, 'reason': pe['reason']})
        acts.append(f'🔴 시가 매도 {p["name"]}({p["ticker"]}) {p["qty"]:,}주 @{o:,.0f} → {ret * 100:+.1f}% ({pe["reason"]})')
    st['pending'] = []
    # ③ 판정 · 종가 매수
    crash_on, cinfo = crash_signal(snaps)
    mk = f'시장 {cinfo["mkt"] * 100:+.2f}% (σ20 {cinfo["sigma"] * 100 if cinfo["sigma"] else 0:.2f}%, 폭락 기준 {cinfo["threshold"] * 100 if cinfo["threshold"] else 0:+.2f}%)'
    cands, stats = rank_momentum(snaps)
    top5 = cands[:5]
    held_t = {p['ticker'] for p in st['positions']}
    in_crash = any(p.get('kind') == 'crash' for p in st['positions'])
    mode = 'locked' if st['locked'] else ('crash' if crash_on and not in_crash else ('crash_hold' if in_crash else 'momentum'))
    if mode == 'crash':
        for p in list(st['positions']):                       # 보유를 종가에 팔고
            c_ = px[p['ticker']]['close']; st['cash'] += p['qty'] * c_ * (1 - SELLC); st['positions'].remove(p)
            ret = c_ / p['entry_price'] * (1 - SELLC) / (1 + BUYC) - 1
            st['history'].append({**p, 'exit_date': asof, 'exit_price': c_, 'ret': ret, 'reason': '폭락 전환'})
            acts.append(f'🔴 폭락 전환 매도 {p["name"]} @{c_:,.0f} → {ret * 100:+.1f}%')
        cc, cst = crash_candidates(snaps, cinfo['mkt'])
        picks = cc[:config.CT_SLOTS]
        nav = st['cash']
        for s in picks:
            qty = int(nav * 0.995 / config.CT_SLOTS // (s.price * (1 + BUYC)))
            if qty < 1: continue
            st['cash'] -= qty * s.price * (1 + BUYC)
            st['positions'].append({'ticker': s.ticker, 'name': s.name, 'qty': qty, 'entry_price': s.price, 'entry_date': asof, 'entry_idx': tidx, 'peak_close': s.price, 'kind': 'crash'})
            acts.append(f'🚨 폭락 전환 매수 {s.name}({s.ticker}) {qty:,}주 @{s.price:,.0f} (오늘 {s.change_pct:+.1f}%)')
        log.append(f'- {mk} → **폭락 전환** (급락 후보 {len(cc)}, 탈락 {cst["rejects"]})')
    elif mode == 'momentum':
        free = config.CT_SLOTS - len(st['positions'])
        picks = [s for s in cands if s.ticker not in held_t][:max(free, 0)]
        nav = nav_at('close')
        for s in picks:
            budget = nav * 0.995 / config.CT_SLOTS
            qty = int(min(budget, st['cash']) // (s.price * (1 + BUYC)))
            if qty < 1: continue
            st['cash'] -= qty * s.price * (1 + BUYC)
            st['positions'].append({'ticker': s.ticker, 'name': s.name, 'qty': qty, 'entry_price': s.price, 'entry_date': asof, 'entry_idx': tidx, 'peak_close': s.price, 'kind': 'momentum'})
            acts.append(f'🟢 종가 매수 {s.name}({s.ticker}) {qty:,}주 @{s.price:,.0f} ({config.CT_LOOKBACK}일 {s.ret_n * 100:+.1f}%)')
        log.append(f'- {mk} | 모멘텀 상위 {stats["top"]}/{stats["eligible"]} (컷 {stats["cutoff"] * 100 if stats.get("cutoff") is not None else 0:+.1f}%), 탈락 {stats["rejects"]}')
    else:
        log.append(f'- {mk} | {"🔒 목표 달성 — 현금 유지" if mode == "locked" else "폭락 전환 보유 중"}')
    log.append('- 상위 5: ' + ', '.join(f'{s.name}({s.ticker}) {s.ret_n * 100:+.1f}%/오늘 {s.change_pct:+.1f}%' for s in top5))
    # ④ 마감 판정
    for p in st['positions']:
        if p['ticker'] not in px: continue
        c_ = px[p['ticker']]['close']; p['peak_close'] = max(p['peak_close'], c_)
        if p['entry_date'] == asof: continue
        reasons = exit_reasons(p, c_, tidx)
        if reasons: st['pending'].append({'ticker': p['ticker'], 'reason': '; '.join(reasons)})
    nav = nav_at('close')
    if not st['locked'] and lock_hit(nav, st['anchor']):
        st['locked'] = True; st['locked_date'] = asof
        st['pending'] = [{'ticker': p['ticker'], 'reason': f'목표 +{config.CT_LOCK_PCT:g}% 달성 락'} for p in st['positions']]
        acts.append(f'🎯 목표 달성! 순자산 {nav:,.0f}원 — 내일 시가 전량 매도 후 월말까지 현금')
    for a in acts: log.append(f'- {a}')
    for p in st['positions']:
        c_ = px.get(p['ticker'], {}).get('close', p['entry_price']); held = tidx - p['entry_idx']
        log.append(f'- 📌 보유 {p["name"]}({p["ticker"]}) {p["qty"]:,}주 진입 {p["entry_price"]:,.0f} 종가 {c_:,.0f} {(c_ / p["entry_price"] - 1) * 100:+.1f}% 고점 {p["peak_close"]:,.0f} {held}일 [{p["kind"]}]')
    if st['pending']: log.append('- 내일 시가 매도: ' + ', '.join(f'{name_of.get(x["ticker"], x["ticker"])} ({x["reason"]})' for x in st['pending']))
    log.append(f'- 💰 순자산 {nav:,.0f}원 | 월초 대비 {(nav / st["anchor"] - 1) * 100:+.2f}% | 시작({st["start"]}) 대비 {(nav / CAP - 1) * 100:+.2f}%')
    st['done'].append(asof); st['done'] = st['done'][-400:]
    save_state(st)
    with open(LOG, 'a', encoding='utf-8') as fh: fh.write('\n'.join(log) + '\n\n')
    new = not os.path.exists(CSV)
    with open(CSV, 'a', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(['date', 'market_pct', 'crash', 'mode', 'top1', 'top1_ret20', 'top2', 'top3', 'top4', 'top5', 'action', 'holding', 'nav', 'month_pct'])
        w.writerow([asof, round(cinfo['mkt'] * 100, 2), crash_on, mode, top5[0].name if top5 else '', round(top5[0].ret_n * 100, 1) if top5 else '',
                    *[s.name for s in top5[1:5]] + [''] * (4 - len(top5[1:5])), ' | '.join(acts), ', '.join(p['name'] for p in st['positions']), round(nav), round((nav / st['anchor'] - 1) * 100, 2)])
    print('\n'.join(log))
    return 0


if __name__ == '__main__':
    sys.exit(main())
