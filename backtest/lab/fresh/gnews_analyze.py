"""
급락 사건의 '전날~당일' 구글 뉴스로 뉴스 탓 / 시장 탓을 갈라 수익을 비교한다 (gnews_collect.py 결과 사용)

수익   설계 기준 체결: D 종가 매수 → D+6 시가 매도, 비용 왕복 0.31%. 참고로 D 종가 → D+1 시가 (하룻밤) 도 본다.
기사 분류 (제목만 씀)
  회사명 기사   제목에 회사명이 들어간 기사 (업종 묶음 기사 · 시황 기사는 뺀다)
  급락 보도     제목에 급락 · 폭락 · 하한가 · 추락 · 약세 · 하락 (빠졌다는 보도 자체 — 원인이 아니다)
  악재 키워드   실적(적자 · 어닝쇼크 · 영업손실 …) / 자본조달(유상증자 · 전환사채 …) / 법·규제(소송 · 압수수색 · 횡령 …) /
                바이오(임상 실패 · FDA …) / 신용·상장(감사의견 · 거래정지 · 관리종목 …) / 지분(최대주주 · 블록딜 · 오버행 …) /
                전망(목표주가 하향 · 계약 해지 …)
묶음   G0 기사 없음 / G1 회사명 기사 없음 (시황 · 업종 기사만) / G2 회사명 기사 있으나 악재 키워드 없음 / G3 악재 키워드 있음
비교   폭락일(T5) 사건과 그 외 사건 따로, 기간별로. 거래 터짐(vr ≥ 3) · 잔차(z ≤ −3) 대용 지표와 교차표.
주의   받은 사건은 폭락일 → 최근 순이라, 수집이 끝나기 전엔 최근 연도 비중이 크다. 제목만 보므로 분류는 거칠다.
"""
import json, re, sys
import numpy as np, warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, c = F.o.astype(float), F.c.astype(float); tr = F.traded.astype(bool)
BUYC, SELLC = 0.00065, 0.00265
cf = np.where(tr, c, np.nan)
for t in range(1, T):
    m = np.isnan(cf[t]); cf[t, m] = cf[t - 1, m]
oo = np.where(tr, o, np.nan)
didx = {d: i for i, d in enumerate(F.dates)}; cidx = {cd: j for j, cd in enumerate(F.codes)}
KW = {
    '실적': ['적자', '어닝쇼크', '어닝 쇼크', '실적 쇼크', '실적쇼크', '실적 부진', '영업손실', '순손실', '실적 악화', '하회', '매출 감소', '이익 감소', '실적 충격', '실적 우려'],
    '자본조달': ['유상증자', '유증', '전환사채', '교환사채', '신주인수권', '증자', ' CB', ' BW'],
    '법·규제': ['소송', '제재', '과징금', '검찰', '압수수색', '횡령', '배임', '구속', '기소', '리콜', '공정위', '금감원', '징계', '수사'],
    '바이오': ['임상 실패', '임상 중단', '임상 보류', 'FDA', '보완요구', 'CRL', '승인 거부', '허가 반려', '품목허가 실패', '임상 결과'],
    '신용·상장': ['감사의견', '거래정지', '관리종목', '상장폐지', '불성실공시', '신용등급', '워크아웃', '회생', '부도'],
    '지분': ['최대주주', '블록딜', '지분 매각', '오버행', '대량매도', '장내매도', '매도 공시', '지분 처분', '보호예수'],
    '전망': ['목표주가 하향', '목표가 하향', '투자의견 하향', '하향 조정', '계약 해지', '수주 취소', '계약 취소', '매도 의견', '비중 축소'],
}
DROP = ['급락', '폭락', '하한가', '추락', '약세', '하락', '곤두박질', '휘청', '급락세']


def classify(rec):
    name = rec['name']; items = rec['items']
    named = [it for it in items if name in it['t']]
    cats = set()
    for it in named:
        for k, ws in KW.items():
            if any(w in it['t'] for w in ws): cats.add(k)
    drop = any(any(w in it['t'] for w in DROP) for it in named)
    if not items: g = 'G0 기사 없음'
    elif not named: g = 'G1 회사명 기사 없음'
    elif not cats: g = 'G2 회사명 기사, 악재 없음'
    else: g = 'G3 악재 키워드'
    return g, cats, drop, len(named)


rows = []
for line in open('/data/lab/gnews/events.jsonl', encoding='utf-8'):
    try: r = json.loads(line)
    except Exception: continue
    t, j = didx.get(r['date']), cidx.get(r['code'])
    if t is None or j is None or t + 6 >= T: continue
    r6 = oo[t + 6, j] / c[t, j] * (1 - SELLC) / (1 + BUYC) - 1
    r1n = oo[t + 1, j] / c[t, j] * (1 - SELLC) / (1 + BUYC) - 1
    if not np.isfinite(r6): continue
    g, cats, drop, nn = classify(r)
    rows.append(dict(**{k: r[k] for k in ('date', 'code', 'name', 'r1', 'mkt', 'vr', 'z', 'crash', 'n_items')}, g=g, cats=cats, drop=drop, nn=nn, r6=r6, r1n=r1n,
                     per=('2011~2019' if r['date'] < '20200101' else '2020~2025' if r['date'] < '20260101' else '2026'),
                     proxy_news=((r['vr'] or 0) >= 3) or ((r['z'] if r['z'] is not None else 0) <= -3)))
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P(f'급락 사건 뉴스 분석 — 받은 사건 {len(rows)}건 (폭락일 {sum(r["crash"] for r in rows)}건), 수익 = D 종가 매수 → D+6 시가, 비용 포함')


def st(sel, key='r6'):
    v = np.array([r[key] for r in sel])
    return f'{len(v):4d}건 평균 {v.mean() * 100:+6.2f}% 중앙 {np.median(v) * 100:+6.2f}% 승 {(v > 0).mean() * 100:3.0f}%' if len(v) >= 10 else f'{len(v):4d}건 (표본 부족)'


for crash_flag, title in ((True, '폭락일(T5) 사건'), (False, '그 외 사건 (시장 멀쩡하거나 작은 하락)')):
    P(f'\n■ {title}')
    for per in ('2011~2019', '2020~2025', '2026', '전체'):
        sel0 = [r for r in rows if r['crash'] == crash_flag and (per == '전체' or r['per'] == per)]
        if len(sel0) < 10: continue
        P(f'  [{per}] {len(sel0)}건')
        for g in ('G0 기사 없음', 'G1 회사명 기사 없음', 'G2 회사명 기사, 악재 없음', 'G3 악재 키워드'):
            P(f'    {g:<22} {st([r for r in sel0 if r["g"] == g])}  | 하룻밤 {st([r for r in sel0 if r["g"] == g], "r1n").split("평균")[1][:8] if len([r for r in sel0 if r["g"] == g]) >= 10 else ""}')
        P(f'    {"급락 보도만 (G2 중)":<22} {st([r for r in sel0 if r["g"].startswith("G2") and r["drop"]])}')
        P(f'    {"대용지표: 뉴스 의심(vr≥3|z≤−3)":<22} {st([r for r in sel0 if r["proxy_news"]])}')
        P(f'    {"대용지표: 시장 탓":<22} {st([r for r in sel0 if not r["proxy_news"]])}')
        P(f'    {"악재 없음 & 대용지표 시장 탓":<22} {st([r for r in sel0 if r["g"] != "G3 악재 키워드" and not r["proxy_news"]])}')
        P(f'    {"악재 있음 | 대용지표 뉴스":<22} {st([r for r in sel0 if r["g"] == "G3 악재 키워드" or r["proxy_news"]])}')

P('\n■ 악재 종류별 (전체 사건)')
for k in KW:
    P(f'  {k:<8} {st([r for r in rows if k in r["cats"]])}')
P('\n■ 회사명 기사 수별 (전체 사건) — 관심이 몰린 급락일수록?')
for lo, hi, lab_ in ((0, 0, '0건'), (1, 2, '1~2건'), (3, 5, '3~5건'), (6, 10, '6~10건'), (11, 999, '11건↑')):
    P(f'  {lab_:<8} {st([r for r in rows if lo <= r["nn"] <= hi])}')
P('\n■ 교차표: 제목 분류 × 대용지표 (전체, 건수 / 평균)')
for g in ('G0 기사 없음', 'G1 회사명 기사 없음', 'G2 회사명 기사, 악재 없음', 'G3 악재 키워드'):
    a = [r for r in rows if r['g'] == g and not r['proxy_news']]; b = [r for r in rows if r['g'] == g and r['proxy_news']]
    P(f'  {g:<22} 시장탓 {st(a)} | 뉴스의심 {st(b)}')
P('\n■ 악재 키워드 사건 예 (최근 10건)')
for r in [r for r in rows if r['g'] == 'G3 악재 키워드'][:10]:
    P(f'  {r["date"]} {r["name"]} {r["r1"] * 100:+.1f}% → {r["r6"] * 100:+.1f}% {sorted(r["cats"])}')
open('backtest/results/lab_gnews_dips.md', 'w', encoding='utf-8').write(
    '# 급락 사건 구글 뉴스 분석 (backtest/lab/fresh/gnews_analyze.py)\n\n정의는 스크립트 머리말 참고. 수집 상태에 따라 사건 수가 달라진다.\n\n```\n' + '\n'.join(lines) + '\n```\n')
