"""
급락 사건의 '전날~당일' 공시(OpenDART)로 악재 공시 유무를 갈라 수익을 비교한다 (dart_collect.py 결과 사용)

수익   D 종가 매수 → D+6 시가 매도, 비용 왕복 0.31% (설계 체결). 하룻밤(D 종가 → D+1 시가)도 본다.
공시 분류 (report_nm 으로)
  악재   유상증자 · 전환사채 · 신주인수권부사채 · 교환사채 · 감자 / 감사의견(거절·한정·부적정) · 상장폐지 · 관리종목 · 불성실공시 · 매매거래정지 · 투자주의환기
         / 소송등의제기 · 행정처분 · 영업정지 · 과징금 / 계약해지 · 공급계약해지 / 최대주주변경 / 중대재해 / 횡령·배임
  실적   영업(잠정)실적 · 손익구조변동 · 분기·반기·사업보고서 (좋은지 나쁜지는 제목만으론 모른다)
  조회   조회공시요구(현저한시황변동) — 거래소가 "왜 빠졌냐" 고 물은 것 (알려진 이유가 없다는 뜻) / 풍문·보도 조회·해명
  투자판단  투자판단관련주요경영사항 (임상 결과 등)
  호재   자기주식취득 · 공급계약체결 · 무상증자 · 현금배당
  기타   나머지 (대량보유·임원소유상황·증권신고서 등)
묶음   D0 공시 없음 / D1 기타·호재만 / D2 실적 (악재 없음) / D3 악재 공시 / 조회·투자판단은 따로 표시
주의   list.json 은 접수일만 준다. D 당일 공시는 장 마감 뒤(우리가 산 뒤) 것일 수 있어 '원인' 과 '결과' 가 섞인다.
       그래서 전날(D−1) 공시만으로 나눈 것도 같이 본다. corp_code 가 없는(상장폐지 등) 사건은 뺀다.
"""
import json, sys
import numpy as np, warnings; warnings.filterwarnings('ignore')
from collections import Counter, defaultdict
from backtest.lab import panic2_feat as PF

F = PF.load(); T, N = F.T, F.N
o, c = F.o.astype(float), F.c.astype(float); tr = F.traded.astype(bool)
BUYC, SELLC = 0.00065, 0.00265
oo = np.where(tr, o, np.nan)
didx = {d: i for i, d in enumerate(F.dates)}; cidx = {cd: j for j, cd in enumerate(F.codes)}
NEG = ['유상증자', '전환사채', '신주인수권부사채', '교환사채', '감자', '의견거절', '한정', '부적정', '상장폐지', '관리종목', '불성실공시', '매매거래정지',
       '투자주의환기', '소송등의제기', '행정처분', '영업정지', '과징금', '계약해지', '최대주주변경', '중대재해', '횡령', '배임', '회생절차', '파산']
EARN = ['영업(잠정)실적', '잠정실적', '손익구조', '분기보고서', '반기보고서', '사업보고서', '영업실적']
INQ = ['조회공시요구(현저한시황변동)', '조회공시 요구(현저한 시황변동)', '현저한시황변동']
RUMOR = ['풍문', '보도']
INVEST = ['투자판단관련주요경영사항', '투자판단 관련 주요경영사항']
POS = ['자기주식취득', '공급계약체결', '무상증자', '현금ㆍ현물배당', '현금배당']


def classify(items):
    cats = set()
    for it in items:
        nm = it['nm'].replace(' ', '')
        if any(k.replace(' ', '') in nm for k in NEG): cats.add('악재')
        if any(k.replace(' ', '') in nm for k in EARN): cats.add('실적')
        if any(k.replace(' ', '') in nm for k in INQ): cats.add('조회')
        if any(k in nm for k in RUMOR) and '조회' in nm or '해명' in nm: cats.add('풍문')
        if any(k.replace(' ', '') in nm for k in INVEST): cats.add('투자판단')
        if any(k.replace(' ', '') in nm for k in POS): cats.add('호재')
    return cats


def group(items):
    if not items: return 'D0 공시 없음'
    cats = classify(items)
    if '악재' in cats: return 'D3 악재 공시'
    if '실적' in cats: return 'D2 실적 공시'
    return 'D1 기타·호재만'


rows = []
for line in open('/data/lab/dart/events_dart.jsonl', encoding='utf-8'):
    try: r = json.loads(line)
    except Exception: continue
    if r['items'] is None: continue
    t, j = didx.get(r['date']), cidx.get(r['code'])
    if t is None or j is None or t + 6 >= T: continue
    r6 = oo[t + 6, j] / c[t, j] * (1 - SELLC) / (1 + BUYC) - 1
    r1n = oo[t + 1, j] / c[t, j] * (1 - SELLC) / (1 + BUYC) - 1
    if not np.isfinite(r6): continue
    prev = [it for it in r['items'] if it['dt'] < r['date']]; same = [it for it in r['items'] if it['dt'] == r['date']]
    rows.append(dict(**{k: r[k] for k in ('date', 'code', 'name', 'r1', 'mkt', 'vr', 'z', 'crash', 'n_items')}, items=r['items'],
                     g=group(r['items']), gprev=group(prev), cats=classify(r['items']), r6=r6, r1n=r1n,
                     per=('2011~2019' if r['date'] < '20200101' else '2020~2025' if r['date'] < '20260101' else '2026'),
                     proxy_news=((r['vr'] or 0) >= 3) or ((r['z'] if r['z'] is not None else 0) <= -3)))
# 네이버 뉴스 묶음 붙이기 (있으면)
NV = {}
try:
    sys.argv = [sys.argv[0], 'naver']
    import importlib; ga = importlib.import_module('backtest.lab.fresh.gnews_analyze')
    NV = {(x['date'], x['code']): x['g'] for x in ga.rows}
except Exception as e:
    print('네이버 뉴스 결합 실패:', e)
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
P(f'급락 사건 공시 분석 [OpenDART] — 사건 {len(rows)}건 (폭락일 {sum(r["crash"] for r in rows)}건), 수익 = D 종가 매수 → D+6 시가, 비용 포함')


def st(sel, key='r6'):
    v = np.array([r[key] for r in sel])
    return f'{len(v):4d}건 평균 {v.mean() * 100:+6.2f}% 중앙 {np.median(v) * 100:+6.2f}% 승 {(v > 0).mean() * 100:3.0f}%' if len(v) >= 10 else f'{len(v):4d}건 (표본 부족)'


GS = ('D0 공시 없음', 'D1 기타·호재만', 'D2 실적 공시', 'D3 악재 공시')
for crash_flag, title in ((True, '폭락일(T5) 사건'), (False, '그 외 사건 (시장 멀쩡하거나 작은 하락)')):
    P(f'\n■ {title}')
    for per in ('2011~2019', '2020~2025', '2026', '전체'):
        sel0 = [r for r in rows if r['crash'] == crash_flag and (per == '전체' or r['per'] == per)]
        if len(sel0) < 10: continue
        P(f'  [{per}] {len(sel0)}건 · 공시 있는 사건 {sum(r["n_items"] > 0 for r in sel0)}건')
        for g in GS:
            s_ = [r for r in sel0 if r['g'] == g]
            P(f'    {g:<16} {st(s_)}' + (f'  | 하룻밤 {np.mean([r["r1n"] for r in s_]) * 100:+5.2f}%' if len(s_) >= 10 else ''))
        P(f'    {"전날(D−1) 공시만: 악재":<16} {st([r for r in sel0 if r["gprev"] == "D3 악재 공시"])}')
        P(f'    {"전날 공시만: 악재 없음":<16} {st([r for r in sel0 if r["gprev"] != "D3 악재 공시"])}')
        P(f'    {"시황변동 조회공시 요구":<16} {st([r for r in sel0 if "조회" in r["cats"]])}')
        P(f'    {"투자판단 주요경영사항":<16} {st([r for r in sel0 if "투자판단" in r["cats"]])}')
        P(f'    {"대용지표 뉴스 의심":<16} {st([r for r in sel0 if r["proxy_news"]])}')
        P(f'    {"악재 공시 없음 & 대용지표 시장 탓":<16} {st([r for r in sel0 if r["g"] != "D3 악재 공시" and not r["proxy_news"]])}')
        P(f'    {"악재 공시 | 대용지표 뉴스":<16} {st([r for r in sel0 if r["g"] == "D3 악재 공시" or r["proxy_news"]])}')

P('\n■ 교차표: 공시 × 대용지표 (전체)')
for g in GS:
    P(f'  {g:<16} 시장탓 {st([r for r in rows if r["g"] == g and not r["proxy_news"]])} | 뉴스의심 {st([r for r in rows if r["g"] == g and r["proxy_news"]])}')
if NV:
    P('\n■ 교차표: 공시 × 네이버 뉴스 (겹치는 사건)')
    for g in GS:
        for ng in ('G0 기사 없음', 'G1 회사명 기사 없음', 'G2 회사명 기사, 악재 없음', 'G3 악재 키워드'):
            s_ = [r for r in rows if r['g'] == g and NV.get((r['date'], r['code'])) == ng]
            if len(s_) >= 10: P(f'  {g:<16} × {ng:<22} {st(s_)}')
P('\n■ 악재 공시 사건 — 공시 제목별 (건수 · 수익)')
cnt = defaultdict(list)
for r in rows:
    if r['g'] == 'D3 악재 공시':
        for it in r['items']:
            nm = it['nm'].strip()
            if any(k.replace(' ', '') in nm.replace(' ', '') for k in NEG): cnt[nm[:40]].append(r['r6'])
for nm, v in sorted(cnt.items(), key=lambda x: -len(x[1]))[:15]:
    P(f'  {len(v):3d}건 평균 {np.mean(v) * 100:+6.1f}%  {nm}')
P('\n■ 가장 흔한 공시 제목 (전체 사건)')
for nm, k in Counter(it['nm'].strip()[:45] for r in rows for it in r['items']).most_common(15): P(f'  {k:4d}  {nm}')
open('backtest/results/lab_dart_dips.md', 'w', encoding='utf-8').write(
    '# 급락 사건 공시 분석 [OpenDART] (backtest/lab/fresh/dart_analyze.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
