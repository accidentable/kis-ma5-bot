"""
설계 시뮬레이터 6차 — overlay 종목 필터를 실제로 적용했을 때 (대용지표 · DART 악재 공시 · 네이버 악재 기사)

고정   base 신고가 k5 · 교체일 시가 매도/종가 매수 · 트리거 T5 · 트리거일 종가 매수 · 5일 뒤 시가 매도 · 5종목 · 일괄 · 후보 낙폭 −7%↓ (4차 결론)
필터 (후보에서 뺀다)
  F0 없음
  F1 대용지표     그날 거래대금 ≥ 20일 평균 × 3  또는 잔차 z ≤ −3 (베타 뺀 하락이 평소의 3배)
  F2 F1 + DART    전날~당일 악재 공시 (dart_analyze 의 NEG 목록) 가 있는 종목 — /data/lab/dart/events_dart.jsonl 에 있는 사건만 판단 가능
  F3 F2 + 네이버   전날~당일 악재 기사 (gnews_analyze 의 KW 목록, 제목에 회사명 포함) — events_naver.jsonl 에 있는 사건만
  ※ 공시 · 기사는 −10%↓ 사건만 수집했다. −7~−10% 후보는 공시 · 기사 필터를 못 거치고 대용지표만 거친다.
평가   2차와 같음
"""
import json, numpy as np, warnings, time, sys; warnings.filterwarnings('ignore')
from backtest.lab.fresh import design_sim3 as S3
from backtest.lab.fresh import design_sim2 as S
from backtest.lab.fresh import dart_analyze as DA
from backtest.lab import panic2_common as C

F, T, r1, U, tr = S.F, S.T, S.r1, S.D.U, S.tr
limdown = S.D.limdown
vr, z = F.vr.astype(float), F.z.astype(float)
didx = {d: i for i, d in enumerate(F.dates)}; cidx = {cd: j for j, cd in enumerate(F.codes)}
proxy = (np.nan_to_num(vr, nan=0) >= 3) | (np.nan_to_num(z, nan=0) <= -3)
bad_dart = np.zeros((T, N := S.D.N), bool); have_dart = np.zeros((T, N), bool)
for line in open('/data/lab/dart/events_dart.jsonl', encoding='utf-8'):
    try: r = json.loads(line)
    except Exception: continue
    t, j = didx.get(r['date']), cidx.get(r['code'])
    if t is None or j is None or r['items'] is None: continue
    have_dart[t, j] = True
    if DA.group(r['items']) == 'D3 악재 공시': bad_dart[t, j] = True
bad_news = np.zeros((T, N), bool); have_news = np.zeros((T, N), bool)
try:
    sys.argv = [sys.argv[0], 'naver']
    import importlib; GA = importlib.import_module('backtest.lab.fresh.gnews_analyze')
    for x in GA.rows:
        t, j = didx.get(x['date']), cidx.get(x['code'])
        if t is None or j is None: continue
        have_news[t, j] = True
        if x['g'] == 'G3 악재 키워드': bad_news[t, j] = True
except Exception as e:
    print('네이버 결합 실패:', e, file=sys.stderr)
base_m = U & np.isfinite(r1) & (r1 <= -0.07) & (r1 > -0.20) & ~limdown
FILTERS = {'F0 없음': base_m, 'F1 대용지표': base_m & ~proxy, 'F2 +DART 악재공시': base_m & ~proxy & ~bad_dart, 'F3 +네이버 악재기사': base_m & ~proxy & ~bad_dart & ~bad_news}

if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    T5d = np.where(S.TRIG['T5'])[0]; T5d = T5d[T5d >= C.didx(F, '20110103')]
    P(f'6차 — overlay 종목 필터 적용 (후보 −7%↓, T5 폭락일 {len(T5d)}일). 폭락일 후보 중 공시 판단 가능 {have_dart[T5d].sum()}건 · 기사 판단 가능 {have_news[T5d].sum()}건')
    for fn, fm in FILTERS.items():
        P(f'  {fn:<16} 폭락일 후보 합계 {fm[T5d].sum():4d}건 · 날짜별 중앙값 {np.median(fm[T5d].sum(1)):.0f}개 · 5개 미만인 날 {(fm[T5d].sum(1) < 5).sum()}일')
    P('\n  열: 2011~19 월평균 하위10% | 2020~25 월평균 하위10% 중앙 −10%↓ +10%↑ | 2026 | 경로 | 연 MDD | 월 매매금액 매수건수')
    for fn, fm in FILTERS.items():
        S.DIPL = [np.where(fm[t])[0][np.argsort(r1[t, fm[t]])].tolist() for t in range(T)]
        S.evaluate(('T5', 5, 5, False, False), f'{fn:<16} kc5', P)
    open('backtest/results/lab_design_sim6.md', 'w', encoding='utf-8').write(
        '# 설계 시뮬레이터 6차 — overlay 종목 필터 (backtest/lab/fresh/design_sim6.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
