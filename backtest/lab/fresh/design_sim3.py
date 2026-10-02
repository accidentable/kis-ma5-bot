"""
설계 시뮬레이터 3차 — 트리거 품질: 변동성 기준과 절대 기준을 둘 다 요구

2차 진단: T2(−2.5σ) 는 조용한 장(σ20 0.5~0.8%) 에서 −1.2~−2% 하락에도 발동했고 그런 사건은 대부분 손해였다 (2021년 전부, 2024-07-25 등).
T1(−3% 절대) 은 2026 같은 고변동 장에서 −3~−5% 하락에 발동해 손해였다. 둘 다 만족할 때만 발동하면 어느 장에서든 '그 시기 기준으로 큰 폭락' 만 잡는다.

격자 (미리 정함, 2 × 2 × 2 = 8)
  trig   T4 = (시장 −3%↓) & (−2.5σ↓)   /  T5 = (시장 −3%↓) & (−3σ↓)
  kc     5 / 10
  staged 일괄 / 분할
고정   base 신고가 k5 · 교체일 시가 매도/종가 매수 · overlay 트리거일 종가 매수 · 5일 뒤 시가 매도 · regime 필터 없음 (2차에서 탈락)
추가 출력  연도별 경로 수익 (base 만 · 각 설정), T4 트리거일 목록과 7일 결과
선택 기준  2차와 같음
"""
import numpy as np, warnings, time, itertools, sys; warnings.filterwarnings('ignore')
from backtest.lab.fresh import design_sim2 as S
from backtest.lab import panic2_common as C

F, T = S.F, S.T
mkt = S.mret
sig20 = np.full(T, np.nan)
for t in range(21, T): sig20[t] = np.nanstd(mkt[t - 20:t])
S.TRIG['T4'] = (mkt <= -0.03) & (mkt <= -2.5 * sig20)
S.TRIG['T5'] = (mkt <= -0.03) & (mkt <= -3.0 * sig20)

if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('3차 설계 — 트리거 = 절대 & 변동성 둘 다 · base 신고가 k5 · 1억 · 비용 포함')
    P('  열: 2011~19 월평균 하위10% | 2020~25 월평균 하위10% 중앙 −10%↓ +10%↑ | 2026 월평균 하위10% | 복리 경로 구간별 | 전기간 연환산 MDD | 2020~25 월 매매금액 매수건수')
    RES = {}
    for trig, kc, staged in itertools.product(('T4', 'T5'), (5, 10), (False, True)):
        label = f'{trig} h5 kc{kc:<2}       {"분할" if staged else "일괄"}'
        RES[label] = S.evaluate((trig, 5, kc, False, staged), label, P)
    for label, cfg in (('T1 h5 kc5        일괄', ('T1', 5, 5, False, False)), ('T2 h5 kc5        일괄', ('T2', 5, 5, False, False)),
                       ('T0 base만            ', ('T0', 5, 5, False, False))):
        RES[label] = S.evaluate(cfg, label, P)
    d0 = C.didx(F, '20110103')
    P('\n■ 연도별 경로 수익')
    P('  연도  ' + ' '.join(f'{k[:20]:>20}' for k in RES))
    for y in range(2011, 2027):
        i0 = max(C.didx(F, f'{y}0101') - d0 - 1, 0); i1 = min(C.didx(F, f'{y + 1}0101') - d0 - 1, T - 1 - d0)
        P(f'  {y}  ' + ' '.join(f'{(v["navs"][i1] / v["navs"][i0] - 1) * 100:>+19.1f}%' for v in RES.values()))
    P('\n■ 트리거 횟수 (연평균): ' + ', '.join(f'{k} {S.TRIG[k][d0:].sum() / ((T - d0) / 246):.1f}회' for k in ('T1', 'T2', 'T4', 'T5')))
    nav = RES['T4 h5 kc5        일괄']['navs']
    P('\n■ T4 트리거일 전부 — 시장 등락 · σ20 · 트리거 전날 → +7일 NAV 변화')
    prev_end = -99
    for t in np.where(S.TRIG['T4'])[0]:
        if t < d0 or t + 7 - d0 >= len(nav): continue
        tag = ' (앞 사건 보유 중)' if t - prev_end <= 6 else ''
        if not tag: prev_end = t + 6
        P(f'  {F.dates[t]} 시장 {mkt[t] * 100:+5.1f}% σ20 {sig20[t] * 100:4.1f}% | {(nav[t + 7 - d0] / nav[t - 1 - d0] - 1) * 100:+6.1f}%{tag}')
    import pickle; pickle.dump({k: dict(ann=v['ann'], mdd=v['mdd'], seg=v['seg'], navs=v['navs'], m={pn: r[0] for pn, r in v['rows'].items()},
                                        amt={pn: r[1] for pn, r in v['rows'].items()}, buys={pn: r[2] for pn, r in v['rows'].items()}) for k, v in RES.items()},
                               open('/data/lab/design_sim3.pkl', 'wb'))
    open('backtest/results/lab_design_sim3.md', 'w', encoding='utf-8').write(
        '# 설계 시뮬레이터 3차 — 트리거 품질 (backtest/lab/fresh/design_sim3.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
