"""
설계 시뮬레이터 4차 — overlay 후보의 낙폭 기준: −5% / −7% / −10% (사용자 질문: "10% 는 너무 크지 않나, 5% 가 적당해 보인다")

고정   base 신고가 k5 · 교체일 시가 매도/종가 매수 · 트리거 T5 (시장 −3% & −3σ) · 트리거일 종가 매수 · 5일 뒤 시가 매도 · 일괄 진입
변경   overlay 후보 = 그날 등락이 [기준, −20%) 인 종목 (하한가 제외), 더 빠진 순. 기준 −5% / −7% / −10%. 종목 수 5 / 10.
추가   ① 폭락일(T5) 사건 수준: 낙폭 구간별(−5~−7 / −7~−10 / −10~−20) 5일 수익 (종가 → D+6 시가, 비용 포함), 대용지표 필터 적용 전후
       ② 폭락일마다 후보 수: −10% 기준으로 5개를 못 채운 날이 얼마나 되나
"""
import numpy as np, warnings, time, sys; warnings.filterwarnings('ignore')
from collections import defaultdict
from backtest.lab.fresh import design_sim3 as S3          # T5 트리거 등록
from backtest.lab.fresh import design_sim2 as S
from backtest.lab import panic2_common as C

F, T, r1, U, tr, o, c, cf = S.F, S.T, S.r1, S.D.U, S.tr, S.o, S.c, S.cf
limdown = S.D.limdown
BUYC, SELLC = S.BUYC, S.SELLC
vr, z = F.vr.astype(float), F.z.astype(float)
T5 = S.TRIG['T5']


def dip_lists(thr):
    m = U & np.isfinite(r1) & (r1 <= thr) & (r1 > -0.20) & ~limdown
    return m, [np.where(m[t])[0][np.argsort(r1[t, m[t]])].tolist() for t in range(T)]


if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('4차 — overlay 후보 낙폭 기준 비교 (T5 · 5일 · 종가 매수 · 일괄). 열은 2차와 같음')
    for thr in (-0.05, -0.07, -0.10):
        m, S.DIPL = dip_lists(thr)
        for kc in (5, 10):
            S.evaluate(('T5', 5, kc, False, False), f'후보 {thr * 100:+.0f}%↓ kc{kc:<2}          ', P)
    def lead(x, k): out = np.full_like(x, np.nan); out[:-k] = x[k:]; return out
    ret = lead(np.where(tr, o, np.nan), 6) / np.where(tr, c, np.nan) * (1 - SELLC) / (1 + BUYC) - 1
    proxy = (np.nan_to_num(vr, nan=0) >= 3) | (np.nan_to_num(z, nan=0) <= -3)
    P('\n■ 폭락일(T5) 사건 — 낙폭 구간별 5일 수익 (종가 → D+6 시가, 비용 포함). 날짜평균 = 같은 날 묶음')
    base = U & np.isfinite(r1) & ~limdown & T5[:, None]
    for lo, hi, nm in ((-0.07, -0.05, '−5~−7%'), (-0.10, -0.07, '−7~−10%'), (-0.13, -0.10, '−10~−13%'), (-0.20, -0.13, '−13~−20%')):
        for fn, fm in (('전체', np.ones_like(base)), ('대용지표 시장탓만', ~proxy)):
            mm = base & (r1 > lo) & (r1 <= hi) & fm
            tt, jj = np.where(mm); v = ret[tt, jj]; ok = np.isfinite(v); tt, v = tt[ok], v[ok]
            d = defaultdict(list)
            for t_, x in zip(tt, v): d[t_].append(x)
            dm = np.array([np.mean(x) for x in d.values()])
            P(f'  {nm:<9} {fn:<10} {len(v):4d}건 평균 {v.mean() * 100:+6.2f}% 중앙 {np.median(v) * 100:+6.2f}% 승 {(v > 0).mean() * 100:3.0f}% | 날짜평균 {dm.mean() * 100:+6.2f}% ({len(dm)}일)')
    P('\n■ 폭락일마다 후보 수 (하한가 제외)')
    days = np.where(T5)[0]; days = days[days >= C.didx(F, '20110103')]
    for thr in (-0.05, -0.07, -0.10):
        m, _ = dip_lists(thr); cnt = m[days].sum(1)
        P(f'  기준 {thr * 100:+.0f}%: 폭락일 {len(days)}일, 후보 중앙값 {np.median(cnt):.0f}개, 5개 미만인 날 {(cnt < 5).sum()}일, 10개 미만 {(cnt < 10).sum()}일')
    open('backtest/results/lab_design_sim4.md', 'w', encoding='utf-8').write(
        '# 설계 시뮬레이터 4차 — overlay 후보 낙폭 기준 (backtest/lab/fresh/design_sim4.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
