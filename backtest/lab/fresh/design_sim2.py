"""
설계 시뮬레이터 2차 — 1차에서 남은 설계(신고가 k5 · 교체일 시가 매도/종가 매수 · overlay 트리거 당일 종가 매수) 주변을 좁혀 본다

1차 결과: 월평균은 overlay 가 높였지만 2020~2025 복리 경로는 오히려 낮았다 (폭락이 며칠 이어지면 첫 폭락일 종가에 전부 넣은 돈이
연속으로 깨진다). 그래서 2차는 '한 번에 다 넣지 않기' 와 '약세장에서 base 비우기' 를 추가한다.

고정   base = 신고가 k5 (시총 200 · 60일 수익률 > 0 · 20일 내 급등일 없음 · 종가/250일 고가 순), 21거래일 교체, 교체일 시가 매도·종가 매수,
       보유 중 +10%↑ 급등일 종가 매도. overlay 종목 = 그날 −10~−20% (하한가 제외) 더 빠진 순. overlay 매수 = 트리거일 종가, 매도 = h 일 뒤 시가.
격자 (미리 정함, 2 × 3 × 2 × 2 × 2 = 48)
  trig    T1 시장 −3%↓ / T2 시장 −2.5σ↓ (σ = 직전 20일 시장 일간 변동성)
  hold    3 / 5 / 8 거래일
  kc      overlay 종목 수 5 / 10
  regime  없음 / MA120: 시장(대상 동일가중 지수) 이 전날 120일 이동평균 아래면 base 를 시가에 전부 팔고 현금 (overlay 는 그대로)
  staged  없음 / 분할: 첫 트리거에 절반(base 절반 매도), 보유 중 또 트리거면 나머지 절반 (그날 급락주로). 그 뒤 트리거는 무시
평가   1차와 같음 (20일 월 분포 2011~19 / 2020~25 / 2026, 전 기간 복리 경로 · MDD · 구간 수익)
선택 기준 (미리)  2020~2025 의 복리 경로와 월 하위 10% 를 가장 중시. 2011~2019 월평균이 시장(+0.08) 이하면 탈락. 가장 단순한 쪽 우선.
"""
import numpy as np, warnings, time, itertools, sys; warnings.filterwarnings('ignore')
from backtest.lab.fresh import design_sim as D
from backtest.lab import panic2_common as C

F, T, o, c, cf, tr, r1 = D.F, D.T, D.o, D.c, D.cf, D.tr, D.r1
BUYC, SELLC, CAP, MONTH, R_DAYS = D.BUYC, D.SELLC, D.CAP, D.MONTH, D.R_DAYS
NH, surge, TRIG, PER = D.NH, D.surge, D.TRIG, D.PER
mret = np.nan_to_num(np.nanmean(np.where(D.U, np.clip(r1, -.3, .3), np.nan), axis=1))
idx = np.cumprod(1 + mret)
ma120 = np.full(T, np.nan)
for t in range(120, T): ma120[t] = idx[t - 120:t].mean()
REGIME_ON = np.ones(T, bool)
REGIME_ON[1:] = ~(idx[:-1] < ma120[:-1])          # 전날 지수가 전날 MA120 아래면 off (MA 없는 초기는 on)
dipm = D.dipm
DIPL = [np.where(dipm[t])[0][np.argsort(r1[t, dipm[t]])].tolist() for t in range(T)]
K = 5


def simulate(s, e, trig, hold, kc, regime, staged, cap0=CAP, liquidate_end=True):
    cash, amt, buys = cap0, 0.0, 0
    pos = {}; last_rot = None; pend_buy = []; stage = 0; navs = []

    def nav(t): return cash + sum(p[1] * cf[t, j] * (1 - SELLC) for j, p in pos.items())

    def sell(j, t, px):
        nonlocal cash, amt
        if not tr[t, j] or not px > 0: return False
        q = pos.pop(j)[1]; cash += q * px * (1 - SELLC); amt += q * px; return True

    def buy(j, t, px, kind, budget, exit_t=None):
        nonlocal cash, amt, buys
        if j in pos or not tr[t, j] or not px > 0: return False
        q = np.floor(min(budget, cash) / (px * (1 + BUYC)))
        if q < 1: return False
        cash -= q * px * (1 + BUYC); amt += q * px; buys += 1; pos[j] = [kind, q, exit_t]; return True

    def base_rank(t):
        return lambda j: NH[t - 1].index(j) if j in NH[t - 1] else 999

    for t in range(s, e + 1):
        n0 = nav(t - 1) if t > 0 else cap0; slot = n0 / K
        # ── 시가: overlay 청산, base 정리
        for j in [j for j, p in pos.items() if p[0] == 'dip' and p[2] == t]: sell(j, t, o[t, j])
        dip_held = any(p[0] == 'dip' for p in pos.values())
        if not dip_held: stage = 0
        on = REGIME_ON[t] if regime else True
        if not on:
            for j in [j for j, p in pos.items() if p[0] == 'base']: sell(j, t, o[t, j])
        elif not dip_held:
            if last_rot is None or t - last_rot >= R_DAYS:
                top = set(NH[t - 1][:K])
                for j in [j for j, p in pos.items() if p[0] == 'base' and j not in top]: sell(j, t, o[t, j])
                last_rot = t
            need = K - sum(1 for p in pos.values() if p[0] == 'base')
            pend_buy = [(j, 'base', slot) for j in [j for j in NH[t - 1] if j not in pos and tr[t, j]][:max(need, 0)]]
        # ── 종가
        for j, kind, budget in pend_buy: buy(j, t, c[t, j], kind, budget)
        pend_buy = []
        for j in [j for j, p in pos.items() if p[0] == 'base' and surge[t, j] and t < e]: sell(j, t, c[t, j])
        if TRIG[trig][t] and t < e and DIPL[t]:
            dips = [j for j in DIPL[t] if j not in pos][:kc]
            if not dips: pass
            elif not staged and stage == 0:
                for j in sorted([j for j, p in pos.items() if p[0] == 'base' and tr[t, j]], key=base_rank(t), reverse=True): sell(j, t, c[t, j])
                nv = nav(t)
                for j in dips: buy(j, t, c[t, j], 'dip', nv / kc, t + hold + 1)
                stage = 2
            elif staged and stage == 0:
                bl = sorted([j for j, p in pos.items() if p[0] == 'base' and tr[t, j]], key=base_rank(t), reverse=True)
                for j in bl[:(K + 1) // 2]: sell(j, t, c[t, j])
                nv = nav(t)
                for j in dips: buy(j, t, c[t, j], 'dip', 0.5 * nv / kc, t + hold + 1)
                stage = 1
            elif staged and stage == 1:
                for j in [j for j, p in pos.items() if p[0] == 'base' and tr[t, j]]: sell(j, t, c[t, j])
                nv = nav(t)
                for j in dips: buy(j, t, c[t, j], 'dip', nv / kc, t + hold + 1)
                stage = 2
        navs.append(nav(t))
    if liquidate_end:
        for j in list(pos):
            if tr[e, j]: sell(j, e, c[e, j])
    return dict(ret=nav(e) / cap0 - 1, amt=amt, buys=buys, navs=np.array(navs))


def evaluate(cfg, label, P):
    trig, hold, kc, regime, staged = cfg
    d0 = C.didx(F, '20110103')
    full = simulate(d0, T - 1, trig, hold, kc, regime, staged, liquidate_end=False)
    navs = full['navs']; peak = np.maximum.accumulate(navs); mdd = ((navs / peak) - 1).min()
    ann = (navs[-1] / CAP) ** (246 / (T - d0)) - 1
    seg = {pn: navs[min(b + MONTH, T - 1) - d0] / navs[max(a - d0 - 1, 0)] - 1 for pn, (a, b) in PER.items()}
    rows = {}
    for pn, (a, b) in PER.items():
        out = [simulate(s, s + MONTH - 1, trig, hold, kc, regime, staged) for s in range(a, b)]
        rows[pn] = (np.array([x['ret'] for x in out]), np.array([x['amt'] for x in out]), np.array([x['buys'] for x in out]))
    r1_, r2_, r3_ = rows['2011~2019'][0], rows['2020~2025'][0], rows['2026'][0]
    P(f'  {label} | {r1_.mean() * 100:+5.2f} {np.percentile(r1_, 10) * 100:+5.1f} | {r2_.mean() * 100:+5.2f} {np.percentile(r2_, 10) * 100:+5.1f} '
      f'{np.median(r2_) * 100:+5.2f} {(r2_ <= -.1).mean() * 100:3.0f}% {(r2_ >= .1).mean() * 100:3.0f}% | {r3_.mean() * 100:+5.2f} {np.percentile(r3_, 10) * 100:+5.1f} '
      f'| 경로 {seg["2011~2019"] * 100:+6.0f}% {seg["2020~2025"] * 100:+6.1f}% {seg["2026"] * 100:+6.1f}% | 연 {ann * 100:+5.1f}% MDD {mdd * 100:5.1f}% '
      f'| 월 {rows["2020~2025"][1].mean() / 1e8:4.1f}억 {rows["2020~2025"][2].mean():4.1f}건')
    return dict(rows=rows, navs=navs, ann=ann, mdd=mdd, seg=seg)


if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('2차 설계 — base 신고가 k5 (시가 매도·종가 매수) · overlay 트리거일 종가 매수 · 1억 · 비용 포함')
    P('  열: 2011~19 월평균 하위10% | 2020~25 월평균 하위10% 중앙 −10%↓ +10%↑ | 2026 월평균 하위10% | 복리 경로 구간별 | 전기간 연환산 MDD | 2020~25 월 매매금액 매수건수')
    P(f'  {"설정":<34} | 시장 (대상 동일가중, 비용 전) 월평균: 2011~19 +0.08 · 2020~25 +0.89 · 2026 +0.97')
    RES = {}
    for i, cfg in enumerate(itertools.product(('T1', 'T2'), (3, 5, 8), (5, 10), (False, True), (False, True))):
        trig, hold, kc, regime, staged = cfg
        label = f'{trig} h{hold} kc{kc:<2} {"MA120" if regime else "     "} {"분할" if staged else "일괄"}'
        RES[label] = evaluate(cfg, label, P)
        if i % 8 == 7: print(f'  … {i + 1}/48 {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    P('\n■ 비교용: overlay 없음 (T0) · regime 없음/MA120')
    for regime in (False, True):
        label = f'T0 base만 {"MA120" if regime else "     "}'
        RES[label] = evaluate(('T0', 5, 5, regime, False), label, P)
    import pickle; pickle.dump({k: dict(ann=v['ann'], mdd=v['mdd'], seg=v['seg'], navs=v['navs'],
                                        m=({pn: r[0] for pn, r in v['rows'].items()})) for k, v in RES.items()},
                               open('/data/lab/design_sim2.pkl', 'wb'))
    open('backtest/results/lab_design_sim2.md', 'w', encoding='utf-8').write(
        '# 설계 시뮬레이터 2차 (backtest/lab/fresh/design_sim2.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
