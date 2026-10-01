"""
설계 시뮬레이터 7차 — 평소 날 '깨끗한 급락' 을 1~2 슬롯 얹으면? (공시 분석에서 시장 멀쩡한 날의 깨끗한 급락이 5일 +3% 로 나와서)

깨끗한 급락 (normal-day dip) 후보: 폭락일(T5)이 아닌 날, 그날 −10~−20% (하한가 제외), 대용지표 깨끗(거래대금 < 20일 평균 × 3, 잔차 z > −3),
  전날~당일 DART 공시 중 실적 · 악재 · 시황변동 조회 · 투자판단 공시가 없음 (D0 공시 없음 또는 D1 기타·호재만). 더 빠진 순.
동작   그날 종가에 신고가 보유 중 순위 가장 낮은 것을 팔고 그 돈으로 산다 (슬롯 교체). 동시에 최대 S 개 (S = 0 / 1 / 2). 5거래일 뒤 시가 매도 → 그날 종가에 base 로 복귀.
       T5 폭락 overlay 는 그대로 (후보 −7%↓ · 대용지표 + DART 악재 필터). 폭락 overlay 보유 중엔 평소 급락 안 산다.
비교   S = 0 (6차 F2 와 같음) / 1 / 2.  2차와 같은 지표 + 2020~25 월 매매금액 · 매수 건수 · 거래일 수
"""
import json, numpy as np, warnings, time, sys; warnings.filterwarnings('ignore')
from backtest.lab.fresh import design_sim6 as S6
from backtest.lab.fresh import design_sim2 as S
from backtest.lab.fresh import dart_analyze as DA
from backtest.lab import panic2_common as C

F, T, N, o, c, cf, tr, r1, U = S.F, S.T, S6.N, S.o, S.c, S.cf, S.tr, S.r1, S.D.U
BUYC, SELLC, CAP, MONTH, R_DAYS = S.BUYC, S.SELLC, S.CAP, S.MONTH, S.R_DAYS
NH, surge, PER = S.NH, S.surge, S.PER
T5 = S.TRIG['T5']; K, KC, HOLD = 5, 5, 5
didx = {d: i for i, d in enumerate(F.dates)}; cidx = {cd: j for j, cd in enumerate(F.codes)}
unclean = np.zeros((T, N), bool)                      # 실적 · 악재 · 조회 · 투자판단 공시
for line in open('/data/lab/dart/events_dart.jsonl', encoding='utf-8'):
    try: r = json.loads(line)
    except Exception: continue
    t, j = didx.get(r['date']), cidx.get(r['code'])
    if t is None or j is None or r['items'] is None: continue
    cats = DA.classify(r['items'])
    if cats & {'악재', '실적', '조회', '투자판단'}: unclean[t, j] = True
clean_m = U & np.isfinite(r1) & (r1 <= -0.10) & (r1 > -0.20) & ~S6.limdown & ~S6.proxy & ~unclean & ~T5[:, None]
NDIP = [np.where(clean_m[t])[0][np.argsort(r1[t, clean_m[t]])].tolist() for t in range(T)]
f2 = S6.FILTERS['F2 +DART 악재공시']
DIPL = [np.where(f2[t])[0][np.argsort(r1[t, f2[t]])].tolist() for t in range(T)]


def simulate(s, e, slots, cap0=CAP, liquidate_end=True):
    cash, amt, buys = cap0, 0.0, 0
    pos = {}; last_rot = None; pend_buy = []; navs = []; days = set()

    def nav(t): return cash + sum(p[1] * cf[t, j] * (1 - SELLC) for j, p in pos.items())

    def sell(j, t, px):
        nonlocal cash, amt
        if not tr[t, j] or not px > 0: return False
        q = pos.pop(j)[1]; cash += q * px * (1 - SELLC); amt += q * px; days.add(t); return True

    def buy(j, t, px, kind, budget, exit_t=None):
        nonlocal cash, amt, buys
        if j in pos or not tr[t, j] or not px > 0: return False
        q = np.floor(min(budget, cash) / (px * (1 + BUYC)))
        if q < 1: return False
        cash -= q * px * (1 + BUYC); amt += q * px; buys += 1; days.add(t); pos[j] = [kind, q, exit_t]; return True

    def base_rank(t): return lambda j: NH[t - 1].index(j) if j in NH[t - 1] else 999

    for t in range(s, e + 1):
        n0 = nav(t - 1) if t > 0 else cap0; slot = n0 / K
        for j in [j for j, p in pos.items() if p[0] in ('dip', 'ndip') and p[2] == t]: sell(j, t, o[t, j])
        crash_held = any(p[0] == 'dip' for p in pos.values())
        if not crash_held:
            if last_rot is None or t - last_rot >= R_DAYS:
                top = set(NH[t - 1][:K])
                for j in [j for j, p in pos.items() if p[0] == 'base' and j not in top]: sell(j, t, o[t, j])
                last_rot = t
            need = K - sum(1 for p in pos.values() if p[0] in ('base', 'ndip'))
            pend_buy = [(j, 'base', slot) for j in [j for j in NH[t - 1] if j not in pos and tr[t, j]][:max(need, 0)]]
        for j, kind, budget in pend_buy: buy(j, t, c[t, j], kind, budget)
        pend_buy = []
        for j in [j for j, p in pos.items() if p[0] == 'base' and surge[t, j] and t < e]: sell(j, t, c[t, j])
        if T5[t] and t < e and DIPL[t] and not crash_held:
            dips = [j for j in DIPL[t] if j not in pos][:KC]
            if dips:
                for j in [j for j, p in pos.items() if p[0] in ('base', 'ndip') and tr[t, j]]: sell(j, t, c[t, j])
                nv = nav(t)
                for j in dips: buy(j, t, c[t, j], 'dip', nv / KC, t + HOLD + 1)
        elif slots and t < e and not crash_held and NDIP[t]:
            n_nd = sum(1 for p in pos.values() if p[0] == 'ndip')
            for j in [j for j in NDIP[t] if j not in pos]:
                if n_nd >= slots: break
                bl = sorted([x for x, p in pos.items() if p[0] == 'base' and tr[t, x]], key=base_rank(t), reverse=True)
                if not bl: break
                sell(bl[0], t, c[t, bl[0]])
                if buy(j, t, c[t, j], 'ndip', slot, t + HOLD + 1): n_nd += 1
        navs.append(nav(t))
    if liquidate_end:
        for j in list(pos):
            if tr[e, j]: sell(j, e, c[e, j])
    return dict(ret=nav(e) / cap0 - 1, amt=amt, buys=buys, days=len(days), navs=np.array(navs))


if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('7차 — 평소 날 깨끗한 급락 슬롯 (S) 추가. 열: 2011~19 월평균 하위10% | 2020~25 월평균 하위10% 중앙 −10%↓ +10%↑ | 2026 | 경로 | 연 MDD | 2020~25 월 매매금액 매수건수 거래일')
    P(f'  깨끗한 급락 후보: 하루 평균 {clean_m[C.didx(F, "20110103"):].sum(1).mean():.2f}개, 있는 날 비율 {(clean_m[C.didx(F, "20110103"):].sum(1) > 0).mean() * 100:.0f}%')
    d0 = C.didx(F, '20110103')
    for slots in (0, 1, 2):
        full = simulate(d0, T - 1, slots, liquidate_end=False); navs = full['navs']
        peak = np.maximum.accumulate(navs); mdd = ((navs / peak) - 1).min(); ann = (navs[-1] / CAP) ** (246 / (T - d0)) - 1
        seg = {pn: navs[min(b + MONTH, T - 1) - d0] / navs[max(a - d0 - 1, 0)] - 1 for pn, (a, b) in PER.items()}
        rows = {}
        for pn, (a, b) in PER.items():
            out = [simulate(s, s + MONTH - 1, slots) for s in range(a, b)]
            rows[pn] = (np.array([x['ret'] for x in out]), np.array([x['amt'] for x in out]), np.array([x['buys'] for x in out]), np.array([x['days'] for x in out]))
        r1_, r2_, r3_ = rows['2011~2019'][0], rows['2020~2025'][0], rows['2026'][0]
        P(f'  S={slots} | {r1_.mean() * 100:+5.2f} {np.percentile(r1_, 10) * 100:+5.1f} | {r2_.mean() * 100:+5.2f} {np.percentile(r2_, 10) * 100:+5.1f} {np.median(r2_) * 100:+5.2f} '
          f'{(r2_ <= -.1).mean() * 100:3.0f}% {(r2_ >= .1).mean() * 100:3.0f}% | {r3_.mean() * 100:+5.2f} {np.percentile(r3_, 10) * 100:+5.1f} '
          f'| 경로 {seg["2011~2019"] * 100:+6.0f}% {seg["2020~2025"] * 100:+6.1f}% {seg["2026"] * 100:+6.1f}% | 연 {ann * 100:+5.1f}% MDD {mdd * 100:5.1f}% '
          f'| 월 {rows["2020~2025"][1].mean() / 1e8:4.1f}억 {rows["2020~2025"][2].mean():4.1f}건 거래일 {rows["2020~2025"][3].mean():4.1f}일')
    open('backtest/results/lab_design_sim7.md', 'w', encoding='utf-8').write(
        '# 설계 시뮬레이터 7차 — 평소 날 깨끗한 급락 슬롯 (backtest/lab/fresh/design_sim7.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
