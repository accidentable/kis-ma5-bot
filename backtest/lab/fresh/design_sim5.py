"""
설계 시뮬레이터 5차 — 수익실현(익절) 기준: 없음 / 누적 +5% / 누적 +10% (사용자: "한 달 단타 싸움이면 +5% 가 맞지 않나")

고정   base 신고가 k5 · 교체일 시가 매도/종가 매수 · 트리거 T5 (시장 −3% & −3σ) · overlay 트리거일 종가 매수 · 5종목 · 일괄
base 청산 규칙 (4종)
  급등일 +10% (현행)   보유 중 하루 +10%↑ 오른 날 종가 매도
  누적 +5% 익절        매수가 대비 장중 +5% 에 닿으면 그 가격에 매도 (시가가 이미 위면 시가). 급등일 규칙 없음
  누적 +10% 익절       같은 방식, +10%
  없음                 교체일까지 들고 있는다
overlay 청산 규칙 (3종)
  5일 뒤 시가 (현행)   D+6 시가
  +5% 익절             매수가 대비 장중 +5% 닿으면 매도, 못 닿으면 D+6 시가
  +10% 익절            같은 방식, +10%
익절로 비운 자리는 다음 날 종가에 순위대로 다시 채운다 (base). overlay 는 익절 뒤 그 돈을 base 로 돌린다 (다음 날 종가).
평가   2차와 같음. 2020~2025 월 매매금액 · 매수 건수도 같이 (대회 조건 참고).
"""
import numpy as np, warnings, time, itertools, sys; warnings.filterwarnings('ignore')
from backtest.lab.fresh import design_sim3 as S3          # T5 등록
from backtest.lab.fresh import design_sim2 as S
from backtest.lab import panic2_common as C

F, T, o, c, cf, tr, r1 = S.F, S.T, S.o, S.c, S.cf, S.tr, S.r1
h = F.h.astype(float)
BUYC, SELLC, CAP, MONTH, R_DAYS = S.BUYC, S.SELLC, S.CAP, S.MONTH, S.R_DAYS
NH, surge, TRIG, PER, DIPL = S.NH, S.surge, S.TRIG, S.PER, S.DIPL
K, KC, HOLD = 5, 5, 5


def simulate(s, e, base_exit, ov_exit, cap0=CAP, liquidate_end=True):
    cash, amt, buys = cap0, 0.0, 0
    pos = {}; last_rot = None; pend_buy = []; navs = []
    tp_base = {'급등일+10%': None, '누적+5%': 0.05, '누적+10%': 0.10, '없음': None}[base_exit]
    tp_ov = {'5일시가': None, '+5%익절': 0.05, '+10%익절': 0.10}[ov_exit]

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
        cash -= q * px * (1 + BUYC); amt += q * px; buys += 1; pos[j] = [kind, q, exit_t, px]; return True

    def tp_check(t, kind, tp):
        """장중 익절: 시가가 목표 위면 시가, 아니면 고가가 목표에 닿으면 목표가"""
        for j in [j for j, p in pos.items() if p[0] == kind]:
            if not tr[t, j]: continue
            tgt = pos[j][3] * (1 + tp)
            if o[t, j] >= tgt: sell(j, t, o[t, j])
            elif h[t, j] >= tgt: sell(j, t, tgt)

    for t in range(s, e + 1):
        n0 = nav(t - 1) if t > 0 else cap0; slot = n0 / K
        # ── 시가 · 장중
        for j in [j for j, p in pos.items() if p[0] == 'dip' and p[2] == t]: sell(j, t, o[t, j])
        if tp_ov is not None: tp_check(t, 'dip', tp_ov)
        if tp_base is not None: tp_check(t, 'base', tp_base)
        dip_held = any(p[0] == 'dip' for p in pos.values())
        if not dip_held:
            if last_rot is None or t - last_rot >= R_DAYS:
                top = set(NH[t - 1][:K])
                for j in [j for j, p in pos.items() if p[0] == 'base' and j not in top]: sell(j, t, o[t, j])
                last_rot = t
            need = K - sum(1 for p in pos.values() if p[0] == 'base')
            pend_buy = [(j, 'base', slot) for j in [j for j in NH[t - 1] if j not in pos and tr[t, j]][:max(need, 0)]]
        # ── 종가
        for j, kind, budget in pend_buy: buy(j, t, c[t, j], kind, budget)
        pend_buy = []
        if base_exit == '급등일+10%':
            for j in [j for j, p in pos.items() if p[0] == 'base' and surge[t, j] and t < e]: sell(j, t, c[t, j])
        if TRIG['T5'][t] and t < e and DIPL[t] and not any(p[0] == 'dip' for p in pos.values()):
            dips = [j for j in DIPL[t] if j not in pos][:KC]
            if dips:
                for j in [j for j, p in pos.items() if p[0] == 'base' and tr[t, j]]: sell(j, t, c[t, j])
                nv = nav(t)
                for j in dips: buy(j, t, c[t, j], 'dip', nv / KC, t + HOLD + 1)
        navs.append(nav(t))
    if liquidate_end:
        for j in list(pos):
            if tr[e, j]: sell(j, e, c[e, j])
    return dict(ret=nav(e) / cap0 - 1, amt=amt, buys=buys, navs=np.array(navs))


def evaluate(cfg, label, P):
    d0 = C.didx(F, '20110103')
    full = simulate(d0, T - 1, *cfg, liquidate_end=False)
    navs = full['navs']; peak = np.maximum.accumulate(navs); mdd = ((navs / peak) - 1).min()
    ann = (navs[-1] / CAP) ** (246 / (T - d0)) - 1
    seg = {pn: navs[min(b + MONTH, T - 1) - d0] / navs[max(a - d0 - 1, 0)] - 1 for pn, (a, b) in PER.items()}
    rows = {}
    for pn, (a, b) in PER.items():
        out = [simulate(s, s + MONTH - 1, *cfg) for s in range(a, b)]
        rows[pn] = (np.array([x['ret'] for x in out]), np.array([x['amt'] for x in out]), np.array([x['buys'] for x in out]))
    r1_, r2_, r3_ = rows['2011~2019'][0], rows['2020~2025'][0], rows['2026'][0]
    P(f'  {label} | {r1_.mean() * 100:+5.2f} {np.percentile(r1_, 10) * 100:+5.1f} | {r2_.mean() * 100:+5.2f} {np.percentile(r2_, 10) * 100:+5.1f} '
      f'{np.median(r2_) * 100:+5.2f} {(r2_ <= -.1).mean() * 100:3.0f}% {(r2_ >= .1).mean() * 100:3.0f}% | {r3_.mean() * 100:+5.2f} {np.percentile(r3_, 10) * 100:+5.1f} '
      f'| 경로 {seg["2011~2019"] * 100:+6.0f}% {seg["2020~2025"] * 100:+6.1f}% {seg["2026"] * 100:+6.1f}% | 연 {ann * 100:+5.1f}% MDD {mdd * 100:5.1f}% '
      f'| 월 {rows["2020~2025"][1].mean() / 1e8:4.1f}억 {rows["2020~2025"][2].mean():4.1f}건')
    return rows


if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('5차 — 익절 기준 비교 (base 신고가 k5 · T5 overlay · 종가 매수). 열: 2011~19 월평균 하위10% | 2020~25 월평균 하위10% 중앙 −10%↓ +10%↑ | 2026 | 경로 | 연 MDD | 월 매매금액 매수건수')
    for be, oe in itertools.product(('급등일+10%', '누적+5%', '누적+10%', '없음'), ('5일시가', '+5%익절', '+10%익절')):
        evaluate((be, oe), f'base {be:<8} overlay {oe:<7}', P)
    open('backtest/results/lab_design_sim5.md', 'w', encoding='utf-8').write(
        '# 설계 시뮬레이터 5차 — 익절 기준 (backtest/lab/fresh/design_sim5.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
