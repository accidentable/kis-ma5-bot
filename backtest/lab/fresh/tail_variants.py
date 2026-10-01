"""
집중 모멘텀 변형 · 연도별 분해 · 성공한 달의 예 (대회용 리서치 Phase D)

바탕: D 모멘텀 집중 + 목표 락 30% + 손절 7% (Phase C 에서 남은 틀). 여기서 바꾸는 것만 (미리 정함):
  순위 기간   5일 / 10일 / 20일 / 60일 수익률 상위 10% 중 높은 순
  보유       5일 / 10일 / 20일(월말까지, 손절만)
  종목 수    1 / 2
  폭락 전환  없음 / 있음 (T5 날 급락주로 5일)
출력  ① 변형 격자 (+20 · +30 · −20 · −30 확률, 중앙, 평균)
      ② 대표 설정의 연도별 +30% 확률 · −20% 확률 · 평균 (달력 달 기준: 매달 첫 거래일 시작 20일 창)
      ③ 달력 달 기준 +30% 넘긴 달 목록 (최근 30개) 과 그때 산 종목
"""
import numpy as np, warnings, time, sys, itertools; warnings.filterwarnings('ignore')
from backtest.lab.fresh import tail_strategies as TS
from backtest.lab.fresh import tail_policy as TP
from backtest.lab import panic2_common as C

F, T, cf, E, limup = TS.F, TS.T, TS.cf, TS.E, TS.limup
PER = TS.PER


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


RET = {5: F.ret5.astype(float), 10: cf / lag(cf, 10) - 1, 20: F.ret20.astype(float), 60: F.ret60.astype(float)}
for n, r in RET.items():
    TS.CAND[f'D{n}'] = TS.lists(TS.pct_mask(r, 0.9, 1.0) & ~limup, r)


def run_month(s, cand, k, hold, lock=0.30, stop=0.07, mix=False):
    """tail_policy.run_month 와 같되 보유 기간을 바꿀 수 있게 (hold=20 이면 월말까지)"""
    o, c, tr, T5, CAND = TS.o, TS.c, TS.tr, TS.T5, TS.CAND
    BUYC, SELLC, CAP, W = TS.BUYC, TS.SELLC, TS.CAP, TS.W
    e = s + W - 1; cash, pos, sell_open, locked, ov = CAP, {}, set(), False, False; names = []

    def nav(t): return cash + sum(p[0] * cf[t, j] * (1 - SELLC) for j, p in pos.items())

    def sell(j, t, px):
        nonlocal cash
        if not tr[t, j] or not px > 0: return False
        cash += pos.pop(j)[0] * px * (1 - SELLC); return True

    def buy(j, t, px, kind, budget, hold_):
        nonlocal cash
        if j in pos or not tr[t, j] or not px > 0: return False
        q = np.floor(min(budget, cash) / (px * (1 + BUYC)))
        if q < 1: return False
        cash -= q * px * (1 + BUYC); pos[j] = [q, px, px, t + hold_, kind]; names.append((str(F.dates[t]), str(F.names[j]))); return True

    for t in range(s, e + 1):
        for j in list(sell_open): sell(j, t, o[t, j])
        sell_open = set()
        if locked:
            for j in list(pos): sell(j, t, o[t, j])
            continue
        for j in [j for j, p in pos.items() if p[3] == t]: sell(j, t, o[t, j])
        crash_now = mix and T5[t] and t < e
        if crash_now and not ov:
            for j in list(pos): sell(j, t, c[t, j])
            nv = nav(t)
            for j in CAND['A'][t][:k]: buy(j, t, c[t, j], 'A', nv / k, t + 6)
            ov = True
        else:
            if ov and not any(p[4] == 'A' for p in pos.values()): ov = False
            for j, p in pos.items():
                if stop and cf[t, j] <= p[1] * (1 - stop): sell_open.add(j)
            if not ov and t < e:
                nv = nav(t)
                for j in [j for j in CAND[cand][t] if j not in pos][:max(k - len(pos), 0)]: buy(j, t, c[t, j], cand, nv / k, min(t + hold + 1, e + 1))
        if lock and nav(t) >= CAP * (1 + lock): locked = True
    for j in list(pos): sell(j, e, c[e, j])
    return cash / CAP - 1, names


def fmt(r): return f'{(r >= .2).mean() * 100:4.1f} {(r >= .3).mean() * 100:4.1f} {(r <= -.2).mean() * 100:4.1f} {(r <= -.3).mean() * 100:4.1f} {np.median(r) * 100:+5.1f} {r.mean() * 100:+5.1f}'


if __name__ == '__main__':
    t0 = time.time(); lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('① 변형 격자 — 락30 · 손절7. 각 기간 열: +20%↑ · +30%↑ · −20%↓ · −30%↓ 확률(%) · 중앙 · 평균')
    P(f'  {"설정":<30} | {"2011~2019":^34} | {"2020~2025":^34} | {"2026":^34}')
    RES = {}
    for n, hold, k, mix in itertools.product((5, 10, 20, 60), (5, 10, 20), (1, 2), (False, True)):
        label = f'순위{n:>2}일 보유{hold:>2} k{k}{" +폭락" if mix else "     "}'
        r = {pn: np.array([run_month(s, f'D{n}', k, hold, mix=mix)[0] for s in range(a, b)]) for pn, (a, b) in PER.items()}
        RES[label] = r; P(f'  {label:<30} | ' + ' | '.join(fmt(r[pn]) for pn in PER))
    P('\n  2020~2025 +30% 확률 상위 10:')
    for label, r in sorted(RES.items(), key=lambda kv: -(kv[1]['2020~2025'] >= .3).mean())[:10]:
        P(f'    {label:<30} 2020~25 +30% {(r["2020~2025"] >= .3).mean() * 100:4.1f}% −30% {(r["2020~2025"] <= -.3).mean() * 100:4.1f}% | 2011~19 +30% {(r["2011~2019"] >= .3).mean() * 100:4.1f}% | 2026 +30% {(r["2026"] >= .3).mean() * 100:4.1f}%')
    # ② ③ 대표 설정: 순위 20일 · 보유 10 · k2 · +폭락 (달력 달)
    P('\n② 대표 설정 (순위 20일 · 보유 10 · k2 · 락30 · 손절7 · +폭락) — 달력 달 기준 (매달 첫 거래일에 시작, 20거래일)')
    months = {}
    for t in range(C.didx(F, '20110103'), T - 21):
        ym = str(F.dates[t])[:6]
        if ym not in months: months[ym] = t
    rows = {ym: run_month(s, 'D20', 2, 10, mix=True) for ym, s in months.items()}
    P('  연도  달수  +20%↑  +30%↑  −20%↓  평균   최고달')
    for y in range(2011, 2027):
        ys = {ym: v for ym, v in rows.items() if ym.startswith(str(y))}
        if not ys: continue
        r = np.array([v[0] for v in ys.values()]); best = max(ys.items(), key=lambda kv: kv[1][0])
        P(f'  {y}  {len(r):3d}  {(r >= .2).sum():3d}    {(r >= .3).sum():3d}    {(r <= -.2).sum():3d}   {r.mean() * 100:+5.1f}%  {best[0]} {best[1][0] * 100:+.0f}%')
    P('\n③ +30% 넘긴 달 (최근 30개) — 그 달에 산 종목')
    hits = [(ym, v) for ym, v in rows.items() if v[0] >= .3]
    P(f'  전체 {len(rows)}달 중 +30% {len(hits)}달 ({len(hits) / len(rows) * 100:.0f}%), +20% {sum(v[0] >= .2 for v in rows.values())}달, −20% {sum(v[0] <= -.2 for v in rows.values())}달')
    for ym, (r, names) in hits[-30:]:
        P(f'  {ym} {r * 100:+5.0f}%  ' + ', '.join(f'{d[4:]} {nm}' for d, nm in names[:6]) + (' …' if len(names) > 6 else ''))
    open('backtest/results/lab_tail_variants.md', 'w', encoding='utf-8').write(
        '# 집중 모멘텀 변형 · 연도별 · 성공한 달 (backtest/lab/fresh/tail_variants.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
