"""
설계 시뮬레이터 — 평소 전략 × 폭락 overlay × 자금 배분 × 체결 시각을 한 번에 비교 (1억, 2011~2026)

평소(base) 전략 (k 종목, 21거래일마다 교체, 상위 k 안에 남은 종목은 유지, 보유 중 하루 +10%↑ 면 그날 종가 매도 후 다음 기회에 채움)
  B0 현금        아무것도 안 든다 (overlay 만)
  B2 신고가 k5   봇 기본: 시총 200 · 60일 수익률 > 0 · 20일 내 급등일 없음 · 종가/250일 최고가 높은 순 (k = 5)
  B3 신고가 k10  같은 규칙, k = 10
  B4 조용한 대형주 k10  점수 = (회전율 낮음, 60일 변동성 낮음, 시총 큼, 250일 고가 근접) 각 대상 안 백분위의 평균, 급등일(20일 내 +10%) 제외
  B5 조용한 대형주 k20
체결 시각 (exec)
  open   교체일 시가에 팔고 시가에 산다 (지금 봇)
  close  교체일 시가에 팔고 그날 종가에 산다 (낮 동안 비움 — 밤/낮 분해에서 낮 수익이 평균 마이너스)
폭락 overlay (trigger)
  T0 없음 / T1 대상 평균 등락 −3%↓ / T2 −2.5σ↓ (σ = 직전 20일 시장 일간 변동성) / T3 −4%↓
  동작: 트리거일 종가(overlay_entry=close) 또는 다음 날 시가(open) 에 그날 −10~−20% (하한가 제외) 종목을 더 빠진 순으로 5개 매수,
        h 거래일 뒤 시가에 매도 (기본 h=5 → D+6 시가). overlay 보유 중엔 새 트리거 무시.
자금 배분 (reserve)
  0.0   평소에 전부 투자. overlay 는 base 중 순위 낮은 것부터 시가에 팔아 자금을 만든다
  0.3   평소 70% 만 투자, 30% 현금 예비. overlay 는 예비 현금으로만 산다 (base 는 안 판다)
비용   매수 0.065%, 매도 0.245%. 대상 = 코스피 시총 100 + 코스닥 시총 150 (base 는 시총 200, 급등·정지·하한가 이력 제외)
평가   ① 20거래일 월 (시작일 하루씩, 마지막 날 전량 평가): 평균 · 중앙 · 손실달 · −10%↓ · +10%↑ · 월 매매금액 · 매수 건수
       ② 전 기간 복리 한 경로: 연환산 수익 · 최대 낙폭(MDD) · 구간별 수익
       기간 2011~2019 / 2020~2025 / 2026
선택 기준 (미리): 2020~2025 월평균과 하위 10% 를 주로 보되, 2011~2019 에서 시장(대상 동일가중) 이하면 탈락. 기간별로 설정 안 바꿈.
"""
import numpy as np, warnings, time, itertools, sys; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C

t0 = time.time()
F = PF.load(); T, N = F.T, F.N
o, c = F.o.astype(float), F.c.astype(float); tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
BUYC, SELLC = 0.00065, 0.00265
CAP, MONTH, R_DAYS = 1e8, 20, 21
KC, HOLD = 5, 5


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


cf = ffill(np.where(tr, c, np.nan)); pc = lag(cf, 1); r1 = c / pc - 1
limdown = F.limdown.astype(bool); limup = F.limup.astype(bool); bad20 = F.bad20.astype(bool)
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
sig20 = np.full(T, np.nan)
for t in range(21, T): sig20[t] = np.nanstd(mkt[t - 20:t])
TRIG = {'T0': np.zeros(T, bool), 'T1': mkt <= -0.03, 'T2': mkt <= -2.5 * sig20, 'T3': mkt <= -0.04}
cap = F.caprank.astype(float)
EB = F.tradable & (cap <= 200) & ~bad20 & ~limdown & ~limup
NH_OK = EB & (F.ret60 > 0) & (F.max20 < 0.10) & np.isfinite(F.hi250)


def top_list(score, ok, k=30):
    out = []
    for t in range(T):
        s = np.where(ok[t], score[t], np.nan); idx = np.where(np.isfinite(s))[0]
        out.append(idx[np.argsort(-s[idx])[:k]].tolist())
    return out


def pct_rank(x, mask):
    out = np.full((T, N), np.nan)
    for t in range(T):
        idx = np.where(mask[t] & np.isfinite(x[t]))[0]
        if len(idx) < 20: continue
        r = np.empty(len(idx)); r[np.argsort(x[t, idx])] = np.arange(len(idx)); out[t, idx] = r / (len(idx) - 1)
    return out


turn1, vol60, capv, hi250 = (getattr(F, k).astype(float) for k in ('turn1', 'vol60', 'cap', 'hi250'))
NH = top_list(c / hi250, NH_OK)
Q = top_list((pct_rank(-turn1, EB) + pct_rank(-lag(vol60, 1), EB) + pct_rank(capv, EB) + pct_rank(c / hi250, EB)) / 4,
             EB & (F.max20 < 0.10))
surge = tr & (r1 >= 0.10)
dipm = U & np.isfinite(r1) & (r1 <= -0.10) & (r1 > -0.20) & ~limdown
DIP = [np.where(dipm[t])[0][np.argsort(r1[t, dipm[t]])][:KC].tolist() for t in range(T)]
BASES = {'B0 현금': (None, 0), 'B2 신고가 k5': (NH, 5), 'B3 신고가 k10': (NH, 10), 'B4 조용한대형 k10': (Q, 10), 'B5 조용한대형 k20': (Q, 20)}
print(f'준비 {time.time() - t0:.0f}s', flush=True)


def simulate(s, e, base, k, exec_, trig, ov_entry, reserve, cap0=CAP, liquidate_end=True):
    """s..e 날짜 구간. 반환 dict(ret, amt, buys, nav path)"""
    cash, amt, buys = cap0, 0.0, 0
    pos = {}                                   # j → [kind, qty, exit_t]
    last_rot = None; pend_buy = []             # 종가에 살 것 (exec=close): [(j, kind, budget)]
    pend_dip = None                            # (t_entry_open, list) overlay 다음 날 시가 매수
    navs = []

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

    for t in range(s, e + 1):
        n0 = nav(t - 1) if t > 0 else cap0
        inv = n0 * (1 - reserve); slot = inv / k if k else 0
        dip_held = any(p[0] == 'dip' for p in pos.values())
        # ── 시가
        for j in [j for j, p in pos.items() if p[0] == 'dip' and p[2] == t]: sell(j, t, o[t, j])
        if pend_dip and pend_dip[0] == t:
            budget = (n0 * reserve if reserve > 0 else n0) / KC
            if reserve == 0:                           # base 를 팔아 자금 마련 (순위 낮은 것부터)
                need = len(pend_dip[1])
                order = sorted([j for j, p in pos.items() if p[0] == 'base' and tr[t, j]],
                               key=lambda j: base[t - 1].index(j) if j in base[t - 1] else 999, reverse=True)
                for j in order[:need]: sell(j, t, o[t, j])
            for j in pend_dip[1]: buy(j, t, o[t, j], 'dip', budget, t + HOLD + 1)
            pend_dip = None
        if base is not None and not (dip_held and reserve == 0):
            rot = last_rot is None or t - last_rot >= R_DAYS
            if rot:
                top = set(base[t - 1][:k])
                for j in [j for j, p in pos.items() if p[0] == 'base' and j not in top]: sell(j, t, o[t, j])
                last_rot = t
            nb = sum(1 for p in pos.values() if p[0] == 'base'); need = k - nb - sum(1 for p in pos.values() if p[0] == 'dip')
            cands = [j for j in base[t - 1] if j not in pos and tr[t, j]][:max(need, 0)]
            if exec_ == 'open':
                for j in cands: buy(j, t, o[t, j], 'base', slot)
            else:
                pend_buy = [(j, 'base', slot) for j in cands]
        # ── 종가
        for j, kind, budget in pend_buy: buy(j, t, c[t, j], kind, budget)
        pend_buy = []
        for j in [j for j, p in pos.items() if p[0] == 'base' and surge[t, j] and t < e]: sell(j, t, c[t, j])
        if TRIG[trig][t] and not dip_held and not any(p[0] == 'dip' for p in pos.values()) and t < e and DIP[t]:
            if ov_entry == 'close':
                budget = (nav(t) * reserve if reserve > 0 else nav(t)) / KC
                if reserve == 0:
                    order = sorted([j for j, p in pos.items() if p[0] == 'base' and tr[t, j]],
                                   key=lambda j: base[t - 1].index(j) if (base and j in base[t - 1]) else 999, reverse=True)
                    for j in order[:len(DIP[t])]: sell(j, t, c[t, j])
                for j in DIP[t]: buy(j, t, c[t, j], 'dip', budget, t + HOLD + 1)
            else:
                pend_dip = (t + 1, DIP[t])
        navs.append(nav(t))
    if liquidate_end:
        for j in list(pos):
            if tr[e, j]: sell(j, e, c[e, j])
    return dict(ret=nav(e) / cap0 - 1, amt=amt, buys=buys, navs=np.array(navs))


PER = {'2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')), '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101') - MONTH),
       '2026': (C.didx(F, '20260101'), T - MONTH)}


def main():
    CONFIGS = []
    for bn, exec_ in itertools.product(BASES, ('open', 'close')):
        if bn == 'B0 현금' and exec_ == 'close': continue
        for trig in TRIG:
            for reserve in (0.0, 0.3):
                if trig == 'T0' and reserve > 0: continue
                if bn == 'B0 현금' and reserve > 0: continue
                CONFIGS.append((bn, exec_, trig, 'close', reserve))
    # overlay 진입 시각 비교는 대표 설정 둘에서만
    CONFIGS += [('B2 신고가 k5', 'open', 'T2', 'open', 0.0), ('B4 조용한대형 k10', 'close', 'T2', 'open', 0.0)]
    lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('설계 시뮬레이터 — 1억 · 월 = 20거래일 · 비용 포함. 열: 평균 / 중앙 / 손실달 / −10%↓ / +10%↑ | 월 매매금액 · 매수건수 | 전기간 연환산 · MDD')


    def mstats(r): return f'{r.mean() * 100:+5.2f} {np.median(r) * 100:+5.2f} {(r < 0).mean() * 100:3.0f}% {(r <= -.1).mean() * 100:3.0f}% {(r >= .1).mean() * 100:3.0f}%'


    # 시장 기준: 대상 동일가중 (비용 없음) — 월 수익
    mret = np.nan_to_num(np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1))
    cum = np.cumprod(1 + mret)
    for pn, (a, b) in PER.items():
        r = np.array([cum[s + MONTH - 1] / cum[s - 1] - 1 for s in range(a, b)])
        P(f'  [{pn}] 시장 (대상 동일가중, 비용 전)     {mstats(r)}')
    RES = {}
    for ci, (bn, exec_, trig, ove, reserve) in enumerate(CONFIGS):
        base, k = BASES[bn]
        name = f'{bn:<16} {exec_:<5} {trig} ov@{ove:<5} 예비{reserve:.0%}'
        full = simulate(C.didx(F, '20110103'), T - 1, base, k, exec_, trig, ove, reserve, liquidate_end=False)
        navs = full['navs']; days0 = C.didx(F, '20110103')
        peak = np.maximum.accumulate(navs); mdd = ((navs / peak) - 1).min()
        yrs = (T - days0) / 246; ann = (navs[-1] / CAP) ** (1 / yrs) - 1
        seg = {}
        for pn, (a, b) in PER.items():
            i0, i1 = a - days0, min(b + MONTH, T - 1) - days0
            seg[pn] = navs[i1] / navs[max(i0 - 1, 0)] - 1
        rows = {}
        for pn, (a, b) in PER.items():
            out = [simulate(s, s + MONTH - 1, base, k, exec_, trig, ove, reserve) for s in range(a, b)]
            rows[pn] = (np.array([x['ret'] for x in out]), np.array([x['amt'] for x in out]), np.array([x['buys'] for x in out]))
        RES[name] = (rows, ann, mdd, seg)
        P(f'\n{name} | 전기간 연 {ann * 100:+5.1f}% MDD {mdd * 100:5.1f}% | 구간 ' + ' '.join(f'{pn} {seg[pn] * 100:+6.1f}%' for pn in PER))
        for pn in PER:
            r, amt, nb = rows[pn]
            P(f'    {pn:<9} {mstats(r)} | {amt.mean() / 1e8:4.1f}억 {nb.mean():4.1f}건')
        if ci % 10 == 9: print(f'  … {ci + 1}/{len(CONFIGS)} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)

    P('\n■ 요약표 (2011~19 월평균 / 2020~25 월평균 · 하위10% · 중앙 / 2026 월평균 | 전기간 연환산 · MDD)')
    for name, (rows, ann, mdd, seg) in RES.items():
        r1_, r2_, r3_ = rows['2011~2019'][0], rows['2020~2025'][0], rows['2026'][0]
        P(f'  {name} | {r1_.mean() * 100:+5.2f} | {r2_.mean() * 100:+5.2f} {np.percentile(r2_, 10) * 100:+5.1f} {np.median(r2_) * 100:+5.2f} | {r3_.mean() * 100:+5.2f} | {ann * 100:+5.1f}% {mdd * 100:5.1f}%')
    open('backtest/results/lab_design_sim.md', 'w', encoding='utf-8').write(
        '# 설계 시뮬레이터 — base × overlay × 배분 × 체결 시각 (backtest/lab/fresh/design_sim.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)


if __name__ == '__main__':
    main()
