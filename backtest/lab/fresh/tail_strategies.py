"""
한 달 +20 / +30% 를 찍을 확률로 전략을 고른다 — 꼬리 포착 전략 비교 (대회용 리서치 Phase B)

평가   20거래일 창(시작일 하루씩 2011~2026), 1억, 비용 매수 0.065% 매도 0.245%. 창 끝엔 전량 종가 평가.
       지표: 월 +20%↑ 확률 / +30%↑ 확률 / 창 안 어느 종가에서든 +30%↑ 찍은 확률(목표 락이 가능한 확률) / −20%↓ 확률 / 중앙 / 평균 / 월 매수 건수
대상   코스피 시총 200 + 코스닥 시총 150, 주가 1,000원↑, 20일 내 정지·하한가 없음
공통   진입은 신호일 종가 (상한가는 못 사니 다음 날 시가). 손절은 종가 기준 판정 → 다음 날 시가 매도. 추적 손절은 최고 종가 대비.
       슬롯이 비면 그날 후보로 다시 채운다 (후보 없으면 현금). 목표 락: 어느 날 종가에 NAV ≥ +30% 면 다음 날 시가에 전량 매도, 월말까지 현금.
전략 (결과 보기 전에 정함)
  A 폭락 올인      T5 폭락일(시장 −3% & −3σ) 에만 −7%↓ 급락주(가격지표 필터) k개 종가 매수, 5일 뒤 시가 매도. 평소 현금
  B 상한가 추격    상한가 마감 종목을 다음 날 시가 매수, 추적 −10%, 최대 10일
  C 급등 추격      +10~20% 마감(상한가 아님) 종가 매수, 손절 −7% · 추적 −10%, 최대 10일
  D 모멘텀 집중    20일 수익률 상위 10% 중 높은 순 k개, 손절 −7%, 최대 10일 뒤 교체
  E 고변동 모멘텀   60일 변동성 상위 10% ∩ 20일 수익률 상위 절반, 수익률 높은 순, 손절 −7% · 추적 −12%, 최대 10일
  F 돌파 추격      20일 신고가 돌파 & 거래 2배↑ (거래 배수 높은 순), 추적 −8%, 최대 15일
  G 신고가+손절     봇 base(신고가 상위 k) 를 첫날 종가 매수, 손절 −7% 후 재진입 없음(이익 종목에 집중이 남음), 월말까지
  H 평소 급락      폭락일 아닌 날 −10%↓ 깨끗한 급락(가격지표·DART 깨끗) k개, 5일 뒤 시가 매도
  I 혼합           평소 D(모멘텀 집중), T5 폭락일엔 A 로 전환(전량 교체), overlay 끝나면 D 복귀
집중도 k = 1 · 2 · 3 · 5 (대회 '지수 종목 5개 이상 거래' 는 한 달 누적 거래 종목 수라 k 가 작아도 채워진다)
"""
import json, numpy as np, warnings, time, sys, itertools; warnings.filterwarnings('ignore')
from numpy.lib.stride_tricks import sliding_window_view as swv
from backtest.lab import panic2_feat as PF, panic2_common as C
from backtest.lab.fresh import dart_analyze as DA

t0 = time.time()
F = PF.load(); T, N = F.T, F.N
o, h, l, c = (getattr(F, k).astype(float) for k in 'ohlc'); tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 200)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
BUYC, SELLC, CAP, W = 0.00065, 0.00265, 1e8, 20


def lag(x, k): out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


cf = np.where(tr, c, np.nan)
for t in range(1, T):
    m = np.isnan(cf[t]); cf[t, m] = cf[t - 1, m]
pc = lag(cf, 1); r1 = c / pc - 1
mkt = np.nanmean(np.where(U, np.clip(r1, -.3, .3), np.nan), axis=1)
sig20 = np.full(T, np.nan)
for t in range(21, T): sig20[t] = np.nanstd(mkt[t - 20:t])
T5 = (mkt <= -0.03) & (mkt <= -3 * sig20)
limup, limdown, bad20 = F.limup.astype(bool), F.limdown.astype(bool), F.bad20.astype(bool)
ret20, ret60, hi250, vr, z, vol60, max20, cap = (getattr(F, k).astype(float) for k in ('ret20', 'ret60', 'hi250', 'vr', 'z', 'vol60', 'max20', 'caprank'))
pad20 = np.vstack([np.full((19, N), np.nan), cf]); hi20_prev = lag(np.nanmax(swv(pad20, 20, axis=0), axis=-1), 1)
proxy = (np.nan_to_num(vr, nan=0) >= 3) | (np.nan_to_num(z, nan=0) <= -3)
E = U & np.isfinite(r1) & ~bad20
didx = {d: i for i, d in enumerate(F.dates)}; cidx = {cd: j for j, cd in enumerate(F.codes)}
unclean = np.zeros((T, N), bool)
for line in open('/data/lab/dart/events_dart.jsonl', encoding='utf-8'):
    try: r = json.loads(line)
    except Exception: continue
    t, j = didx.get(r['date']), cidx.get(r['code'])
    if t is None or j is None or r['items'] is None: continue
    if DA.classify(r['items']) & {'악재', '실적', '조회', '투자판단'}: unclean[t, j] = True


def pct_mask(x, lo, hi):
    out = np.zeros((T, N), bool)
    for t in range(T):
        ii = np.where(E[t] & np.isfinite(x[t]))[0]
        if len(ii) < 20: continue
        order = ii[np.argsort(x[t, ii])]; n = len(order); out[t, order[int(n * lo):int(n * hi)]] = True
    return out


def lists(mask, score, desc=True):
    out = []
    for t in range(T):
        ii = np.where(mask[t] & np.isfinite(score[t]))[0]
        out.append(ii[np.argsort(-score[t, ii] if desc else score[t, ii])].tolist())
    return out


mom_top = pct_mask(ret20, 0.9, 1.0); mom_half = pct_mask(ret20, 0.5, 1.0); vol_top = pct_mask(lag(vol60, 1), 0.9, 1.0)
CAND = {
    'A': lists(E & T5[:, None] & (r1 <= -0.07) & ~limdown & ~proxy, r1, desc=False),
    'B': lists(U & limup & ~bad20, vr),
    'C': lists(E & (r1 >= 0.10) & (r1 < 0.20) & ~limup, r1),
    'D': lists(mom_top & ~limup, ret20),
    'E': lists(vol_top & mom_half & ~limup, ret20),
    'F': lists(E & (c > hi20_prev) & (vr >= 2) & ~limup, vr),
    'G': lists(F.tradable & (cap <= 200) & ~bad20 & ~limdown & ~limup & (ret60 > 0) & (max20 < 0.10) & np.isfinite(hi250), c / hi250),
    'H': lists(E & ~T5[:, None] & (r1 <= -0.10) & ~limdown & ~proxy & ~unclean, r1, desc=False),
}
# 전략 규칙: entry 'close'|'open', hold 최대 보유일, stop 손절(종가 판정), trail 추적, signal_only 신호일에만 진입, once 첫날만 진입
RULES = {
    'A 폭락 올인': dict(cand='A', entry='close', hold=5, exit_open=True, stop=None, trail=None, signal_only=True),
    'B 상한가 추격': dict(cand='B', entry='open', hold=10, exit_open=True, stop=None, trail=0.10, signal_only=True),
    'C 급등 추격': dict(cand='C', entry='close', hold=10, exit_open=True, stop=0.07, trail=0.10, signal_only=True),
    'D 모멘텀 집중': dict(cand='D', entry='close', hold=10, exit_open=True, stop=0.07, trail=None, signal_only=False),
    'E 고변동 모멘텀': dict(cand='E', entry='close', hold=10, exit_open=True, stop=0.07, trail=0.12, signal_only=False),
    'F 돌파 추격': dict(cand='F', entry='close', hold=15, exit_open=True, stop=None, trail=0.08, signal_only=True),
    'G 신고가+손절': dict(cand='G', entry='close', hold=99, exit_open=True, stop=0.07, trail=None, signal_only=False, once=True),
    'H 평소 급락': dict(cand='H', entry='close', hold=5, exit_open=True, stop=None, trail=None, signal_only=True),
}
PER = {'2011~2019': (C.didx(F, '20110103'), C.didx(F, '20200101')), '2020~2025': (C.didx(F, '20200101'), C.didx(F, '20260101')),
       '2026': (C.didx(F, '20260101'), T - W - 1)}
print(f'준비 {time.time() - t0:.0f}s', flush=True)


def run_month(s, rule, k, lock=None, mix=None):
    """rule: RULES 항목. mix: 'A' 면 T5 폭락일에 A 로 전환. 반환 (수익, 창 안 최고 NAV 비율, 매수 건수)"""
    e = s + W - 1
    cash, buys, pos = CAP, 0, {}             # j → [qty, entry_px, peak_close, exit_t, kind]
    pend_open = []                           # 다음 날 시가에 살 것 (j, kind)
    sell_open = set()                        # 다음 날 시가에 팔 것
    peak_nav, locked, entered_once = CAP, False, False
    ov = False

    def nav(t): return cash + sum(p[0] * cf[t, j] * (1 - SELLC) for j, p in pos.items())

    def sell(j, t, px):
        nonlocal cash
        if not tr[t, j] or not px > 0: return False
        cash += pos.pop(j)[0] * px * (1 - SELLC); return True

    def buy(j, t, px, kind, budget, hold):
        nonlocal cash, buys
        if j in pos or not tr[t, j] or not px > 0: return False
        q = np.floor(min(budget, cash) / (px * (1 + BUYC)))
        if q < 1: return False
        cash -= q * px * (1 + BUYC); buys += 1; pos[j] = [q, px, px, t + hold, kind]; return True

    for t in range(s, e + 1):
        # ── 시가: 예약 매도 · 예약 매수 · 만기 매도
        for j in list(sell_open): sell(j, t, o[t, j])
        sell_open = set()
        if locked:
            for j in list(pos): sell(j, t, o[t, j])
        for j in [j for j, p in pos.items() if p[3] == t]: sell(j, t, o[t, j])
        if not locked:
            for j, kind, hold in pend_open:
                buy(j, t, o[t, j], kind, nav(t - 1) / k, hold)
        pend_open = []
        # ── 종가: 손절·추적 판정(다음 날 시가 매도), 신호 진입, 목표 락
        if locked: continue
        r = RULES[rule]
        crash_now = mix == 'A' and T5[t] and t < e
        if crash_now and not ov:                                   # 폭락 전환: 전량 종가 매도 → A 후보 종가 매수
            for j in list(pos): sell(j, t, c[t, j])
            nv = nav(t)
            for j in CAND['A'][t][:k]: buy(j, t, c[t, j], 'A', nv / k, 6)
            ov = True
        else:
            if ov and not any(p[4] == 'A' for p in pos.values()): ov = False
            for j, p in pos.items():
                p[2] = max(p[2], cf[t, j])
                if (r['stop'] and cf[t, j] <= p[1] * (1 - r['stop'])) or (r['trail'] and cf[t, j] <= p[2] * (1 - r['trail'])):
                    sell_open.add(j)
            if not ov and t < e and not (r.get('once') and entered_once):
                free = k - len(pos) - len(pend_open)
                cands = [j for j in CAND[r['cand']][t] if j not in pos][:max(free, 0)]
                if cands: entered_once = True
                hold = r['hold'] + 1 if r['hold'] < 99 else 99        # 보유 n일 = n+1 일째 시가 매도 (99 = 월말까지)
                if r['entry'] == 'close':
                    nv = nav(t)
                    for j in cands: buy(j, t, c[t, j], r['cand'], nv / k, hold)
                else:
                    pend_open += [(j, r['cand'], hold) for j in cands]
        nv = nav(t); peak_nav = max(peak_nav, nv)
        if lock and nv >= CAP * (1 + lock): locked = True
    for j in list(pos): sell(j, e, c[e, j])
    return cash / CAP - 1, peak_nav / CAP - 1, buys


def evaluate(label, rule, k, lock=None, mix=None, P=print):
    out = {}
    for pn, (a, b) in PER.items():
        res = np.array([run_month(s, rule, k, lock, mix) for s in range(a, b)])
        r, pk, nb = res[:, 0], res[:, 1], res[:, 2]
        out[pn] = (r, pk, nb)
    cells = []
    for pn in PER:
        r, pk, nb = out[pn]
        cells.append(f'{(r >= .2).mean() * 100:4.1f} {(r >= .3).mean() * 100:4.1f} {(pk >= .3).mean() * 100:4.1f} {(r <= -.2).mean() * 100:4.1f} {np.median(r) * 100:+5.1f} {r.mean() * 100:+5.1f} {nb.mean():4.1f}')
    P(f'  {label:<24} | ' + ' | '.join(cells))
    return out


if __name__ == '__main__':
    lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
    P('꼬리 포착 전략 — 20일 창, 1억, 비용 포함. 각 기간 열: +20%↑ 확률 · +30%↑ 확률 · 창 안 +30% 찍음 · −20%↓ 확률 · 중앙 · 평균 · 월 매수건수 (%)')
    P(f'  {"전략 (k=종목 수)":<24} | {"2011~2019":^40} | {"2020~2025":^40} | {"2026":^40}')
    RES = {}
    for rule in RULES:
        for k in (1, 2, 3, 5):
            RES[(rule, k)] = evaluate(f'{rule} k{k}', rule, k, P=P)
        print(f'  … {rule} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    P('\n■ 혼합 I: 평소 D 모멘텀 집중, T5 폭락일엔 A 로 전환')
    for k in (1, 2, 3, 5):
        RES[('I', k)] = evaluate(f'I 혼합(D+A) k{k}', 'D 모멘텀 집중', k, mix='A', P=P)
    P('\n■ 목표 락 (+30% 찍으면 다음 날 시가 전량 매도 후 현금) 적용')
    for rule, k in (('A 폭락 올인', 2), ('C 급등 추격', 2), ('D 모멘텀 집중', 2), ('E 고변동 모멘텀', 2), ('G 신고가+손절', 5), ('G 신고가+손절', 2)):
        RES[(rule + ' 락', k)] = evaluate(f'{rule} k{k} 락30', rule, k, lock=0.30, P=P)
    for k in (2, 3):
        RES[('I 락', k)] = evaluate(f'I 혼합(D+A) k{k} 락30', 'D 모멘텀 집중', k, lock=0.30, mix='A', P=P)
    import pickle; pickle.dump({str(k): {pn: v[0] for pn, v in r.items()} for k, r in RES.items()}, open('/data/lab/tail_strategies.pkl', 'wb'))
    open('backtest/results/lab_tail_strategies.md', 'w', encoding='utf-8').write(
        '# 꼬리 포착 전략 비교 (backtest/lab/fresh/tail_strategies.py)\n\n정의는 스크립트 머리말 참고.\n\n```\n' + '\n'.join(lines) + '\n```\n')
    print(f'끝 {time.time() - t0:.0f}s', file=sys.stderr)
