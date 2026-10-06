"""
이틀 −15% 급락 종목 — 실제로 쓸 규칙으로 좁히기 (drop15_context.py 이어서)

신호 정의는 drop15_context.py 와 같다.
  이틀 수익률  D−2 종가 → D 종가 −15% 이하, 코스피 시총 100 + 코스닥 시총 150 (전날 순위, 상장폐지 포함)
  하한가 제외  D−1, D 중 하루라도 하한가 마감이면 뺀다
  재신호       같은 종목이 5거래일 안에 다시 뜨면 건너뛴다
  매수         D+1 시가

규칙 후보 (결과 보기 전에 정함)
  A  시장 2일 −3%↓
  B  시장 2일 −3%↓ 또는 섹터 2일 −4%↓
  C  B & 이틀 나눠 빠짐 (D−1, D 모두 −5%↓)
  D  B & 52주 고점 대비 −20%↓ (D−2 종가 / 그 전 250일 최고 종가)
  (참고) 전체 = 하한가만 뺀 모든 신호

청산 (매수일 = 1일째)
  5일 · 7일 · 10일   그날 종가
  트레일링 −7%       장중 고가 기준 고점에서 −7% 에 닿으면 판다. 같은 날 고가·저가 순서는 모르니 저가가 먼저라고 본다
                     (그날 손절 가격 = 전날까지 고점 × 0.93, 시가가 이미 아래면 시가). 10일째 종가까지 안 닿으면 그때 판다
  익절 +15% · 10일   +15% 지정가. 시가가 이미 넘으면 시가에, 못 닿으면 10일째 종가
비용: 매수 0.065% (수수료 0.015% + 슬리피지 0.05%), 매도 0.245% (+ 세금 0.2%) → 왕복 0.31%

포트폴리오 (1억 · 최대 5종목 · 종목당 2천만 · 한 달 = 20거래일)
  첫날 전날 종가 신호부터 받는다 (대회 시작 날 아침에 이미 아는 신호). 마지막 날 신호는 못 산다.
  빈 슬롯보다 신호가 많으면 이틀 낙폭이 큰 순. 이미 들고 있는 종목은 다시 안 산다. 정지일엔 못 판다.
  한 달 끝에 남은 종목은 마지막 날 종가에서 매도 비용을 뺀 값으로 평가.
  기간: 2011~2019 / 2020~ / 2020~2025 (2026 은 변동성이 유난히 커서 신호가 몰려 있어 따로 뺀 구간도 본다)
  단독   신호 없을 땐 현금
  혼합   평소엔 봇 기본 전략(52주 신고가 5종목: 시총 200, 60일 수익률 > 0, 최근 20일 +10% 급등일 없음, 종가 / 250일 최고가 순,
         하루 +10% 오르면 그날 종가 매도 후 다음 날 시가에 다음 순위로 채움). 급락 신호가 오면 신고가 보유 중 순위가 가장 낮은 것부터
         D+1 시가에 팔고 갈아탄다. 급락 포지션이 청산되면 다음 날 시가에 신고가 순위로 채운다. 봇의 시장 패닉 모드는 안 넣었다.
"""
import numpy as np, warnings, sys, time; warnings.filterwarnings('ignore')
from collections import defaultdict
from functools import lru_cache
from numpy.lib.stride_tricks import sliding_window_view as swv
from backtest.lab import panic2_feat as PF, panic2_common as C

t0 = time.time()
F = PF.load(); T, N = F.T, F.N
o, h, l, c = (getattr(F, k).astype(float) for k in 'ohlc')
tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
BUYC, SELLC = 0.00015 + 0.0005, 0.00015 + 0.002 + 0.0005
COST = BUYC + SELLC
CAP, K, MONTH = 1e8, 5, 20
DATES = F.dates


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


def lag(x, k):
    out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


cf = ffill(np.where(tr, c, np.nan))
r1 = c / lag(cf, 1) - 1
r2 = cf / lag(cf, 2) - 1
pad = np.vstack([np.full((249, N), np.nan), cf])
hi = np.nanmax(swv(pad, 250, axis=0), axis=-1)
dd250 = lag(cf, 2) / lag(hi, 2) - 1

# ── 섹터 · 시장 (drop15_context.py 와 같은 정의)
GROUPS = [
    ('반도체·전자·장비', ('반도체', '전자부품', '통신 및 방송 장비', '영상 및 음향', '컴퓨터 및 주변', '사진장비', '측정',
                    '전구', '절연선', '전동기', '일차전지', '기타 전기장비', '가정용 기기', '특수 목적용 기계')),
    ('바이오·헬스', ('의약', '의료', '생물', '자연과학 및 공학 연구개발')),
    ('화학·소재', ('화학', '플라스틱', '고무', '석유', '철강', '비철', '금속', '시멘트', '유리', '비금속', '요업', '종이', '나무')),
    ('기계·자동차·조선', ('기계', '자동차', '선박', '항공기', '철도', '무기', '운송장비')),
    ('IT·미디어·통신', ('소프트웨어', '프로그래밍', '정보', '출판', '영화', '방송', '광고', '오디오', '통신업', '기록매체', '오락')),
    ('금융', ('금융', '보험', '은행', '신탁')),
    ('건설', ('건설', '공사', '건축')),
    ('소비·유통', ('식품', '음료', '소매', '도매', '의복', '섬유', '담배', '사료', '육류', '낙농', '여행', '음식', '교습',
               '교육', '가구', '장난감', '귀금속', '상품', '판매', '생활', '작물')),
]
sec = np.asarray(F.sector)
gid = np.full(N, -1)
for j in range(N):
    if sec[j] < 0: continue
    name = F.sectors[sec[j]]
    for g, (_, keys) in enumerate(GROUPS):
        if any(k in name for k in keys): gid[j] = g; break
    else:
        gid[j] = len(GROUPS)
gid = np.where(gid < 0, len(GROUPS) + 1, gid)
peer = tr & (F.raw >= 1000)
r2c = np.clip(np.nan_to_num(r2), -0.5, 0.5)
sec_r2 = np.full((T, N), np.nan)
for g in range(len(GROUPS) + 1):
    cols = gid == g
    if cols.sum() < 3: continue
    pm = peer[:, cols] & np.isfinite(r2[:, cols])
    s2 = np.where(pm, r2c[:, cols], 0).sum(1, keepdims=True); n2 = pm.sum(1, keepdims=True)
    sec_r2[:, cols] = np.where(n2 - pm > 0, (s2 - np.where(pm, r2c[:, cols], 0)) / np.maximum(n2 - pm, 1), np.nan)
mkt_r2 = np.nanmean(np.where(U, np.clip(r2, -0.5, 0.5), np.nan), axis=1)

# ── 신호 (날짜 × 종목)
LIM = 0.295
base = U & np.isfinite(r2) & (r2 <= -0.15) & ~(r1 <= -LIM) & ~(lag(r1, 1) <= -LIM)
mk = (mkt_r2 <= -0.03)[:, None]
sk = np.nan_to_num(sec_r2, nan=0) <= -0.04
B_ = base & (mk | sk)
RULES = {
    'A 시장−3%': base & mk,
    'B 시장−3%|섹터−4%': B_,
    'C B&이틀나눠': B_ & (r1 <= -0.05) & (lag(r1, 1) <= -0.05),
    'D B&고점−20%': B_ & (dd250 <= -0.20),
    '(참고) 전체': base,
}
# 같은 종목 5거래일 안 재신호 무시
for k_, m in RULES.items():
    m = m.copy(); last = np.full(N, -99)
    for t in range(T):
        js = np.where(m[t])[0]
        keep = t - last[js] > 5
        m[t, js[~keep]] = False
        last[js[keep]] = t
    RULES[k_] = m

EXITS = ['5일', '7일', '10일', '트레일링−7%', '익절+15%']


@lru_cache(maxsize=None)
def exit_trade(t_buy, j, kind):
    """t_buy 시가에 산 j 의 청산 (청산일, 청산 가격). 정지일엔 못 팔고 다음 거래일로 미룬다. 데이터 끝이면 None."""
    E = o[t_buy, j]
    hold = {'5일': 5, '7일': 7, '10일': 10}.get(kind, 10)
    last = t_buy + hold - 1
    if last >= T: return None
    if kind in ('트레일링−7%', '익절+15%'):
        peak = E if kind == '트레일링−7%' else None
        for t in range(t_buy, last + 1):
            if not tr[t, j]: continue
            if kind == '트레일링−7%':
                stop = peak * 0.93
                if t > t_buy and o[t, j] <= stop: return t, o[t, j]
                if l[t, j] <= stop: return t, stop
                peak = max(peak, h[t, j])
            else:
                tgt = E * 1.15
                if t > t_buy and o[t, j] >= tgt: return t, o[t, j]
                if h[t, j] >= tgt: return t, tgt
    t = last
    while t < T and not tr[t, j]: t += 1
    if t >= T: return None
    return t, c[t, j]


# 이벤트 표: 규칙 × 청산 → 건별 수익
print(f'준비 {time.time() - t0:.0f}s', flush=True)
a0, split, end = C.didx(F, '20110103'), C.didx(F, '20200101'), T - 1
TRADE = {}
for rn, m in RULES.items():
    for ex in EXITS:
        rows = []
        for t in range(a0, end):
            for j in np.where(m[t])[0]:
                if t + 1 >= T or not tr[t + 1, j] or not o[t + 1, j] > 0: continue
                res = exit_trade(t + 1, int(j), ex)
                if res is None: continue
                te, px = res
                rows.append((t, j, te, px / o[t + 1, j] * (1 - SELLC) / (1 + BUYC) - 1))
        TRADE[rn, ex] = np.array(rows) if rows else np.zeros((0, 4))
print(f'이벤트 {time.time() - t0:.0f}s', flush=True)

# ── 신고가 순위 (봇 기본 전략)
cap = F.caprank.astype(float)
nh_score = np.where(F.tradable & (cap <= 200) & (F.ret60 > 0) & (F.max20 < 0.10) & np.isfinite(F.hi250), c / F.hi250, np.nan)
NH = []
for t in range(T):
    s = nh_score[t]; idx = np.where(np.isfinite(s))[0]
    NH.append(idx[np.argsort(-s[idx])[:30]].tolist())
surge = tr & (r1 >= 0.10)

SIGDAY = {rn: [np.where(m[t])[0][np.argsort(r2[t, np.where(m[t])[0]])].tolist() for t in range(T)] for rn, m in RULES.items()}


def run_month(s, rn, ex, mixed):
    """s = 대회 첫날. 반환: (월 수익률, 급락 매수 건수, 매매금액, 신호 있었나)"""
    e = s + MONTH - 1
    cash, amt, n_dip = CAP, 0.0, 0
    pos = {}                                   # j → dict(kind, qty, exit_t, exit_px, rank)
    pend = {}                                  # 오늘 시가에 살 것: j → kind
    had_sig = any(SIGDAY[rn][t] for t in range(s - 1, e))

    def sell(j, px):
        nonlocal cash, amt
        p = pos.pop(j); v = p['qty'] * px
        cash += v * (1 - SELLC); amt += v

    def buy(j, t, kind, budget):
        nonlocal cash, amt
        px = o[t, j]
        if not (tr[t, j] and px > 0): return False
        qty = np.floor(min(budget, cash) / (px * (1 + BUYC)))
        if qty < 1: return False
        cash -= qty * px * (1 + BUYC); amt += qty * px
        p = dict(kind=kind, qty=qty)
        if kind == 'dip':
            res = exit_trade(t, int(j), ex)
            p['exit'] = res if res is not None else (10 ** 9, np.nan)
        pos[j] = p
        return True

    for t in range(s, e + 1):
        # ── 시가: 급락 신호 (전날 종가 판정)
        sigs = [j for j in SIGDAY[rn][t - 1] if j not in pos]
        if sigs:
            free = K - len(pos)
            want = sigs[:K]
            if mixed and len(want) > free:
                # 신고가 보유 중 순위 낮은 것부터 판다 (정지면 못 팖)
                nh_pos = sorted([j for j, p in pos.items() if p['kind'] == 'nh' and tr[t, j]],
                                key=lambda j: NH[t - 1].index(j) if j in NH[t - 1] else 999, reverse=True)
                for j in nh_pos[:len(want) - free]: sell(j, o[t, j])
            free = K - len(pos)
            for j in want[:free]:
                if buy(j, t, 'dip', CAP / K): n_dip += 1
        # ── 시가: 신고가 채우기 (첫날 · 급락 청산 뒤 · 급등 매도 뒤)
        if mixed:
            for j in NH[t - 1]:
                if len(pos) >= K: break
                if j in pos or not tr[t, j]: continue
                buy(j, t, 'nh', CAP / K)
        # ── 장중 · 종가: 급락 청산, 신고가 급등 매도
        for j in list(pos):
            p = pos[j]
            if p['kind'] == 'dip' and p['exit'][0] == t:
                sell(j, p['exit'][1])
            elif p['kind'] == 'nh' and surge[t, j] and t < e:
                sell(j, c[t, j])
    nav = cash + sum(p['qty'] * cf[e, j] * (1 - SELLC) for j, p in pos.items())
    return nav / CAP - 1, n_dip, amt, had_sig


def mstats(r):
    r = np.asarray(r)
    return (f'평균 {r.mean() * 100:+5.2f}% 중앙 {np.median(r) * 100:+5.2f}% 손실달 {(r < 0).mean() * 100:3.0f}% '
            f'−10%↓ {(r <= -0.1).mean() * 100:3.0f}% +10%↑ {(r >= 0.1).mean() * 100:3.0f}%')


y26 = C.didx(F, '20260101')
PERS = {'2011~2019': (a0 + 1, split), '2020~': (split, T - MONTH - 1), '2020~2025': (split, y26 - MONTH)}
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))

P(f'이틀 −15% 급락 (하한가 제외) → D+1 시가 매수 · 비용 왕복 0.31% · 2011-01 ~ {DATES[-1]}')
P('\n■ 1. 규칙별 신호 수 (재신호 제외 후)')
nmon = (end - a0) / 21
for rn, m in RULES.items():
    n1 = m[a0:split].sum(); n2 = m[split:end].sum()
    days = m[a0:end].any(1).sum()
    P(f'  {rn:<18} {m[a0:end].sum():5d}건 ({m[a0:end].sum() / nmon:4.1f}/월) · 2011~19 {n1} · 2020~ {n2} · 신호 있는 날 {days}일')

P('\n■ 2. 건별 수익 (규칙 × 청산) — 평균 / 승률 / t (같은 날 신호는 한 묶음)')
P(f'  {"규칙":<18}' + ''.join(f'| {ex:^28}' for ex in EXITS))


def tstat(arr):
    if len(arr) < 3: return np.nan
    d = defaultdict(list)
    for t, _, _, r in arr: d[int(t)].append(r)
    v = np.array([np.mean(x) for x in d.values()])
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan


for pn, (a, b) in [('전체', (a0, end))] + list(PERS.items()):
    P(f'  [{pn}]')
    for rn in RULES:
        line = f'  {rn:<18}'
        for ex in EXITS:
            A = TRADE[rn, ex]; A = A[(A[:, 0] >= a) & (A[:, 0] < b)] if len(A) else A
            if len(A) < 10: line += f'| {"표본 부족 (" + str(len(A)) + "건)":^28}'; continue
            line += f'| {A[:, 3].mean() * 100:+6.2f}% 승{(A[:, 3] > 0).mean() * 100:3.0f}% t{tstat(A):+5.1f} n{len(A):4d} '
        P(line)

P('\n■ 3. 한 달(20거래일) 포트폴리오 — 1억 · 최대 5종목 · 급락 단독 (신호 없으면 현금)')
P('      신호없는달 = 그 한 달 동안 살 신호가 한 번도 없었던 비율')
SIM = {}
for rn in RULES:
    for ex in EXITS:
        for mixed in (False, True):
            for pn, (a, b) in PERS.items():
                SIM[rn, ex, mixed, pn] = [run_month(s, rn, ex, mixed) for s in range(a, b)]
    print(f'  시뮬 {rn} {time.time() - t0:.0f}s', flush=True)
for rn in RULES:
    P(f'  {rn}')
    for ex in EXITS:
        for pn in PERS:
            R = SIM[rn, ex, False, pn]
            r = np.array([x[0] for x in R]); nd = np.array([x[1] for x in R]); ns = np.array([x[3] for x in R])
            amt = np.array([x[2] for x in R])
            P(f'    {ex:<9} {pn:<9} {mstats(r)} | 거래 {nd.mean():4.1f}건/월 매매금액 {amt.mean() / 1e8:4.1f}억 | 신호없는달 {(~ns).mean() * 100:3.0f}%')

P('\n■ 4. 혼합 — 평소 신고가 5종목, 급락 신호 오면 갈아타기 (같은 시뮬레이터)')
# 비교 기준: 신호를 절대 안 받는 신고가 단독 (빈 규칙)
RULES['신고가만'] = np.zeros((T, N), bool); SIGDAY['신고가만'] = [[] for _ in range(T)]
BASE = {}
for pn, (a, b) in PERS.items():
    R = BASE[pn] = [run_month(s, '신고가만', '5일', True) for s in range(a, b)]
    r = np.array([x[0] for x in R]); amt = np.array([x[2] for x in R])
    P(f'  신고가만 (기준)  {pn:<9} {mstats(r)} | 매매금액 {amt.mean() / 1e8:4.1f}억')
for rn in [k for k in RULES if k != '신고가만']:
    P(f'  {rn}')
    for ex in EXITS:
        for pn in PERS:
            R = SIM[rn, ex, True, pn]
            r = np.array([x[0] for x in R]); nd = np.array([x[1] for x in R]); amt = np.array([x[2] for x in R])
            P(f'    {ex:<9} {pn:<9} {mstats(r)} | 급락 {nd.mean():4.1f}건/월 매매금액 {amt.mean() / 1e8:4.1f}억')

P('\n■ 5. 연도별 · 몰림 확인 (건별, 10일 종가 청산)')
covid = (C.didx(F, '20200215'), C.didx(F, '20200501'))
P('  코로나 뺌 = 2020-02-15 ~ 2020-04-30 과 겹치는 한 달 시작일을 뺀다')
for pn in ('2020~', '2020~2025'):
    _kb = [x for s, x in zip(range(*PERS[pn]), BASE[pn]) if not (s <= covid[1] and s + MONTH > covid[0])]
    P(f'  (기준) 신고가만 {pn:<9} 코로나 뺌: {mstats([x[0] for x in _kb])}')
for rn in [k for k in RULES if k != '신고가만']:
    A = TRADE[rn, '10일']
    if len(A) == 0: continue
    yrs = np.array([DATES[int(t)][:4] for t in A[:, 0]]); mos = np.array([DATES[int(t)][:6] for t in A[:, 0]])
    tot = A[:, 3].sum()
    P(f'  {rn}  (총 {len(A)}건, 수익 합 {tot * 100:+.0f}%p)')
    P('    ' + ' '.join(f'{y}:{(yrs == y).sum():3d}건 {A[yrs == y, 3].mean() * 100:+5.1f}%' for y in sorted(set(yrs))))
    by = sorted(((A[mos == mo, 3].sum(), mo, (mos == mo).sum()) for mo in set(mos)), reverse=True)
    P('    수익 많이 낸 달: ' + ', '.join(f'{mo[:4]}-{mo[4:]} {n}건 {s * 100:+.0f}%p ({s / tot * 100:.0f}%)' for s, mo, n in by[:5]))
    top3 = {mo for _, mo, _ in by[:3]}
    for lab_, ex_mask in (('2020-03 뺌', mos != '202003'), ('상위 3개 달 뺌', ~np.isin(mos, list(top3))), ('2026 뺌', yrs != '2026')):
        B2 = A[ex_mask]
        P(f'    {lab_:<12} {len(B2):4d}건 평균 {B2[:, 3].mean() * 100:+5.2f}% 승 {(B2[:, 3] > 0).mean() * 100:3.0f}% t {tstat(B2):+4.1f}')
    # 포트폴리오도: 2020-02-15 ~ 2020-04-30 과 겹치는 달 뺌
    for ex in ('5일', '10일'):
        for pn in ('2020~', '2020~2025'):
            for mixed in (False, True):
                R = SIM[rn, ex, mixed, pn]; ss = range(*PERS[pn])
                keep = [x for s, x in zip(ss, R) if not (s <= covid[1] and s + MONTH > covid[0])]
                r = np.array([x[0] for x in keep])
                P(f'    포트폴리오 {ex:<3} {pn:<9} {"혼합" if mixed else "단독"} 코로나 뺌: {mstats(r)}')

txt = '\n'.join(lines)
open('backtest/results/lab_drop15_rules.md', 'w', encoding='utf-8').write(
    '# 이틀 −15% 급락 — 규칙 좁히기 (drop15_rules.py)\n\n규칙·청산·시뮬레이션 정의는 스크립트 머리말 참고.\n\n```\n' + txt + '\n```\n')
print(f'끝 {time.time() - t0:.0f}s')
