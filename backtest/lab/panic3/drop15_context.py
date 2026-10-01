"""
이틀 합계 −15% 이하로 빠진 종목 (D−2 종가 → D 종가) 을 D+1 시가에 샀을 때 — 종목 · 섹터 · 시장 상황별로 쪼개 본다.
코스피 시총 100 + 코스닥 시총 150 (전날 순위, 상장폐지 포함). 같은 종목은 5거래일 안에 다시 뜬 신호를 건너뛴다.

결과 지표 (매수가 = D+1 시가, 비용 0.31% 는 '청산' 류에만 차감)
  5일 종가     D+5 종가에 판다 (매수일 = 1일째)
  최고 · 최저  5일 안 장중 최고가 · 최저가 / 매수가 − 1 (중간에 가장 좋았던 / 나빴던 순간)
  +5% · +10%   5일 안에 한 번이라도 닿은 비율
  익절5 · 익절10  +5% · +10% 지정가 매도, 못 닿으면 D+5 종가 (다음 날 시가가 이미 넘으면 시가에)
  최적일       1~10일 중 평균 종가 수익이 가장 높은 날과 그 값

조건 (전부 신호 전, D−2 기준으로 계산 — 급락 자체는 빼고 '평소 흐름' 만 본다)
  평소 추세    D−2 까지 60거래일 수익률
  고점 위치    D−2 종가 / 그 전 250거래일 최고 종가 − 1
  섹터 급락    같은 대분류 다른 종목들의 같은 이틀 평균 수익률 (섹터도 같이 빠졌나)
  섹터 추세    같은 대분류 다른 종목들의 D−2 까지 60일 평균 수익률
  시장         유니버스 전 종목의 같은 이틀 평균 수익률
  업종은 2018-11 스냅샷 (그 뒤 상장 · 그 전 폐지 종목은 '미분류')
"""
import numpy as np, warnings, json; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C

F = PF.load(); T, N = F.T, F.N
o, h, l, c = (getattr(F, k).astype(float) for k in 'ohlc')
tr = F.traded.astype(bool)
U = tr & (F.raw >= 1000) & (((F.lab == 1) & (F.kprank1 <= 100)) | ((F.lab == 2) & (F.kqrank1 <= 150)))
COST = 0.00015 * 2 + 0.002 + 0.0005 * 2


def ffill(x):
    x = x.copy()
    for t in range(1, len(x)):
        m = np.isnan(x[t]); x[t, m] = x[t - 1, m]
    return x


cf = ffill(np.where(tr, c, np.nan))
hh = np.where(tr, h, np.nan); ll = np.where(tr, l, np.nan)


def lag(x, k):
    out = np.full_like(x, np.nan); out[k:] = x[:-k]; return out


r1 = c / lag(cf, 1) - 1
r2 = cf / lag(cf, 2) - 1
r60 = lag(cf, 2) / lag(cf, 62) - 1                         # D−2 까지 60일
hi250 = np.full((T, N), np.nan)
roll = np.full(N, np.nan)
# D−2 기준 직전 250일 최고 종가 (단순 루프, 1회 계산)
from numpy.lib.stride_tricks import sliding_window_view as swv
pad = np.vstack([np.full((249, N), np.nan), cf])
hi = np.nanmax(swv(pad, 250, axis=0), axis=-1)             # hi[t] = max(cf[t−249..t])
hi250 = lag(hi, 2)
dd250 = lag(cf, 2) / hi250 - 1

# ── 업종 대분류
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
        gid[j] = len(GROUPS)                                # 기타
GNAMES = [g for g, _ in GROUPS] + ['기타', '미분류']
gid = np.where(gid < 0, len(GROUPS) + 1, gid)

peer = tr & (F.raw >= 1000)
r2c = np.clip(np.nan_to_num(r2), -0.5, 0.5); r60c = np.clip(np.nan_to_num(r60), -0.9, 3)
sec_r2 = np.full((T, N), np.nan); sec_r60 = np.full((T, N), np.nan)
for g in range(len(GROUPS) + 1):                           # 미분류는 섹터 계산 안 함
    cols = gid == g
    if cols.sum() < 3: continue
    pm = peer[:, cols] & np.isfinite(r2[:, cols])
    s2 = np.where(pm, r2c[:, cols], 0).sum(1, keepdims=True); n2 = pm.sum(1, keepdims=True)
    sec_r2[:, cols] = np.where(n2 - pm > 0, (s2 - np.where(pm, r2c[:, cols], 0)) / np.maximum(n2 - pm, 1), np.nan)
    pm6 = peer[:, cols] & np.isfinite(r60[:, cols])
    s6 = np.where(pm6, r60c[:, cols], 0).sum(1, keepdims=True); n6 = pm6.sum(1, keepdims=True)
    sec_r60[:, cols] = np.where(n6 - pm6 > 0, (s6 - np.where(pm6, r60c[:, cols], 0)) / np.maximum(n6 - pm6, 1), np.nan)
mkt_r2 = np.nanmean(np.where(U, np.clip(r2, -0.5, 0.5), np.nan), axis=1)

# ── 이벤트
sig = U & (r2 <= -0.15) & np.isfinite(r2)
a0 = C.didx(F, '20110101'); split = C.didx(F, '20200101'); end = T - 11
ev = []
lastev = np.full(N, -99)
for t in range(a0, end):
    for j in np.where(sig[t])[0]:
        if t - lastev[j] <= 5: continue
        if not tr[t + 1, j] or not np.isfinite(o[t + 1, j]) or o[t + 1, j] <= 0: continue
        lastev[j] = t
        ev.append((t, j))
ev = np.array(ev)
t_, j_ = ev[:, 0], ev[:, 1]
E = o[t_ + 1, j_]
path = np.stack([cf[t_ + k, j_] / E - 1 for k in range(1, 11)], 1)          # 1~10일째 종가 수익 (비용 전)
H5 = np.stack([hh[t_ + k, j_] for k in range(1, 6)], 1)
L5 = np.stack([ll[t_ + k, j_] for k in range(1, 6)], 1)
O5 = np.stack([np.where(tr[t_ + k, j_], o[t_ + k, j_], np.nan) for k in range(1, 6)], 1)
mx = np.nanmax(H5, 1) / E - 1
mn = np.nanmin(L5, 1) / E - 1
ret5 = path[:, 4] - COST


def tp(x):
    out = ret5.copy(); done = np.zeros(len(E), bool)
    tgt = E * (1 + x)
    for k in range(5):
        gap = (k > 0) & ~done & (O5[:, k] >= tgt)
        out[gap] = O5[gap, k] / E[gap] - 1 - COST; done |= gap
        hit = ~done & (H5[:, k] >= tgt)
        out[hit] = x - COST; done |= hit
    return out


tp5, tp10 = tp(0.05), tp(0.10)
X = dict(
    period=np.where(t_ < split, 0, 1),
    r1=r1[t_, j_], prevr1=r1[t_ - 1, j_], r2=r2[t_, j_],
    r60=r60[t_, j_], dd250=dd250[t_, j_], g=gid[j_], sr2=sec_r2[t_, j_], sr60=sec_r60[t_, j_], m2=mkt_r2[t_],
)
limit = (X['r1'] <= -0.295) | (X['prevr1'] <= -0.295)
months = (end - a0) / 21

lines = []; P = lines.append


def row(lab, m):
    n = int(m.sum())
    if n < 20:
        P(f'  {lab:<22} {n:>5}건  (표본 부족)'); return
    p = path[m] - COST
    best = int(np.nanargmax(np.nanmean(p, 0)))
    r20 = ret5[m & (X['period'] == 1)]
    P(f'  {lab:<22} {n:>5}건 {n / months:4.1f}/월 | 5일 {np.nanmean(ret5[m]) * 100:+5.2f}% (2020~ {np.nanmean(r20) * 100 if len(r20) >= 10 else np.nan:+5.2f}) '
      f'승 {np.nanmean(ret5[m] > 0) * 100:3.0f}% | 최고 {np.nanmean(mx[m]) * 100:+5.1f}% (중앙 {np.nanmedian(mx[m]) * 100:+4.1f}) '
      f'최저 {np.nanmean(mn[m]) * 100:+5.1f}% | +5%닿음 {np.nanmean(mx[m] >= 0.05) * 100:3.0f}% +10% {np.nanmean(mx[m] >= 0.10) * 100:3.0f}% '
      f'| 익절5 {np.nanmean(tp5[m]) * 100:+5.2f}% 익절10 {np.nanmean(tp10[m]) * 100:+5.2f}% | 최적 {best + 1}일 {np.nanmean(p[:, best]) * 100:+5.2f}%')


def cut(v, edges, labs):
    out = []
    for (lo, hi_), lb in zip(zip(edges[:-1], edges[1:]), labs):
        out.append((lb, (v > lo) & (v <= hi_)))
    return out


ALL = np.ones(len(E), bool)
P(f'이틀 합계 −15% 이하 → D+1 시가 매수 (2011~ 전체, 신호 {len(E):,}건)')
P('\n■ 전체 · 기간 · 하한가 포함 여부')
row('전체', ALL)
row('2011~2019', X['period'] == 0); row('2020~', X['period'] == 1)
row('이틀 중 하한가 있음', limit); row('하한가 없음', ~limit)
row('이틀 −15~−20%', X['r2'] > -0.20); row('이틀 −20~−30%', (X['r2'] <= -0.20) & (X['r2'] > -0.30)); row('이틀 −30% 이하', X['r2'] <= -0.30)
row('하루에 몰림 (D ≤ −10%)', (X['r1'] <= -0.10) & (X['prevr1'] > -0.05))
row('이틀 나눠 (둘 다 −5%↓)', (X['r1'] <= -0.05) & (X['prevr1'] <= -0.05))
NL = ~limit
P('\n■ 평소 추세 (D−2 까지 60일 수익률) — 이하 전부 하한가 제외')
for lb, m in cut(X['r60'], [-9, -0.30, -0.10, 0.10, 0.30, 0.6, 99], ['급락 (−30%↓)', '하락 (−30~−10%)', '횡보 (±10%)', '상승 (+10~+30%)', '급등 (+30~+60%)', '폭등 (+60%↑)']):
    row(lb, m & NL)
P('\n■ 52주 고점 대비 위치 (D−2)')
for lb, m in cut(X['dd250'], [-9, -0.6, -0.4, -0.2, -0.1, 9], ['고점 −60%↓', '고점 −40~−60%', '고점 −20~−40%', '고점 −10~−20%', '고점 근처 (−10% 안)']):
    row(lb, m & NL)
P('\n■ 시장 (유니버스 평균 같은 이틀)')
for lb, m in cut(X['m2'], [-9, -0.06, -0.03, -0.01, 9], ['시장 −6%↓ (폭락)', '시장 −3~−6%', '시장 −1~−3%', '시장 −1% 위 (혼자)']):
    row(lb, m & NL)
P('\n■ 섹터도 같이 빠졌나 (같은 대분류 다른 종목 평균, 같은 이틀)')
hasS = np.isfinite(X['sr2'])
for lb, m in cut(X['sr2'], [-9, -0.08, -0.04, -0.01, 9], ['섹터 −8%↓', '섹터 −4~−8%', '섹터 −1~−4%', '섹터 멀쩡 (−1% 위)']):
    row(lb, m & NL)
row('미분류 (섹터 모름)', ~hasS & NL)
P('\n■ 시장은 멀쩡한데 (−1% 위) 섹터가 같이 빠졌나')
calm = X['m2'] > -0.01
for lb, m in cut(X['sr2'], [-9, -0.04, -0.01, 9], ['섹터 −4%↓ (섹터 악재)', '섹터 −1~−4%', '섹터 멀쩡 (종목 악재)']):
    row(lb, m & NL & calm)
P('\n■ 섹터 평소 추세 (같은 대분류 60일 평균, D−2)')
for lb, m in cut(X['sr60'], [-9, -0.10, 0.0, 0.10, 0.25, 9], ['섹터 하락 (−10%↓)', '섹터 약세 (−10~0)', '섹터 약강세 (0~+10)', '섹터 상승 (+10~+25)', '섹터 급등 (+25%↑)']):
    row(lb, m & NL)
P('\n■ 업종 대분류')
for g, gn in enumerate(GNAMES):
    row(gn, (X['g'] == g) & NL)
P('\n■ 평소 추세 × 시장 (하한가 제외)')
for lb, m in cut(X['r60'], [-9, -0.10, 0.10, 0.30, 99], ['하락', '횡보', '상승', '급등']):
    row(f'{lb} · 시장 −3%↓', m & NL & (X['m2'] <= -0.03))
    row(f'{lb} · 시장 −3% 위', m & NL & (X['m2'] > -0.03))
P('\n■ 평소 추세 × 섹터 동반 여부 (시장 −3% 위, 하한가 제외)')
nm = NL & (X['m2'] > -0.03)
for lb, m in cut(X['r60'], [-9, -0.10, 0.10, 0.30, 99], ['하락', '횡보', '상승', '급등']):
    row(f'{lb} · 섹터 −3%↓', m & nm & (X['sr2'] <= -0.03))
    row(f'{lb} · 섹터 −3% 위', m & nm & (X['sr2'] > -0.03))

txt = '\n'.join(lines)
print(txt)
open('backtest/results/lab_drop15_context.md', 'w').write('# 이틀 −15% 급락 종목 — 상황별 (drop15_context.py)\n\n```\n' + txt + '\n```\n')
