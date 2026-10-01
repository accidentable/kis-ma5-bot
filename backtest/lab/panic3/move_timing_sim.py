"""
move_timing.py 후속: (1) 하락 신호를 '시장도 같이 빠진 날' 과 '혼자 빠진 날' 로 나눠 본다
                    (2) 1억 · 최대 5종목 (종목당 2천만) · 한 달(20거래일) 시뮬레이션

시장 = 같은 날 유니버스 전 종목 평균 등락률.  혼자 = 시장 > −1%,  같이 = 시장 ≤ −2%
시뮬레이션: 신호가 뜨면 정해진 시점에 사서 H 거래일 뒤 판다. 빈 슬롯보다 신호가 많으면 그날 많이 빠진 순.
           월말에 아직 들고 있으면 마지막 날 종가로 평가. 시작일을 바꿔가며 한 달 수익률 분포를 본다.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_common as C
from backtest.lab.panic3 import move_timing as M      # 계산 재사용 (import 시 표를 한 번 출력한다)

F, T, U, r1, prev = M.F, M.T, M.U, M.r1, M.prev
mkt = np.nanmean(np.where(U, r1, np.nan), axis=1)
LIM = 0.295
DOWN = {
    '−3~−5%': (r1 <= -0.03) & (r1 > -0.05),
    '−5~−10%': (r1 <= -0.05) & (r1 > -0.10),
    '−10% 이하': (r1 <= -0.10) & (r1 > -LIM),
    '이틀 연속 −5%↓': (r1 <= -0.05) & (prev <= -0.05) & (r1 > -LIM),
}
alone = (mkt > -0.01)[:, None]
together = (mkt <= -0.02)[:, None]

print('\n\n════ 하락 신호: 혼자 빠진 날 vs 시장도 빠진 날 (D+1 시가 매수) ════')
for sname, s in DOWN.items():
    for cname, cm in (('혼자(시장 > −1%)', alone), ('같이(시장 ≤ −2%)', together)):
        m = s & U & cm
        for pn, (a, b) in M.PER.items():
            line = f'  {sname:<12} {cname:<14} [{pn:<9}]'
            st0 = M.stats(m, a, b, M.RET['D+1 시가', 5])
            if st0 is None:
                print(line + ' 표본 부족'); continue
            line += f' {st0[3]:6,}건 ({st0[5] / (b - a) * 100:3.0f}% 날짜)'
            for n in (1, 3, 5, 10):
                st = M.stats(m, a, b, M.RET['D+1 시가', n]); se = M.stats(m, a, b, M.EXC['D+1 시가', n])
                line += f' | {n}일 {st[0] * 100:+5.2f}% (초과 {se[0] * 100:+5.2f}, 승 {st[2] * 100:.0f}%, t {se[4]:+.1f})'
            print(line, flush=True)

# ── 포트폴리오 시뮬레이션 ─────────────────────────────────────
o, c = M.o, M.c
COST = M.COST


def sim(sig, k, kind, H, starts, slots=5, key=None):
    """sig[t,j] 신호 → t+k 일에 kind(o/c) 가격으로 매수, H 거래일 뒤 같은 kind 로 매도."""
    key = r1 if key is None else key
    px = o if kind == 'o' else c
    exitpx = M.of if kind == 'o' else M.cf
    rets, ntr, wins = [], [], []
    for s in starts:
        e = s + 20                            # 평가 마지막 날 (종가)
        free = [s] * slots; acc = 0.0; n = 0; w = 0
        for t in range(s - k, e - k):        # 신호일 (매수일 = t+k 가 창 안)
            js = np.where(sig[t])[0]
            if not len(js): continue
            js = js[np.argsort(key[t, js])]
            d = t + k
            for j in js:
                fs = [i for i in range(slots) if free[i] <= d]
                if not fs: break
                if not F.traded[d, j] or not np.isfinite(px[d, j]): continue
                ent = px[d, j]
                x = d + H
                if x > e or (kind == 'o' and x == e):
                    r = M.cf[e, j] / ent - 1 - COST; x = e
                else:
                    r = exitpx[x, j] / ent - 1 - COST
                if not np.isfinite(r): continue
                free[fs[0]] = x + (1 if kind == 'c' else 0)   # 시가 매도한 슬롯은 그날 시가 매수에 다시 쓸 수 있다
                acc += r / slots; n += 1; w += r > 0
        rets.append(acc); ntr.append(n); wins.append(w / max(n, 1))
    return np.array(rets), np.array(ntr), np.array(wins)


RULES = [
    ('이틀 연속 −5%↓', DOWN['이틀 연속 −5%↓'], 1, 'o', 5),
    ('이틀 연속 −5%↓', DOWN['이틀 연속 −5%↓'], 0, 'c', 5),
    ('이틀 연속 −5%↓', DOWN['이틀 연속 −5%↓'], 1, 'o', 3),
    ('−10% 이하', DOWN['−10% 이하'], 1, 'o', 5),
    ('−10% 이하', DOWN['−10% 이하'], 0, 'c', 5),
    ('−5~−10%', DOWN['−5~−10%'], 1, 'o', 5),
    ('−5% 이하 전부', (r1 <= -0.05) & (r1 > -LIM), 1, 'o', 5),
    ('−5% 이하 전부', (r1 <= -0.05) & (r1 > -LIM), 0, 'c', 5),
    ('−5% 이하 · 혼자', (r1 <= -0.05) & (r1 > -LIM) & alone, 1, 'o', 5),
    ('−3~−5%', DOWN['−3~−5%'], 1, 'o', 5),
    ('+10% 이상', (r1 >= 0.10) & (r1 < LIM), 0, 'c', 5),
    ('이틀 연속 +5%↑', (r1 >= 0.05) & (prev >= 0.05) & (r1 < LIM), 0, 'c', 3),
]
print('\n\n════ 1억 · 최대 5종목 · 한 달(20거래일) ════')
print('  규칙 / 매수 / 보유 | 기간: 월평균 · 중앙 · 손실달 · −10%↓ · +10%↑ · 거래수 · 승률')
base = {}
for pn, (a, b) in M.PER.items():
    st = list(range(a + 2, b - 21, 1))
    # 비교 기준: 유니버스 동일가중 보유 한 달
    bm = np.array([np.nanmean(np.where(U[s], M.cf[s + 20] / c[s] - 1, np.nan)) for s in st])
    base[pn] = bm
    print(f'  [기준] 유니버스 전 종목 한 달 보유 [{pn}] 월평균 {bm.mean() * 100:+.2f}% · 중앙 {np.median(bm) * 100:+.2f}%')
out = []
for name, s, k, kind, H in RULES:
    sig = s & U
    row = f'  {name:<14} {"D+1 시가" if (k, kind) == (1, "o") else "D 종가":<7} {H}일'
    for pn, (a, b) in M.PER.items():
        st = list(range(a + 2, b - 21, 1))
        r, n, w = sim(sig, k, kind, H, st)
        row += (f' | [{pn}] {r.mean() * 100:+5.2f}% · {np.median(r) * 100:+5.2f}% · {(r < 0).mean() * 100:3.0f}% · '
                f'{(r <= -0.1).mean() * 100:3.0f}% · {(r >= 0.1).mean() * 100:3.0f}% · {n.mean():4.1f}건 · {np.nanmean(w) * 100:.0f}%')
        out.append(dict(rule=name, k=k, kind=kind, hold=H, per=pn, mean=r.mean(), med=float(np.median(r)),
                        loss=float((r < 0).mean()), p10=float((r >= 0.1).mean()), n=float(n.mean())))
    print(row, flush=True)

import json
json.dump(out, open('backtest/results/lab_move_timing_sim.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
