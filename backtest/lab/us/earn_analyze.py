"""
실적 발표 반응 뒤 20일 — 조건부 통계 (events.pkl). 진입은 반응일 종가 c[R] (표시 없으면).
열: N · 평균20 · 중앙20 · 시장조정20(−SPY) · t · +20%↑ · +30%↑ · 최고가+30%↑ · 최고가+50%↑ · −20%↓ · 평균1일 · 평균5일 · 다음날 시가(밤)
"""
import numpy as np, pandas as pd, sys, time
E = pd.read_pickle('/data/lab/us/earnings/events.pkl')
E = E[E.c20.notna()].copy(); E['adj20'] = E.c20 - E.uni20            # 조정 = 대상 종목 동일가중 평균 대비 (SPY 대비는 PEAD 표에서 같이)
E['c1_c20'] = (1 + E.c20) / (1 + E.c1) - 1; E['c1_max20'] = (1 + E.max20) / (1 + E.c1) - 1       # R+1 종가 진입 기준
E['c2_c20'] = (1 + E.c20) / (1 + E.c2) - 1; E['c2_max20'] = (1 + E.max20) / (1 + E.c2) - 1       # R+2 종가 진입 기준
E['absret'] = E.dayret.abs()
PER = {'2011~2019': (2011, 2019), '2020~2025': (2020, 2025), '2026': (2026, 2026)}
lines = []; P = lambda s='': (lines.append(s), print(s, flush=True))
def row(name, g, base=''):
    if len(g) < 30: return f'  {name:<34} N={len(g):>5}  표본 부족'
    if base:                                                   # R+1/R+2 종가 진입: 20일·최고가만 바꿔 계산
        g = g.assign(c20=g[f'{base}_c20'], max20=g[f'{base}_max20'], adj20=g[f'{base}_c20'] - g.uni20)
    t = g.adj20.mean() / (g.adj20.std() / np.sqrt(len(g))) if g.adj20.std() > 0 else np.nan
    return (f'  {name:<34} N={len(g):>5} {g.c20.mean() * 100:+6.1f} {g.c20.median() * 100:+6.1f} {g.adj20.mean() * 100:+6.1f} {t:5.1f} | '
            f'{(g.c20 >= .2).mean() * 100:5.1f} {(g.c20 >= .3).mean() * 100:5.1f} {(g.max20 >= .3).mean() * 100:5.1f} {(g.max20 >= .5).mean() * 100:5.1f} {(g.c20 <= -.2).mean() * 100:5.1f} | '
            f'{g.c1.mean() * 100:+5.2f} {g.c5.mean() * 100:+5.2f} {g.o1.mean() * 100:+5.2f}')
HDR = f'  {"":<34} {"":>7} {"평균20":>6} {"중앙20":>6} {"조정20":>6} {"t":>5} | {"+20":>5} {"+30":>5} {"고+30":>5} {"고+50":>5} {"-20":>5} | {"1일":>5} {"5일":>5} {"밤":>5}'
def section(title, groups, pers=PER):
    P(f'\n■ {title}')
    for pn, (a, b) in pers.items():
        sub = E[(E.year >= a) & (E.year <= b)]
        if len(sub) < 30: continue
        P(f' [{pn}]'); P(HDR)
        for name, cond, *base in groups:
            P(row(name, sub[cond(sub)], base[0] if base else ''))
bins_ret = [('≤−20%', lambda d: d.dayret <= -.2), ('−20~−10%', lambda d: (d.dayret > -.2) & (d.dayret <= -.1)), ('−10~−5%', lambda d: (d.dayret > -.1) & (d.dayret <= -.05)),
            ('−5~0%', lambda d: (d.dayret > -.05) & (d.dayret <= 0)), ('0~+5%', lambda d: (d.dayret > 0) & (d.dayret <= .05)), ('+5~10%', lambda d: (d.dayret > .05) & (d.dayret <= .1)),
            ('+10~20%', lambda d: (d.dayret > .1) & (d.dayret <= .2)), ('≥+20%', lambda d: d.dayret > .2)]
P('실적 반응일 R 종가 매수 → 20거래일. 조정20 = 20일 수익률 − 대상 종목 동일가중 평균 20일. 비용 전. 대상: 현재 시총 $2B↑ 생존 종목 (생존 편향 주의)')
section('전체 · 반응일 수익률(dayret) 구간별', [('전체', lambda d: d.dayret.notna())] + bins_ret)
section('갭(전일 종가→시가) 구간별', [('갭 ≤−15%', lambda d: d.gap <= -.15), ('갭 −15~−8%', lambda d: (d.gap > -.15) & (d.gap <= -.08)), ('갭 +8~15%', lambda d: (d.gap >= .08) & (d.gap < .15)), ('갭 ≥+15%', lambda d: d.gap >= .15)])
section('EPS 서프라이즈 구간별 (반응 무관)', [('서프 ≤−20%', lambda d: d.surprise <= -20), ('서프 −20~0%', lambda d: (d.surprise > -20) & (d.surprise < 0)), ('서프 0~+10%', lambda d: (d.surprise >= 0) & (d.surprise < 10)),
                                     ('서프 +10~30%', lambda d: (d.surprise >= 10) & (d.surprise < 30)), ('서프 ≥+30%', lambda d: d.surprise >= 30), ('서프 없음', lambda d: d.surprise.isna())])
section('서프라이즈 × 반응 방향', [('서프+ & 반응 ≥+5%', lambda d: (d.surprise > 0) & (d.dayret >= .05)), ('서프+ & 반응 ≥+10%', lambda d: (d.surprise > 0) & (d.dayret >= .1)),
                               ('서프+ & 반응 ≤−5% (셀더뉴스)', lambda d: (d.surprise > 0) & (d.dayret <= -.05)), ('서프+ & 반응 ≤−10%', lambda d: (d.surprise > 0) & (d.dayret <= -.1)),
                               ('서프− & 반응 ≥+5% (악재 선반영)', lambda d: (d.surprise < 0) & (d.dayret >= .05)), ('서프− & 반응 ≤−5%', lambda d: (d.surprise < 0) & (d.dayret <= -.05)),
                               ('서프− & 반응 ≤−10%', lambda d: (d.surprise < 0) & (d.dayret <= -.1))])
UP = lambda d: d.dayret >= .1; DN = lambda d: d.dayret <= -.1
section('급등(≥+10%) 세부', [('급등 전체', UP), ('급등 & 시가→종가 + (강하게 마감)', lambda d: UP(d) & (d.intra > 0)), ('급등 & 시가→종가 − (밀림)', lambda d: UP(d) & (d.intra < 0)),
                        ('급등 & 거래 2~5배', lambda d: UP(d) & (d.vr >= 2) & (d.vr < 5)), ('급등 & 거래 5배↑', lambda d: UP(d) & (d.vr >= 5)), ('급등 & 거래 2배 미만', lambda d: UP(d) & (d.vr < 2)),
                        ('급등 & 52주 고가 −5% 안', lambda d: UP(d) & (d.hi250 >= -.05)), ('급등 & 52주 고가 −5~−20%', lambda d: UP(d) & (d.hi250 < -.05) & (d.hi250 >= -.2)), ('급등 & 52주 고가 −20% 밖', lambda d: UP(d) & (d.hi250 < -.2)),
                        ('급등 & 발표 전 60일 +20%↑', lambda d: UP(d) & (d.ret60 >= .2)), ('급등 & 발표 전 60일 −20%↓', lambda d: UP(d) & (d.ret60 <= -.2)),
                        ('급등 & 직전 분기도 +5%↑', lambda d: UP(d) & (d.prev_dayret >= .05)), ('급등 & 직전 분기 −5%↓', lambda d: UP(d) & (d.prev_dayret <= -.05)),
                        ('급등 & 시총 100위 안', lambda d: UP(d) & (d.caprank <= 100)), ('급등 & 시총 101~500', lambda d: UP(d) & (d.caprank > 100) & (d.caprank <= 500)), ('급등 & 시총 501~', lambda d: UP(d) & (d.caprank > 500)),
                        ('급등 & 60일 변동성 상위(≥4%/일)', lambda d: UP(d) & (d.vol60 >= .04)), ('급등 & 60일 변동성 하위(<2%/일)', lambda d: UP(d) & (d.vol60 < .02)),
                        ('급등 & 다음날 종가 + → R+1 종가 매수', lambda d: UP(d) & (d.c1 > 0), 'c1'), ('급등 & 다음날 종가 − → R+1 종가 매수', lambda d: UP(d) & (d.c1 < 0), 'c1'),
                        ('급등 & 2일 뒤 −3%↓ 눌림 → R+2 종가 매수', lambda d: UP(d) & (d.c2 <= -.03), 'c2'), ('급등 & 2일 뒤 +3%↑ 연장 → R+2 종가 매수', lambda d: UP(d) & (d.c2 >= .03), 'c2')])
section('급락(≤−10%) 세부', [('급락 전체', DN), ('급락 & 시가→종가 + (낙폭 회복)', lambda d: DN(d) & (d.intra > 0)), ('급락 & 시가→종가 − (더 밀림)', lambda d: DN(d) & (d.intra < 0)),
                        ('급락 & 거래 5배↑', lambda d: DN(d) & (d.vr >= 5)), ('급락 & 거래 2배 미만', lambda d: DN(d) & (d.vr < 2)),
                        ('급락 & 52주 고가 −10% 안', lambda d: DN(d) & (d.hi250 >= -.1)), ('급락 & 52주 고가 −40% 밖', lambda d: DN(d) & (d.hi250 < -.4)),
                        ('급락 & 발표 전 60일 +20%↑', lambda d: DN(d) & (d.ret60 >= .2)), ('급락 & 발표 전 60일 −20%↓', lambda d: DN(d) & (d.ret60 <= -.2)),
                        ('급락 & 직전 분기 +5%↑', lambda d: DN(d) & (d.prev_dayret >= .05)), ('급락 & 직전 분기 −5%↓', lambda d: DN(d) & (d.prev_dayret <= -.05)),
                        ('급락 & 시총 100위 안', lambda d: DN(d) & (d.caprank <= 100)), ('급락 & 시총 101~500', lambda d: DN(d) & (d.caprank > 100) & (d.caprank <= 500)), ('급락 & 시총 501~', lambda d: DN(d) & (d.caprank > 500)),
                        ('급락 & 다음날 종가 + → R+1 종가 매수', lambda d: DN(d) & (d.c1 > 0), 'c1'), ('급락 & 다음날 −3%↓ 추가 하락 → R+1 종가 매수', lambda d: DN(d) & (d.c1 < -.03), 'c1'),
                        ('급락 & 2일 뒤 −5%↓ 과매도 → R+2 종가 매수', lambda d: DN(d) & (d.c2 <= -.05), 'c2'), ('급락 & 2일 뒤 +5%↑ 반등 → R+2 종가 매수', lambda d: DN(d) & (d.c2 >= .05), 'c2'),
                        ('급락 ≤−20% & 서프+', lambda d: (d.dayret <= -.2) & (d.surprise > 0)), ('급락 ≤−20% & 서프−', lambda d: (d.dayret <= -.2) & (d.surprise < 0))])
# 진입 시점 비교 (급등·급락): R 종가 / R+1 시가 / R+1 종가
P('\n■ 진입 시점 비교 — 20일 뒤 수익률 평균 / 최고가 +30% 확률 (R 종가 · R+1 시가 · R+1 종가)')
for pn, (a, b) in PER.items():
    sub = E[(E.year >= a) & (E.year <= b)]
    for name, cond in (('급등 ≥+10%', UP), ('급등 ≥+20%', lambda d: d.dayret >= .2), ('급락 ≤−10%', DN), ('급락 ≤−20%', lambda d: d.dayret <= -.2)):
        g = sub[cond(sub) & sub.o1_c20.notna()]
        if len(g) < 30: continue
        c1_20 = (1 + g.c20) / (1 + g.c1) - 1; c1_max = (1 + g.max20) / (1 + g.c1) - 1
        P(f'  [{pn}] {name:<12} N={len(g):>5}  R종가 {g.c20.mean() * 100:+5.1f}% / {(g.max20 >= .3).mean() * 100:4.1f}%   R+1시가 {g.o1_c20.mean() * 100:+5.1f}% / {(g.o1_max20 >= .3).mean() * 100:4.1f}%   R+1종가 {c1_20.mean() * 100:+5.1f}% / {(c1_max >= .3).mean() * 100:4.1f}%   (밤 {g.o1.mean() * 100:+.2f}%, 다음날 {g.c1.mean() * 100:+.2f}%)')
# 발표 전 매수
P('\n■ 발표 전 5일 종가 매수 → 발표일 종가 (thru) / 발표 뒤 20일 (thru20) · 발표 직전 5일 흐름 (pre5)')
for pn, (a, b) in PER.items():
    sub = E[(E.year >= a) & (E.year <= b) & E.thru20.notna()]
    if len(sub) < 30: continue
    P(f'  [{pn}] N={len(sub):>6}  pre5 {sub.pre5.mean() * 100:+.2f}%  thru {sub.thru.mean() * 100:+.2f}% (중앙 {sub.thru.median() * 100:+.2f}%)  thru20 {sub.thru20.mean() * 100:+.2f}%  thru +20%↑ {(sub.thru >= .2).mean() * 100:.1f}%  −20%↓ {(sub.thru <= -.2).mean() * 100:.1f}%')
    for name, cond in (('시총 100위 안', lambda d: d.caprank <= 100), ('시총 501~', lambda d: d.caprank > 500), ('발표 전 60일 +30%↑', lambda d: d.ret60 >= .3), ('60일 변동성 ≥4%/일', lambda d: d.vol60 >= .04)):
        g = sub[cond(sub)]
        if len(g) >= 30: P(f'      {name:<18} N={len(g):>5}  thru {g.thru.mean() * 100:+.2f}%  thru20 {g.thru20.mean() * 100:+.2f}%  +20%↑ {(g.thru >= .2).mean() * 100:.1f}%  −20%↓ {(g.thru <= -.2).mean() * 100:.1f}%')
# 40·60일 드리프트 (PEAD) — 시장 조정 평균과 t
P('\n■ 20·40·60일 드리프트 (조정 평균 % · t) — 전통적 PEAD 점검. SPY대비 / 대상 동일가중 평균 대비')
for pn, (a, b) in PER.items():
    sub = E[(E.year >= a) & (E.year <= b)]
    if len(sub) < 30: continue
    P(f' [{pn}]')
    for name, cond in (('전체', lambda d: d.dayret.notna()), ('서프 ≥+30%', lambda d: d.surprise >= 30), ('서프 ≤−20%', lambda d: d.surprise <= -20),
                       ('급등 ≥+10%', UP), ('급등 ≥+10% & 서프+', lambda d: UP(d) & (d.surprise > 0)), ('급등 ≥+20%', lambda d: d.dayret >= .2),
                       ('급락 ≤−10%', DN), ('급락 ≤−10% & 서프−', lambda d: DN(d) & (d.surprise < 0)), ('급락 ≤−20%', lambda d: d.dayret <= -.2)):
        g = sub[cond(sub) & sub.c60.notna() & sub.spy60.notna()]
        if len(g) < 30: continue
        cells = []
        for k in (20, 40, 60):
            a1 = g[f'c{k}'] - g[f'spy{k}']; a2 = g[f'c{k}'] - g[f'uni{k}']
            t1 = a1.mean() / (a1.std() / np.sqrt(len(g))) if a1.std() > 0 else np.nan; t2 = a2.mean() / (a2.std() / np.sqrt(len(g))) if a2.std() > 0 else np.nan
            cells.append(f'{k}일 SPY대비 {a1.mean() * 100:+5.1f} (t {t1:4.1f}) 대상대비 {a2.mean() * 100:+5.1f} (t {t2:4.1f})')
        P(f'  {name:<22} N={len(g):>5}  ' + '  '.join(cells))
# 연도별 (급등·급락) — 생존 편향·시대 변화 점검
P('\n■ 연도별: N · 평균20 · 조정20 · 최고가+30%↑ · −20%↓  (급등 ≥+10% | 급등 ≥+20% | 급락 ≤−10% | 급락 ≤−20%)')
for y in sorted(E.year.unique()):
    sub = E[E.year == y]; cells = []
    for cond in (UP, lambda d: d.dayret >= .2, DN, lambda d: d.dayret <= -.2):
        g = sub[cond(sub)]
        cells.append(f'N={len(g):>4} {g.c20.mean() * 100:+5.1f} {g.adj20.mean() * 100:+5.1f} {(g.max20 >= .3).mean() * 100:4.1f} {(g.c20 <= -.2).mean() * 100:4.1f}' if len(g) >= 15 else f'N={len(g):>4} {"표본 부족":^22}')
    P(f'  {y}  ' + ' | '.join(cells))
P('\n■ 연도별 (세부 조건): N · 평균20 · 조정20 · 최고가+30%↑ · −20%↓  (급등 & 변동성≥4% | 급락 & 60일 −20%↓ | 급락 & 52주 −40% 밖)  ‖ 발표 전 5일 매수→발표일 (60일 +30%↑ 종목): N · thru · +20%↑ · −20%↓')
for y in sorted(E.year.unique()):
    sub = E[E.year == y]; cells = []
    for cond in (lambda d: UP(d) & (d.vol60 >= .04), lambda d: DN(d) & (d.ret60 <= -.2), lambda d: DN(d) & (d.hi250 < -.4)):
        g = sub[cond(sub)]
        cells.append(f'N={len(g):>4} {g.c20.mean() * 100:+5.1f} {g.adj20.mean() * 100:+5.1f} {(g.max20 >= .3).mean() * 100:4.1f} {(g.c20 <= -.2).mean() * 100:4.1f}' if len(g) >= 15 else f'N={len(g):>4} {"표본 부족":^22}')
    g = sub[(sub.ret60 >= .3) & sub.thru.notna()]
    cells.append(f'N={len(g):>4} thru {g.thru.mean() * 100:+5.1f} {(g.thru >= .2).mean() * 100:4.1f} {(g.thru <= -.2).mean() * 100:4.1f}' if len(g) >= 15 else f'N={len(g):>4} 표본 부족')
    P(f'  {y}  ' + ' | '.join(cells))
P('\n■ 섹터별 급등(≥+10%) 2020~2025: N · 평균20 · 조정20 · 최고가+30%↑ · −20%↓')
sub = E[(E.year >= 2020) & (E.year <= 2025) & UP(E)]
for sec, g in sorted(sub.groupby('sector'), key=lambda kv: -len(kv[1])):
    if len(g) >= 30: P(f'  {sec:<24} N={len(g):>4} {g.c20.mean() * 100:+5.1f} {g.adj20.mean() * 100:+5.1f} {(g.max20 >= .3).mean() * 100:4.1f} {(g.c20 <= -.2).mean() * 100:4.1f}')
P('\n■ 반응일 품질: |dayret| 중앙값 · 거래 배수(vr) 중앙값 · vr≥2 비율 · |dayret|≥5% 비율 (연도별)')
for y in sorted(E.year.unique()):
    sub = E[E.year == y]
    P(f'  {y}  N={len(sub):>5}  |dayret| 중앙 {sub.absret.median() * 100:4.1f}%  vr 중앙 {sub.vr.median():4.1f}  vr≥2 {(sub.vr >= 2).mean() * 100:4.0f}%  |dayret|≥5% {(sub.absret >= .05).mean() * 100:4.0f}%  장전 {sub.bmo.mean() * 100:3.0f}%')
# 달력 월별 사건 수
P('\n■ 달력 월별 사건 수 (2020~2025 평균): 전체 / 급등 ≥+10% / 급락 ≤−10%')
sub = E[(E.year >= 2020) & (E.year <= 2025)]; m = sub.rdate.str[4:6]
P('  ' + ' '.join(f'{mm}월 {len(sub[m == mm]) / 6:4.0f}/{(UP(sub) & (m == mm)).sum() / 6:3.0f}/{(DN(sub) & (m == mm)).sum() / 6:3.0f}' for mm in sorted(m.unique())))
open('backtest/results/us_earnings.md', 'w', encoding='utf-8').write('# 미국 실적 발표 반응 뒤 20일 (backtest/lab/us/earn_analyze.py)\n\n```\n' + '\n'.join(lines) + '\n```\n')
