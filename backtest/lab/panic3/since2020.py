import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2 as P2, panic2_common as C, engine as E
X=P2.Ctx(); F=X.F; cfgs=P2.build_configs(X); by={c.id:c for c in cfgs}
s0=C.didx(F,'20200101'); st=list(range(s0, X.T-E.MONTH+1))
yrs=np.array([F.dates[s][:4] for s in st])
r0=P2.run_cfg(X,by['R0'],st,control=False)['ret']
rows=[]
for c in cfgs:
    if c.track not in ('R','B') or c.id=='R0': continue
    res=P2.run_cfg(X,c,st,r0); d=res['ret']-r0; dc=res['ret']-res['ctrl'] if c.track!='R' else d
    yd={y: d[yrs==y].mean() for y in sorted(set(yrs))}
    rows.append((c.id,c.desc,res['ret'].mean(),d.mean(),np.nanmean(dc),(res['ret']<=-0.1).mean(),yd, res['flag'].mean(), d[res['flag']].mean() if res['flag'].any() else np.nan))
print(f"R0 mean {r0.mean()*100:+.2f} p10d {(r0<=-0.1).mean()*100:.0f}% n {len(st)}", {y: round(r0[yrs==y].mean()*100,2) for y in sorted(set(yrs))})
for r in sorted(rows,key=lambda x:-x[3]):
    print(f"{r[0]} Δ {r[3]*100:+.2f} Δctrl {r[4]*100:+.2f} mean {r[2]*100:+.2f} p10d {r[5]*100:.0f}% entryWin {r[7]*100:.0f}% Δ|entry {r[8]*100:+.2f} |", ' '.join(f"{y[2:]}:{v*100:+.1f}" for y,v in r[6].items()), '|', r[1][:40])
# 사건 단위 T0 K2 / K2EP 5일 Δ (보유 바구니 대비)
T0=np.where(X.trig('T0'))[0]; T0=T0[T0>=s0]
ep=C.episodes_of(T0,10)
for cid in ('R1','B30','B22','B21'):
    dd,dl=P2.b_day_deltas(X,by[cid],T0)
    et=C.ep_table(dd,dl,10)
    print(cid,'events', et['N'], 'mean %.2f median %.2f pos %.0f%% worst %.1f noBest2 %.2f'%(et['mean']*100,et['median']*100,et['pos']*100,et['worst']*100,et['no_best2']*100))
dd,dl=P2.b_day_deltas(X,by['R1'],T0)
for e in range(ep.max()+1):
    m=ep==e; print(F.dates[T0[m][0]], int(m.sum()),'days', f"K2 5d Δ {np.nanmean(dl[m])*100:+.1f}%")
