import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2 as P2, panic2_common as C, panic2_feat as PF, panic_sim2 as S2, engine as E
X=P2.Ctx(); F=X.F; T=X.T; CAP=1e8
c=F.c.astype(float)
def sh(a,k): o=np.full_like(a,np.nan,dtype=float); o[k:]=a[:-k]; return o
trd=F.tradable; v20p=sh(F.val20.astype(float),1)
base=PF.lists_of(PF.top_lists(np.where(trd&(F.caprank<=200)&(F.ret60>0)&(F.max20<0.10)&np.isfinite(F.hi250),c/F.hi250,np.nan)))
U=trd&(F.cr1<=1000)&(v20p>=2e9)&~F.limup
filt=(np.nan_to_num(F.ibs,nan=1)<0.7)&(np.nan_to_num(F.vr,nan=9)<3)
L={'5일낮은순':PF.lists_of(PF.top_lists(np.where(U,-F.ret5.astype(float),np.nan))),
   '5일낮은순+필터':PF.lists_of(PF.top_lists(np.where(U&filt,-F.ret5.astype(float),np.nan)))}
st={'11~19':list(range(C.didx(F,'20110103'),C.didx(F,'20200101'),2)),'20~':list(range(C.didx(F,'20200101'),T-E.MONTH+1))}
yrs=np.array([F.dates[s][:4] for s in st['20~']])
R0=S2.Hybrid('R0',base,X.surge)
r0={(k,p):np.array([S2.run_month(X.m,R0,i,k,CAP)[0] for i in s]) for k in (2,4,5) for p,s in st.items()}
f=lambda a:f"{a.mean()*100:+.2f}/{(a<=-0.1).mean()*100:.0f}%/{(a>=0.1).mean()*100:.0f}%"
print("기준|고르는법|진입|보유|종목수| Δ11~19 | Δ20~ | 26제외 | 11~19 평균/−10%/+10% | 20~ 평균/−10%/+10% | 연도")
for thr in (-0.03,-0.04,-0.038,-0.042):
  sig=F.x<=thr
  for ln,lst in L.items():
    for entry in ('close','open'):
      for H in (5,):
        hy=S2.Hybrid('x',base,X.surge,mp_sig=sig,mp_rank_day=X.ident,mp_anchor_day=X.ident,mp_top=lst,mp_entry=entry,mp_exit=S2.Exit(hold=H),mp_episode=X.ep_ids(sig))
        for k in (2,4,5):
          if abs(thr) in (0.038,0.042) and k!=5: continue
          o={p:np.array([S2.run_month(X.m,hy,i,k,CAP)[0] for i in s]) for p,s in st.items()}
          d1=o['11~19']-r0[(k,'11~19')]; d2=o['20~']-r0[(k,'20~')]
          print(f"{thr*100:.1f}%|{ln}|{'당일종가' if entry=='close' else '다음날시가'}|{H}일|{k}| {d1.mean()*100:+.2f} | {d2.mean()*100:+.2f} | {d2[yrs!='2026'].mean()*100:+.2f} | {f(o['11~19'])} | {f(o['20~'])} |",' '.join(f"{y[2:]}:{d2[yrs==y].mean()*100:+.1f}" for y in sorted(set(yrs))),flush=True)
