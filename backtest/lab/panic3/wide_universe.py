import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2 as P2, panic2_common as C, panic2_feat as PF, panic_sim2 as S2, engine as E
X=P2.Ctx(); F=X.F; T=X.T
ok=F.ok; cr1=F.cr1; lab=F.lab; ret1=F.ret1.astype(float)
v20p=np.vstack([np.full((1,F.N),np.nan),F.val20[:-1]])
c=F.c.astype(float)
hi10=np.maximum.reduce([np.vstack([np.full((k,F.N),np.nan),c[:-k]]) if k else c for k in range(11)])
ep=c/hi10-1
U={
 'U200 대형(기준)': ok&(cr1<=200),
 'U500': ok&(cr1<=500),
 'U1000': ok&(cr1<=1000),
 'KOSPI+KOSDAQ 유동(거래대금50억↑)': ok&(v20p>=5e9),
 'KOSDAQ 시총300': ok&(lab==2)&(F.kqrank1<=300),
 'KOSDAQ 유동(50억↑)': ok&(lab==2)&(v20p>=5e9),
}
nolim=~F.limdown&(ret1>-0.25)
T0=X.trig('T0'); T2=X.trig('T2')
periods={'2011~2019':(C.didx(F,'20110101'),C.didx(F,'20200101')),'2020~':(C.didx(F,'20200101'),T-E.MONTH+1)}
st={k:list(range(a,b)) for k,(a,b) in periods.items()}
r0={k:P2.run_cfg(X,P2.Cfg('R0','R','',X.R0),s,control=False)['ret'] for k,s in st.items()}
yrs=np.array([F.dates[s][:4] for s in st['2020~']])
print('R0', {k:round(v.mean()*100,2) for k,v in r0.items()})
def run(name,sig,score,H=5):
    top=PF.lists_of(PF.top_lists(score))
    hy=S2.Hybrid(name,X.base,X.surge,mp_sig=sig,mp_rank_day=X.ident,mp_anchor_day=X.ident,mp_top=top,mp_exit=S2.Exit(hold=H),mp_episode=X.ep_ids(sig))
    out={}
    for k,s in st.items():
        rr=np.array([S2.run_month(X.m,hy,i,2)[0] for i in s]); d=rr-r0[k]
        out[k]=(d.mean(), (rr<=-0.1).mean())
        if k=='2020~':
            yd={y:d[yrs==y].mean() for y in sorted(set(yrs))}
            ex26=d[yrs!='2026'].mean()
    # 사건 단위 5일 순수익 (2020~)
    a=periods['2020~'][0]; days=np.where(sig)[0]; days=days[days>=a]
    vals=[]
    for t in days:
        jj=top[t][:2]
        if len(jj): vals.append(np.nanmean(C.fwd(F,np.full(len(jj),t),np.array(jj),'open',H)[0]))
    return out,yd,ex26,np.nanmean(vals)
for tn,sig in (('T0',T0),('T2',T2)):
  for un,m in U.items():
    for sn,sc in (('낙폭순',np.where(m,-ret1,np.nan)),('낙폭순·하한가제외',np.where(m&nolim,-ret1,np.nan)),('10일고점대비',np.where(m,-ep,np.nan))):
      for H in (5,10):
        if H==10 and sn!='낙폭순·하한가제외': continue
        o,yd,ex,ev=run(un,sig,sc,H)
        print(f"{tn}|{un}|{sn}|{H}일 | 11~19 Δ {o['2011~2019'][0]*100:+.2f} | 20~ Δ {o['2020~'][0]*100:+.2f} p10d {o['2020~'][1]*100:.0f}% | 26제외 {ex*100:+.2f} | 사건5일순 {ev*100:+.1f}% |", ' '.join(f"{y[2:]}:{v*100:+.1f}" for y,v in yd.items()), flush=True)
