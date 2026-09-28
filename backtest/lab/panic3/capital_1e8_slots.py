"""모의투자 1억 · 종목 1~5개: 평소 = 현재 봇(가격 상한 없음), 시장 급락일 = 시총 1000위 과매도주로 교체."""
import numpy as np, warnings, pickle; warnings.filterwarnings('ignore')
from backtest.lab import panic2 as P2, panic2_common as C, panic2_feat as PF, panic_sim2 as S2, engine as E
X=P2.Ctx(); F=X.F; T=X.T; CAP=1e8
c=F.c.astype(float)
def sh(a,k): o=np.full_like(a,np.nan,dtype=float); o[k:]=a[:-k]; return o
hi10=np.fmax.reduce([sh(c,k) if k else c for k in range(11)])
trd=F.tradable                                      # 가격 상한 없음 (1억)
v20p=sh(F.val20.astype(float),1)
base_sc=np.where(trd&(F.caprank<=200)&(F.ret60>0)&(F.max20<0.10)&np.isfinite(F.hi250),c/F.hi250,np.nan)
base=PF.lists_of(PF.top_lists(base_sc))
U=trd&(F.cr1<=1000)&(v20p>=2e9)                      # 한 종목 2천만원 이상 사도 되게 20일 거래대금 20억↑
filt=(np.nan_to_num(F.ibs,nan=1)<0.7)&(np.nan_to_num(F.vr,nan=9)<3)
sels={'10일고점대비':-(c/hi10-1),'20일수익률낮은순':-F.ret20.astype(float),'5일수익률낮은순':-F.ret5.astype(float)}
trig={'x≤−3%':F.x<=-0.03,'x≤−4%':F.x<=-0.04}
st={'11~19':list(range(C.didx(F,'20110103'),C.didx(F,'20200101'),2)),'20~':list(range(C.didx(F,'20200101'),T-E.MONTH+1))}
yrs=np.array([F.dates[s][:4] for s in st['20~']])
R0=S2.Hybrid('R0',base,X.surge)
r0={}
for k in range(1,6):
    for pn,s in st.items(): r0[(k,pn)]=np.array([S2.run_month(X.m,R0,i,k,CAP)[0] for i in s])
    print(f"R0 k={k}: 11~19 {r0[(k,'11~19')].mean()*100:+.2f}  20~ {r0[(k,'20~')].mean()*100:+.2f}", flush=True)
lists={}
for sn,sc in sels.items():
    for fn,fm in (('',U),('+필터',U&filt)):
        lists[sn+fn]=PF.lists_of(PF.top_lists(np.where(fm,sc,np.nan)))
rows=[]
for tn,sig in trig.items():
  for ln,L in lists.items():
    for H in (5,10):
      hy=S2.Hybrid('x',base,X.surge,mp_sig=sig,mp_rank_day=X.ident,mp_anchor_day=X.ident,mp_top=L,mp_exit=S2.Exit(hold=H),mp_episode=X.ep_ids(sig))
      for k in range(1,6):
        out={}
        for pn,s in st.items():
            r=np.array([S2.run_month(X.m,hy,i,k,CAP)[0] for i in s]); out[pn]=(r,r-r0[(k,pn)])
        d20=out['20~'][1]
        rows.append(dict(t=tn,sel=ln,H=H,k=k,d1=out['11~19'][1].mean(),d2=d20.mean(),ex=d20[yrs!='2026'].mean(),
            a1=out['11~19'][0],a2=out['20~'][0],yd={y:d20[yrs==y].mean() for y in sorted(set(yrs))}))
  print('done',tn,flush=True)
pickle.dump((rows,{k:v for k,v in r0.items()}),open('/tmp/claude-0/-home-user-kis-ma5-bot/b5be091b-5136-5c7a-b5fc-c4f5261ee9a2/scratchpad/kslots.pkl','wb'))
rows.sort(key=lambda r:-min(r['d1'],r['ex']))
f=lambda a:f"{a.mean()*100:+.2f}/{(a<=-0.1).mean()*100:.0f}%/{(a>=0.1).mean()*100:.0f}%"
print("트리거|고르는법|보유|종목수| Δ11~19 | Δ20~ | Δ20~(26제외) | 11~19 평균/−10%↓/+10%↑ | 20~ 평균/−10%↓/+10%↑ | 연도별Δ(20~)")
for r in rows[:25]:
    print(f"{r['t']}|{r['sel']}|{r['H']}일|{r['k']}| {r['d1']*100:+.2f} | {r['d2']*100:+.2f} | {r['ex']*100:+.2f} | {f(r['a1'])} | {f(r['a2'])} |",' '.join(f"{y[2:]}:{v*100:+.1f}" for y,v in r['yd'].items()))
print('--- 종목수별 평균 (모든 설정)')
for k in range(1,6):
    rr=[r for r in rows if r['k']==k]; print(k,'Δ11~19 %.2f Δ20~ %.2f 26제외 %.2f 양수비율(두기간) %.0f%%'%(np.mean([r['d1'] for r in rr])*100,np.mean([r['d2'] for r in rr])*100,np.mean([r['ex'] for r in rr])*100,np.mean([(r['d1']>0)&(r['ex']>0) for r in rr])*100))
