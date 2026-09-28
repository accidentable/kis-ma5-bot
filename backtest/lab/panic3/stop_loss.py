"""패닉 모드 손절 비교 (1억 · 5종목 · x≤−4% · 5일 낙폭순+필터 · 5일 보유). 종목별 장중 손절 −X% (체결 +0.2% 슬리피지)."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2 as P2, panic2_common as C, panic2_feat as PF, panic_sim2 as S2, engine as E
X=P2.Ctx(); F=X.F; T=X.T; CAP=1e8
c=F.c.astype(float)
def sh(a,k): o=np.full_like(a,np.nan,dtype=float); o[k:]=a[:-k]; return o
trd=F.tradable; v20p=sh(F.val20.astype(float),1)
base=PF.lists_of(PF.top_lists(np.where(trd&(F.caprank<=200)&(F.ret60>0)&(F.max20<0.10)&np.isfinite(F.hi250),c/F.hi250,np.nan)))
U=trd&(F.cr1<=1000)&(v20p>=2e9)&~F.limup&(np.nan_to_num(F.ibs,nan=1)<0.7)&(np.nan_to_num(F.vr,nan=9)<3)
L=PF.lists_of(PF.top_lists(np.where(U,-F.ret5.astype(float),np.nan)))
sig=F.x<=-0.04
st={'11~19':list(range(C.didx(F,'20110103'),C.didx(F,'20200101'),2)),'20~':list(range(C.didx(F,'20200101'),T-E.MONTH+1))}
yrs=np.array([F.dates[s][:4] for s in st['20~']])
R0=S2.Hybrid('R0',base,X.surge)
r0={p:np.array([S2.run_month(X.m,R0,i,5,CAP)[0] for i in s]) for p,s in st.items()}
for name,ex in [('손절 없음',S2.Exit(hold=5)),('−7%',S2.Exit(hold=5,stop=0.07)),('−10%',S2.Exit(hold=5,stop=0.10)),
                ('−15%',S2.Exit(hold=5,stop=0.15)),('−20%',S2.Exit(hold=5,stop=0.20)),
                ('시장 −6% 추가하락 시 전량',S2.Exit(hold=5,mkt_stop=0.06)),('+15% 익절',S2.Exit(hold=5,tp=0.15)),
                ('−10% 손절 +15% 익절',S2.Exit(hold=5,stop=0.10,tp=0.15))]:
    hy=S2.Hybrid('x',base,X.surge,mp_sig=sig,mp_rank_day=X.ident,mp_anchor_day=X.ident,mp_top=L,mp_exit=ex,mp_episode=X.ep_ids(sig))
    o={p:np.array([S2.run_month(X.m,hy,i,5,CAP)[0] for i in s]) for p,s in st.items()}
    f=lambda a:f"{a.mean()*100:+.2f}/{(a<=-0.1).mean()*100:.0f}%/{a.min()*100:.0f}%"
    print(f"{name:22s} | Δ11~19 {(o['11~19']-r0['11~19']).mean()*100:+.2f} | Δ20~ {(o['20~']-r0['20~']).mean()*100:+.2f} | 평균/−10%↓/최악 11~19 {f(o['11~19'])} · 20~ {f(o['20~'])}", flush=True)
