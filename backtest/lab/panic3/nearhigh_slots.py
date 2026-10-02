"""52주 신고가 근접 전략 — 1억 · 가격 상한 없음 · 종목 수 1~10 · 교체 주기 5/10/21일 · 유니버스 시총 200/500 (패닉 모드 없음, 있음)."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2 as P2, panic2_common as C, panic2_feat as PF, panic_sim2 as S2, engine as E
X=P2.Ctx(); F=X.F; T=X.T; CAP=1e8
c=F.c.astype(float)
def sh(a,k): o=np.full_like(a,np.nan,dtype=float); o[k:]=a[:-k]; return o
trd=F.tradable; v20p=sh(F.val20.astype(float),1)
def base_list(top):
    return PF.lists_of(PF.top_lists(np.where(trd&(F.caprank<=top)&(F.ret60>0)&(F.max20<0.10)&np.isfinite(F.hi250),c/F.hi250,np.nan)))
U=trd&(F.cr1<=1000)&(v20p>=2e9)&~F.limup&(np.nan_to_num(F.ibs,nan=1)<0.7)&(np.nan_to_num(F.vr,nan=9)<3)
PL=PF.lists_of(PF.top_lists(np.where(U,-F.ret5.astype(float),np.nan)))
sig=F.x<=-0.04
st={'11~19':list(range(C.didx(F,'20110103'),C.didx(F,'20200101'),2)),'20~':list(range(C.didx(F,'20200101'),T-E.MONTH+1))}
f=lambda a:f"{a.mean()*100:+.2f}/{np.median(a)*100:+.2f}/{(a<=-0.1).mean()*100:.0f}%/{(a>=0.1).mean()*100:.0f}%"
print("유니버스 | 교체 | 종목수 | 패닉 | 2011~19 평균/중앙/−10%↓/+10%↑ | 2020~ 평균/중앙/−10%↓/+10%↑ | 한 달 매매금액(2020~)")
for top in (200,500):
    B=base_list(top)
    for rot in (5,10,21):
        for k in (1,2,3,5,7,10):
            for pan in (False,True):
                if pan and rot!=21: continue
                kw=dict(mp_sig=sig,mp_rank_day=X.ident,mp_anchor_day=X.ident,mp_top=PL,mp_exit=S2.Exit(hold=5),mp_episode=X.ep_ids(sig)) if pan else {}
                hy=S2.Hybrid('x',B,X.surge,base_rot=rot,**kw)
                out=[];turn=[]
                for pn,s in st.items():
                    rr=[]
                    for i in s:
                        r,info=S2.run_month(X.m,hy,i,k,CAP); rr.append(r)
                        if pn=='20~': turn.append(info['entries'])
                    out.append(f(np.array(rr)))
                print(f"시총{top} | {rot}일 | {k} | {'O' if pan else '-'} | {out[0]} | {out[1]} | 매수 {np.mean(turn):.1f}건", flush=True)
