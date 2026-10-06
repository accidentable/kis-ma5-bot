"""
자주 거래하는 칼날 잡기 후보들 (1억 · 최대 5종목 · 평소 전략 없이 사건만). 주 2회 이상 거래가 조건.
사건 (그날 종가 기준, 시총 1000위 · 20일 거래대금 20억↑ · 상한가 아님):
  E1 시장 상승일(+1%↑) 인데 종목 −5%↓           E2 시장 상승일(0%↑) 인데 잔차 z ≤ −3 (종목 고유 급락)
  E3 매일 5일 낙폭 최대 (+필터: 고가권 마감 · 거래대금 폭증 제외)   E4 RSI2 < 5 & 200일선 위 (코너스식 눌림)
  E5 시장 상승일(+1%↑) 인데 5일 낙폭 −15%↓
1) 사건 단위: 다음 날 시가 매수 → 1·2·3·5일 뒤 종가, 비용 0.35% 뺀 순수익 · 같은 날 전체 평균 대비
2) 대회: 사건 상위부터 빈 슬롯에 매수 (다음 날 시가 / 당일 종가), H 일 보유, 월평균 · −10%↓ 확률 · 주당 매수 횟수
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2 as P2, panic2_common as C, panic2_feat as PF, panic_sim2 as S2, engine as E
X=P2.Ctx(); F=X.F; T=X.T; CAP=1e8
c=F.c.astype(float)
def sh(a,k): o=np.full_like(a,np.nan,dtype=float); o[k:]=a[:-k]; return o
v20p=sh(F.val20.astype(float),1)
U=F.tradable&(F.cr1<=1000)&(v20p>=2e9)&~F.limup
x=F.x[:,None]; r1=F.ret1.astype(float); r5=F.ret5.astype(float)
filt=(np.nan_to_num(F.ibs,nan=1)<0.7)&(np.nan_to_num(F.vr,nan=9)<3)
ma200=np.full_like(c,np.nan); cs=np.nancumsum(np.nan_to_num(c),0); ma200[199:]=(cs[199:]-np.vstack([np.zeros((1,F.N)),cs[:-200]]))/200
ev={
 'E1 시장+1%↑·종목−5%↓': (U&(x>=0.01)&(r1<=-0.05), -r1),
 'E2 시장0%↑·잔차z≤−3': (U&(x>=0)&(F.z<=-3), -F.z.astype(float)),
 'E3 매일 5일낙폭 최대+필터': (U&filt&(r5<=-0.10), -r5),
 'E4 RSI2<5·200일선 위': (U&(F.rsi2<5)&(c>ma200), -F.rsi2.astype(float)),
 'E5 시장+1%↑·5일낙폭−15%↓': (U&(x>=0.01)&(r5<=-0.15), -r5),
}
per={'11~19':(C.didx(F,'20110101'),C.didx(F,'20200101')),'20~':(C.didx(F,'20200101'),T-6)}
print("1) 사건 단위 (각 날 상위 5개, 다음 날 시가 매수, 비용 뺀 순수익 / 같은 날 U 평균 대비)")
for n,(m,sc) in ev.items():
    line=[n]
    for pn,(a,b) in per.items():
        t,j=C.events_from_mask(m,sc,5,a,b)
        days=len(np.unique(t)); wk=days/((b-a)/5)
        cells=[]
        for h in (1,2,3,5):
            net,g=C.fwd(F,t,j,'open',h); ew=C.same_day_ew(F,t,U,'open',h)
            exc=np.array([g[i]-ew.get(int(t[i]),np.nan) for i in range(len(t))])
            cells.append(f"{h}일 {np.nanmean(net)*100:+.2f}({np.nanmean(exc)*100:+.2f})")
        line.append(f"[{pn} 신호일 주 {wk:.1f}회] "+' '.join(cells))
    print(' | '.join(line), flush=True)
st={'11~19':list(range(C.didx(F,'20110103'),C.didx(F,'20200101'),2)),'20~':list(range(C.didx(F,'20200101'),T-E.MONTH+1))}
empty=[np.zeros(0,np.int64)]*T
print("\n2) 대회 (평소 전략 없음, 최대 5종목, 1억) — 월평균/−10%↓/+10%↑ · 주당 매수")
for n,(m,sc) in ev.items():
    L=PF.lists_of(PF.top_lists(np.where(m,sc,np.nan)))
    for entry in ('open','close'):
        for H in (1,2,3,5):
            hy=S2.Hybrid('x',empty,X.surge,sp_top=L,sp_entry=entry,sp_exit=S2.Exit(hold=H),sp_max=5)
            out=[]
            for pn,s in st.items():
                rr=[];nb=[]
                for i in s:
                    r,info=S2.run_month(X.m,hy,i,5,CAP); rr.append(r); nb.append(len(info['sp_entries']))
                rr=np.array(rr); out.append(f"{pn} {rr.mean()*100:+.2f}/{(rr<=-0.1).mean()*100:.0f}%/{(rr>=0.1).mean()*100:.0f}% 주{np.mean(nb)/4.2:.1f}회")
            print(f"{n} | {'당일종가' if entry=='close' else '다음날시가'} | {H}일 | "+' | '.join(out), flush=True)
