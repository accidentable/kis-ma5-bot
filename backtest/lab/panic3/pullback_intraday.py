"""
장중 '+3% 이상 오른 종목이 고점 대비 −3% 빠지면 매수' — 일봉으로 근사 (코스피 시총 100 + 코스닥 시총 150, 전날 순위)
일봉은 장중 순서를 모르므로 두 가지로 본다:
  A (확실한 경우만)  고가 ≥ 전날종가×1.03 이고 종가 ≤ 고가×0.97 → 고점 뒤에 −3% 빠진 게 확실. 매수가 = 고가×0.97
  B (넓게)          고가 ≥ 전날종가×1.03 이고 저가 ≤ 고가×0.97 → 고가가 먼저 왔다고 가정. 매수가 = 고가×0.97
매도 (다음 날부터): 시가/장중 +TP% 지정가 · −SL% 손절(+0.2% 슬리피지, 같은 날 둘 다 닿으면 손절로) · 최대 H 거래일째 종가
비용: 수수료 0.015%×2 + 매도세 0.2% + 슬리피지 0.05%×2
포트폴리오: 1억, 최대 5종목 (한 종목 2천만 원), 그날 조건 맞은 종목 중 상승폭(고가/전날종가) 큰 순으로 빈 슬롯만큼
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C
F=PF.load(); T=F.T
o,h,l,c=(getattr(F,k).astype(float) for k in 'ohlc')
pc=np.vstack([np.full((1,F.N),np.nan),c[:-1]])
U=F.traded&(F.raw>=1000)&(((F.lab==1)&(F.kprank1<=100))|((F.lab==2)&(F.kqrank1<=150)))
COST=0.00015*2+0.002+0.0005*2
def trades(mode,TP,SL,H,a,b):
    up=h>=pc*1.03
    trig=U&up&((c<=h*0.97) if mode=='A' else (l<=h*0.97))
    out=[]   # (t, j, 진입가, 청산일, 수익, 순위키)
    for t in range(a,b):
        js=np.where(trig[t])[0]
        for j in js:
            e=h[t,j]*0.97; tp=e*(1+TP); sl=e*(1-SL) if SL else -1
            r=None
            for k in range(1,H+1):
                u=t+k
                if u>=T: break
                if not F.traded[u,j]: continue
                if SL and o[u,j]<=sl: r=o[u,j]/e-1-0.002; break
                if o[u,j]>=tp: r=o[u,j]/e-1; break
                if SL and l[u,j]<=sl: r=sl/e-1-0.002; break
                if h[u,j]>=tp: r=TP; break
                if k==H: r=c[u,j]/e-1; break
            if r is None: continue
            out.append((t,j,u,r-COST,h[t,j]/pc[t,j]))
    return out
def month_sim(tr,starts,slots=5):
    by={}
    for t,j,u,r,g in tr: by.setdefault(t,[]).append((g,j,u,r))
    rets=[];nt=[]
    for s in starts:
        e=s+20; free=[s]*slots; acc=0.0; n=0
        for t in range(s,e):
            cands=sorted(by.get(t,[]),reverse=True)
            for g,j,u,r in cands:
                k=[i for i in range(slots) if free[i]<=t]
                if not k: break
                if u>e:
                    # 창 끝에 아직 보유: 마지막 날 종가로 평가
                    r=c[e,j]/(h[t,j]*0.97)-1-COST if np.isfinite(c[e,j]) else r
                    u=e
                free[k[0]]=u+1; acc+=r/slots; n+=1
        rets.append(acc); nt.append(n)
    return np.array(rets),np.array(nt)
P={'2011~2019':(C.didx(F,'20110101'),C.didx(F,'20200101')),'2020~':(C.didx(F,'20200101'),T-26)}
print('모드 TP SL H | 기간: 건수/주, 승률, 건당 순수익 | 한 달(5슬롯): 평균 / 중앙 / −10%↓ / +10%↑ / 거래수')
for mode in ('A','B'):
    for TP,SL,H in ((0.025,0.04,5),(0.03,0.04,5),(0.025,0,5),(0.05,0.05,5),(0.03,0.03,3)):
        row=[f"{mode} +{TP*100:g}% −{SL*100:g}% {H}일"]
        for pn,(a,b) in P.items():
            tr=trades(mode,TP,SL,H,a,b)
            r=np.array([x[3] for x in tr]); wk=len(tr)/((b-a)/5)
            st=list(range(a,b-21,2 if pn=='2011~2019' else 1))
            mr,nt=month_sim(tr,st)
            row.append(f"[{pn}] {wk:.1f}건/주 승률 {np.mean(r>0)*100:.0f}% 건당 {r.mean()*100:+.2f}% | 월 {mr.mean()*100:+.2f}% / {np.median(mr)*100:+.2f}% / {(mr<=-0.1).mean()*100:.0f}% / {(mr>=0.1).mean()*100:.0f}% / {nt.mean():.0f}건")
        print('  '.join(row),flush=True)
