"""
장중 '−3% 이상 빠졌던 종목이 저점 대비 +3% 반등하면 매수' — 일봉 근사 (코스피 시총 100 + 코스닥 시총 150)
  A (확실한 경우만)  저가 ≤ 전날종가×0.97 이고 종가 ≥ 저가×1.03 → 저점 뒤 +3% 반등이 확실. 매수가 = 저가×1.03
  B (넓게)          저가 ≤ 전날종가×0.97 이고 고가 ≥ 저가×1.03 → 저가가 먼저 왔다고 가정. 매수가 = 저가×1.03
  C (종가 매수)      A 와 같은 날 종가에 산다 (장 막판에 반등 확인 후 매수)
매도 · 비용 · 포트폴리오는 pullback_intraday.py 와 같다. 시장 상황별(그날 시장 평균)로도 나눠 본다.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C
F=PF.load(); T=F.T
o,h,l,c=(getattr(F,k).astype(float) for k in 'ohlc')
pc=np.vstack([np.full((1,F.N),np.nan),c[:-1]])
U=F.traded&(F.raw>=1000)&(((F.lab==1)&(F.kprank1<=100))|((F.lab==2)&(F.kqrank1<=150)))
COST=0.00015*2+0.002+0.0005*2
def trades(mode,TP,SL,H,a,b,mkt=None):
    dn=l<=pc*0.97
    trig=U&dn&((h>=l*1.03) if mode=='B' else (c>=l*1.03))
    if mkt is not None: trig&=mkt[:,None]
    out=[]
    for t in range(a,b):
        for j in np.where(trig[t])[0]:
            e=c[t,j] if mode=='C' else l[t,j]*1.03
            tp=e*(1+TP); sl=e*(1-SL) if SL else -1; r=None
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
            out.append((t,j,u,r-COST,pc[t,j]/l[t,j],e))
    return out
def month_sim(tr,starts,slots=5):
    by={}
    for t,j,u,r,g,e in tr: by.setdefault(t,[]).append((g,j,u,r,e))
    rets=[];nt=[]
    for s in starts:
        en=s+20; free=[s]*slots; acc=0.0; n=0
        for t in range(s,en):
            for g,j,u,r,e in sorted(by.get(t,[]),reverse=True):
                k=[i for i in range(slots) if free[i]<=t]
                if not k: break
                if u>en:
                    r=c[en,j]/e-1-COST if np.isfinite(c[en,j]) else r; u=en
                free[k[0]]=u+1; acc+=r/slots; n+=1
        rets.append(acc); nt.append(n)
    return np.array(rets),np.array(nt)
P={'2011~2019':(C.didx(F,'20110101'),C.didx(F,'20200101')),'2020~':(C.didx(F,'20200101'),T-26)}
x=F.x
print('모드 TP SL H 시장 | 기간: 건수/주, 승률, 건당 순수익 | 한 달(5슬롯, 낙폭 큰 순): 평균/중앙/−10%↓/+10%↑/거래수')
for mode in ('A','C','B'):
  for mk_name,mk in (('전체',None),('시장 상승일(0%↑)',x>=0),('시장 하락일(−1%↓)',x<=-0.01)):
    for TP,SL,H in ((0.025,0.04,5),(0.03,0,5),(0.05,0.05,5)):
        if mk is not None and (TP,SL)!=(0.025,0.04): continue
        row=[f"{mode} +{TP*100:g}% −{SL*100:g}% {H}일 {mk_name}"]
        for pn,(a,b) in P.items():
            tr=trades(mode,TP,SL,H,a,b,mk)
            if not tr: row.append(f'[{pn}] 없음'); continue
            r=np.array([q[3] for q in tr])
            mr,nt=month_sim(tr,list(range(a,b-21,2 if pn=='2011~2019' else 1)))
            row.append(f"[{pn}] {len(tr)/((b-a)/5):.0f}건/주 승률 {np.mean(r>0)*100:.0f}% 건당 {r.mean()*100:+.2f}% | 월 {mr.mean()*100:+.2f}%/{np.median(mr)*100:+.2f}%/{(mr<=-0.1).mean()*100:.0f}%/{(mr>=0.1).mean()*100:.0f}%/{nt.mean():.0f}건")
        print('  '.join(row),flush=True)
