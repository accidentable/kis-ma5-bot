import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C
F=PF.load(); T=F.T
c=F.c.astype(float)
def sh(a,k): o=np.full_like(a,np.nan,dtype=float); o[k:]=a[:-k]; return o
hi10=np.fmax.reduce([sh(c,k) if k else c for k in range(11)])
U=F.ok&(F.cr1<=1000)
feat={
 '당일 낙폭 (ret1)': -F.ret1,
 '10일 고점 대비 낙폭': -(c/hi10-1),
 '5일 수익률 낮은 순': -F.ret5,
 '20일 수익률 낮은 순': -F.ret20,
 '60일 수익률 높은 순 (추세주)': sh(F.ret60.astype(float),1),
 '250일 신고가 근접 (c/hi250)': c/F.hi250,
 '잔차 z 낮은 순 (시장 대비 과하게 빠짐)': -F.z,
 '베타 높은 순': F.bstar,
 '변동성 높은 순 (vol60)': sh(F.vol60.astype(float),1),
 '거래대금 급증 (vr)': F.vr,
 '종가 위치 (IBS 높은=저가서 회복)': F.ibs,
 '시가 갭 하락 큰 순': -F.gap,
 '장중 하락 큰 순 (intra)': -F.intra,
 '시총 작은 순': F.cr1,
 '20일선 이격 (ATR) 낮은 순': -F.dd_atr,
 'RSI2 낮은 순': -F.rsi2,
 '최근 20일 최대 급등 (max20) 큰 순': sh(F.max20.astype(float),1),
 '회전율 높은 순': F.turn1,
 '코스닥 (1) vs 코스피': (F.lab==2).astype(float),
}
x=F.x
buckets={'−2~−3%':(x<=-0.02)&(x>-0.03),'−3~−4%':(x<=-0.03)&(x>-0.04),'≤−4% (참고)':(x<=-0.04)}
per={'2011~2019':(C.didx(F,'20110101'),C.didx(F,'20200101')),'2020~':(C.didx(F,'20200101'),T-11)}
fw={h:{} for h in (5,10)}
def fwd_day(t,jj,h):
    return C.fwd(F,np.full(len(jj),t),jj,'open',h)[0]
res={}
for bn,bm in buckets.items():
    for pn,(a,b) in per.items():
        days=[t for t in np.where(bm)[0] if a<=t<b]
        rec={h:{f:[] for f in feat} for h in (5,10)}; base={h:[] for h in (5,10)}; top2={h:{f:[] for f in feat} for h in (5,10)}
        for t in days:
            jj=np.where(U[t])[0]
            if len(jj)<100: continue
            for h in (5,10):
                r=fwd_day(t,jj,h); ok=np.isfinite(r)
                if ok.sum()<100: continue
                ew=r[ok].mean(); base[h].append(ew)
                for fn,fa in feat.items():
                    v=np.asarray(fa[t,jj],float); g=ok&np.isfinite(v)
                    if g.sum()<50: rec[h][fn].append(np.nan); top2[h][fn].append(np.nan); continue
                    vv,rr=v[g],r[g]; q=np.quantile(vv,0.9)
                    rec[h][fn].append(rr[vv>=q].mean()-ew)
                    o=np.argsort(-vv,kind='stable')[:2]; top2[h][fn].append(rr[o].mean())
        res[(bn,pn)]=(len(base[5]),base,rec,top2)
for bn in buckets:
    print(f"\n### 시장 {bn}  — 시총1000위, 다음 날 시가 매수, 비용 0.35% 뺀 순수익")
    for pn in per:
        n,base,rec,top2=res[(bn,pn)]
        print(f"  [{pn}] 일수 {n} | 전체 평균(EW) 5일 {np.mean(base[5])*100:+.2f}% 10일 {np.mean(base[10])*100:+.2f}%")
    print("  특징 | 상위10% − EW: 5일(11~19 / 20~) | 10일(11~19 / 20~) | 1·2위 두 종목 5일(11~19/20~) | 10일(11~19/20~)")
    rows=[]
    for fn in feat:
        vals=[]
        for h in (5,10):
            for pn in per:
                a=np.array(res[(bn,pn)][2][h][fn]); vals.append((np.nanmean(a), np.nanmean(a)/ (np.nanstd(a)/np.sqrt(np.isfinite(a).sum())+1e-12)))
        t2=[np.nanmean(res[(bn,pn)][3][h][fn]) for h in (5,10) for pn in per]
        rows.append((fn,vals,t2))
    for fn,vals,t2 in sorted(rows,key=lambda r:-(r[1][2][0]+r[1][3][0])):
        print(f"  {fn:32s} | {vals[0][0]*100:+.2f}({vals[0][1]:+.1f}) / {vals[1][0]*100:+.2f}({vals[1][1]:+.1f}) | {vals[2][0]*100:+.2f}({vals[2][1]:+.1f}) / {vals[3][0]*100:+.2f}({vals[3][1]:+.1f}) | {t2[0]*100:+.1f} / {t2[1]*100:+.1f} | {t2[2]*100:+.1f} / {t2[3]*100:+.1f}")
