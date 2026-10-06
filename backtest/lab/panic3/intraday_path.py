import numpy as np, warnings; warnings.filterwarnings('ignore')
from backtest.lab import panic2_feat as PF, panic2_common as C
F=PF.load(); T=F.T
c=F.c.astype(float); o=F.o.astype(float); h=F.h.astype(float); l=F.l.astype(float)
def sh(a,k): x=np.full_like(a,np.nan,dtype=float); x[k:]=a[:-k]; return x
trd=F.tradable; v20p=sh(F.val20.astype(float),1)
U=trd&(F.cr1<=1000)&(v20p>=2e9)&~F.limup
filt=(np.nan_to_num(F.ibs,nan=1)<0.7)&(np.nan_to_num(F.vr,nan=9)<3)
sc=np.where(U&filt,-F.ret5.astype(float),np.nan)
top=PF.top_lists(sc)
days=np.where(F.x<=-0.04)[0]; days=days[(days>=C.didx(F,'20110101'))&(days<T-7)]
pc=lambda t,j: c[t-1,j]
R={}
def add(k,v): R.setdefault(k,[]).append(v)
for t in days:
    jj=top[t][:5]; jj=jj[jj>=0]
    if not len(jj): continue
    # 폭락 당일: 시가 갭, 저가, 종가 (전일 종가 대비)
    add('당일 시가',np.nanmean(o[t,jj]/c[t-1,jj]-1)); add('당일 저가',np.nanmean(l[t,jj]/c[t-1,jj]-1)); add('당일 종가',np.nanmean(c[t,jj]/c[t-1,jj]-1))
    add('당일 저가→종가 반등',np.nanmean(c[t,jj]/l[t,jj]-1))
    # 다음 날: 당일 종가 대비
    add('다음날 시가',np.nanmean(o[t+1,jj]/c[t,jj]-1)); add('다음날 저가',np.nanmean(l[t+1,jj]/c[t,jj]-1)); add('다음날 고가',np.nanmean(h[t+1,jj]/c[t,jj]-1)); add('다음날 종가',np.nanmean(c[t+1,jj]/c[t,jj]-1))
    ex=c[t+5,jj]   # 공통 청산: t+5 종가 (다음 날 시가 진입 FIX5 와 같은 날)
    def net(px): return np.nanmean(ex/px-1)-0.0035
    add('A 당일 시가 매수',net(o[t,jj]))
    add('B 당일 종가 매수',net(c[t,jj]))
    add('C 다음날 시가 매수',net(o[t+1,jj]))
    add('D 다음날 종가 매수',net(c[t+1,jj]))
    for k in (0.02,0.04,0.06):
        lim=o[t+1,jj]*(1-k); fill=l[t+1,jj]<=lim
        px=np.where(fill,lim,c[t+1,jj])      # 안 걸리면 종가에 산다
        add(f'E 다음날 시가−{k*100:.0f}% 지정가 (안 걸리면 종가)',net(px)); add(f'  └ 체결률 −{k*100:.0f}%',fill.mean())
    add('F 다음날 저가 (신만 아는 값)',net(l[t+1,jj]))
print(f"x≤−4% 날 {len(days)}일, 5일 낮은순+필터 상위 5종목 평균 (2011~2026)")
for k,v in R.items(): print(f"  {k:40s} {np.nanmean(v)*100:+.2f}%   (중앙 {np.nanmedian(v)*100:+.2f}%)")
