# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.9,
 "axes.labelsize":9.6,"xtick.labelsize":8.6,"ytick.labelsize":8.6,"legend.frameon":False,"legend.fontsize":8.2})
R=json.load(open('results/classification/full1.json')); SUB=json.load(open('results/classification/subsets.json'))
n17=np.load('results/classification/null17s.npy')
def ptitle(ax,l,t,dy=1.045,fs=11.5):
    tx=ax.text(0,dy,f"({l})",transform=ax.transAxes,fontsize=fs,fontweight="bold",va="bottom",ha="left")
    ax.figure.canvas.draw()
    bb=tx.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.text(bb.x1+0.020,dy,t,transform=ax.transAxes,fontsize=fs-1.3,va="bottom",ha="left")
fig=plt.figure(figsize=(13.4,4.6))
gs=fig.add_gridspec(1,3,wspace=0.30,left=0.055,right=0.985,top=0.845,bottom=0.185)

# (a) permutation nulls
ax=fig.add_subplot(gs[0])
t1=R['perm_tier1']
xs=np.linspace(0.15,1.02,400)
def gauss(x,m,s,h=0.62): return h*np.exp(-0.5*((x-m)/s)**2)
ax.fill_between(xs,0,gauss(xs,t1['null_mean'],t1['null_sd']),color="#E08A1E",alpha=.32,zorder=2)
ax.plot(xs,gauss(xs,t1['null_mean'],t1['null_sd']),color="#E08A1E",lw=1.6,zorder=3,
        label=f"Tier-1 null, compound-block\n({t1['null_mean']*100:.1f} $\\pm$ {t1['null_sd']*100:.1f}%, n=300)")
ax.fill_between(xs,0,gauss(xs,0.342,0.072),color="#999",alpha=.22,zorder=2)
ax.plot(xs,gauss(xs,0.342,0.072),color="#777",lw=1.3,ls='--',zorder=3,
        label="Tier-1 null, sample-level\n(34.2 $\\pm$ 7.2%) — anti-conservative")
ax.fill_between(xs,0,gauss(xs,n17.mean(),n17.std(ddof=1)),color="#2C6FB5",alpha=.30,zorder=2)
ax.plot(xs,gauss(xs,n17.mean(),n17.std(ddof=1)),color="#2C6FB5",lw=1.6,zorder=3,
        label=f"17-class null, sample-level\n({n17.mean()*100:.1f} $\\pm$ {n17.std(ddof=1)*100:.1f}%, n=120)")
ax.vlines(0.985,0,0.95,color="#C0392B",lw=2.0,zorder=5)
ax.text(0.975,0.86,"observed\nTier-1 98.5%",ha='right',fontsize=8.0,color="#C0392B",fontweight='bold',linespacing=1.2)
ax.vlines(0.618,0,0.95,color="#1B4F72",lw=2.0,ls=':',zorder=5)
ax.text(0.607,0.86,"observed\n17-class 61.8%",ha='right',fontsize=7.8,color="#1B4F72",fontweight='bold',linespacing=1.2)
ax.set_xlim(0.10,1.06); ax.set_ylim(0,1.58); ax.set_yticks([])
ax.set_xlabel("LOO-CV accuracy under permuted labels")
ax.legend(loc='upper center',bbox_to_anchor=(0.50,1.010),fontsize=6.8,handlelength=1.3,
          labelspacing=0.40,ncol=1,borderaxespad=0.0)
ax.spines[['top','right','left']].set_visible(False)
ptitle(ax,'a','Permutation nulls')

# (b) per-concentration
ax=fig.add_subplot(gs[1])
pc=R['per_conc']; ks=[10,50,100,500]
t1v=[pc[str(k)]['tier1']/17 for k in ks]; f17v=[pc[str(k)]['flat17']/17 for k in ks]
x=np.arange(4); W=0.36
b1=ax.bar(x-W/2,t1v,W,color="#E08A1E",label="Tier-1 family",zorder=3)
b2=ax.bar(x+W/2,f17v,W,color="#2C6FB5",label="17-class compound",zorder=3)
for rects,vals,ks_ in [(b1,t1v,ks),(b2,f17v,ks)]:
    for r_,v,k in zip(rects,vals,ks_):
        n=int(round(v*17))
        ax.text(r_.get_x()+r_.get_width()/2,v+0.02,f"{n}/17",ha='center',fontsize=7.6)
ax.axhline(1/17,ls=':',color='#888',lw=1.0); ax.text(3.45,1/17+0.022,"5.9% chance",fontsize=7.2,color='#666',ha='right')
ax.set_xticks(x); ax.set_xticklabels([f"{k} $\\mu$M" for k in ks])
ax.set_ylim(0,1.40); ax.set_ylabel("LOO-CV accuracy")
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.005),ncol=2,fontsize=7.8,
          columnspacing=1.4,borderaxespad=0.1)
ax.spines[['top','right']].set_visible(False)
ax.text(0.5,-0.215,"10 $\\mu$M is the lowest level tested,\nnot a detection limit.",transform=ax.transAxes,
        ha='center',fontsize=7.6,color='#555')
ptitle(ax,'b','Accuracy by concentration')

# (c) nested dye subsets
ax=fig.add_subplot(gs[2])
ks2=[1,2,3,4,5,8,12,29]; acc=[SUB[str(k)]['acc'] for k in ks2]
ax.plot(range(len(ks2)),acc,'-o',color="#2E7D32",lw=1.6,ms=6,zorder=4)
for i,(k,a) in enumerate(zip(ks2,acc)):
    ax.text(i,a+0.018,f"{int(round(a*68))}/68",ha='center',fontsize=7.4)
ax.axhline(67/68,ls='--',color="#888",lw=1.1)
ax.text(7.4,67/68-0.030,"full 29-dye panel (67/68)",ha='right',fontsize=7.4,color='#666')
ax.set_xticks(range(len(ks2))); ax.set_xticklabels(ks2)
ax.set_xlabel("Number of dyes in fold-internal subset"); ax.set_ylabel("Tier-1 LOO-CV accuracy")
ax.set_ylim(0.80,1.055); ax.spines[['top','right']].set_visible(False)
ax.text(0.5,-0.215,"Each size is unbiased; the best size is selected\non the same held-out points.",
        transform=ax.transAxes,ha='center',fontsize=7.4,color="#B03A2E")
ptitle(ax,'c','Nested dye-subset analysis')
fig.savefig("figures/newfig8.png",dpi=400,facecolor='white'); print("fig7 (newfig8.png) saved")
