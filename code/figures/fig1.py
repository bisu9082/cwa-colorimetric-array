# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.2,"axes.linewidth":0.9,
 "axes.labelsize":9.8,"xtick.labelsize":8.6,"ytick.labelsize":8.6,"legend.frameon":True,"legend.fontsize":8.2})
D="data/processed/integrated_17compounds.csv"
df=pd.read_csv(D); feat=[c for c in df.columns if c.startswith('Dye')]
N=[f"Dye{i}_N" for i in range(1,30)]; U=[f"Dye{i}_U" for i in range(1,30)]
order=['DMMP','GA','GB','GD','GF','VX','AC','CG','CK','HD','HN','L','PS','A-230','A-232','A-234','A-242']
SER={**{a:'d1' for a in order[:6]},**{a:'d2' for a in order[6:13]},**{a:'d3' for a in order[13:]}}
C={'d1':"#1F5FA9",'d2':"#C0392B",'d3':"#2E7D32"}
LBL={'d1':'OP nerve','d2':'Blood/blister/choking','d3':'Novichok A-series'}
def ptitle(ax,l,t,dy=1.045,fs=12.0):
    tx=ax.text(0,dy,f"({l})",transform=ax.transAxes,fontsize=fs,fontweight="bold",va="bottom",ha="left")
    ax.figure.canvas.draw()
    bb=tx.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.text(bb.x1+0.022,dy,t,transform=ax.transAxes,fontsize=fs-1.4,va="bottom",ha="left")

fig=plt.figure(figsize=(15.0,6.0))
# (c) gets the widest column so its legend fits
gs=fig.add_gridspec(1,3,width_ratios=[1.02,0.78,1.28],wspace=0.30,
                    left=0.055,right=0.985,top=0.845,bottom=0.135)

# ---------- (a) heatmap ----------
ax=fig.add_subplot(gs[0])
d500=df[df['Concentration']==500].set_index('Agent')
M=np.vstack([np.concatenate([d500.loc[a,N].values, d500.loc[a,U].values]) for a in order]).astype(float)
im=ax.imshow(M,aspect='auto',cmap='YlOrRd',vmin=0,vmax=np.nanmax(M))
ax.axvline(28.5,color='k',lw=1.2)
for b in [5.5,12.5]: ax.axhline(b,color='k',lw=1.2)
ax.set_yticks(range(17)); ax.set_yticklabels(order,fontsize=8.4)
for t,a in zip(ax.get_yticklabels(),order): t.set_color(C[SER[a]])
ax.set_xticks([0,9,19,29,38,48,57]); ax.set_xticklabels([1,10,20,1,10,20,29],fontsize=8.0)
ax.set_xlabel("Dye index   (normal block | UV block)")
ax.text(14,-0.90,"Normal illumination",ha='center',fontsize=8.8,fontweight='bold')
ax.text(43,-0.90,"UV 365 nm",ha='center',fontsize=8.8,fontweight='bold')
fig.colorbar(im,ax=ax,fraction=0.040,pad=0.02,label="$\\Delta E$")
ptitle(ax,'a','Measured 58-channel $\\Delta E$ at 500 $\\mu$M',dy=1.085)

# ---------- (b) concentration dependence ----------
ax=fig.add_subplot(gs[1])
concs=[10,50,100,500]
for s_ in ['d1','d2','d3']:
    ags=[a for a in order if SER[a]==s_]; mu=[];sd=[]
    for c in concs:
        sub=df[(df['Concentration']==c)&(df['Agent'].isin(ags))]
        per=sub[feat].mean(axis=1).values; mu.append(per.mean()); sd.append(per.std(ddof=1))
    ax.errorbar(range(4),mu,yerr=sd,marker='o',ms=6,lw=1.9,capsize=3.5,color=C[s_],
                label=f"{LBL[s_]} (n = {len(ags)})")
ax.set_xticks(range(4)); ax.set_xticklabels(concs)
ax.set_xlabel("Concentration ($\\mu$M)"); ax.set_ylabel("Mean $\\Delta E$ across 58 channels")
ax.legend(loc='upper left',fontsize=7.6); ax.spines[['top','right']].set_visible(False)
ax.text(0.5,-0.175,"Error bars: SD across the targets of a family.",transform=ax.transAxes,
        ha='center',fontsize=7.8,color='#555')
ptitle(ax,'b','Concentration dependence',dy=1.085)

# ---------- (c) normal vs UV totals ----------
ax=fig.add_subplot(gs[2])
xs=np.array([d500.loc[a,N].values.astype(float).sum() for a in order])
ys=np.array([d500.loc[a,U].values.astype(float).sum() for a in order])
for s_ in ['d1','d2','d3']:
    m=np.array([SER[a]==s_ for a in order])
    ax.scatter(xs[m],ys[m],s=62,color=C[s_],edgecolor='none',label=LBL[s_],zorder=4)
lim=max(xs.max(),ys.max())*1.16
ax.plot([0,lim],[0,lim],ls='--',lw=1.1,color='#AAA',zorder=2)
ax.text(lim*0.955,lim*0.995,"y = x",fontsize=8.2,color='#888',ha='right')
off={'GF':(6,10),'GD':(6,-13),'L':(7,8),'CG':(6,-13),'A-242':(9,4),'VX':(8,4),'A-234':(9,2),
     'A-232':(9,-3),'A-230':(9,-4),'PS':(6,9),'CK':(4,-12),'HN':(6,-12),'AC':(0,-14),
     'HD':(-6,-11),'GB':(-8,4),'GA':(-7,8),'DMMP':(-8,-6)}
for a,x,y in zip(order,xs,ys):
    dx,dy=off[a]
    ax.annotate(a,(x,y),textcoords="offset points",xytext=(dx,dy),fontsize=7.9,
                color=C[SER[a]],ha='left' if dx>=0 else 'right',va='center')
ax.set_xlim(-lim*0.05,lim); ax.set_ylim(-lim*0.05,lim)
ax.set_xlabel("$\\Sigma\\Delta E$, normal channels"); ax.set_ylabel("$\\Sigma\\Delta E$, UV channels")
ax.legend(loc='lower right',fontsize=8.4,handletextpad=0.5,borderpad=0.55,framealpha=0.96)
ax.spines[['top','right']].set_visible(False)
ptitle(ax,'c','Normal vs. UV channel totals',dy=1.085)
fig.savefig("figures/newfig1.png",dpi=400,facecolor='white'); print("fig1 saved")
