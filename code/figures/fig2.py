# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.2,"axes.linewidth":0.9,
 "axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,"legend.frameon":True,"legend.fontsize":8.4})
D="data/processed/integrated_17compounds.csv"
df=pd.read_csv(D); feat=[c for c in df.columns if c.startswith('Dye')]
df=df[df['Concentration']>0].reset_index(drop=True)
X=df[feat].values; y=df['Agent'].values
SER={**{a:'d1' for a in ['DMMP','GA','GB','GD','GF','VX']},
     **{a:'d2' for a in ['AC','CG','CK','HD','HN','L','PS']},
     **{a:'d3' for a in ['A-230','A-232','A-234','A-242']}}
ys=np.array([SER[a] for a in y])
C={'d1':"#1F5FA9",'d2':"#C0392B",'d3':"#2E7D32"}
def ptitle(ax,l,t,dy=1.045,fs=12.5):
    tx=ax.text(0,dy,f"({l})",transform=ax.transAxes,fontsize=fs,fontweight="bold",va="bottom",ha="left")
    ax.figure.canvas.draw()
    bb=tx.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.text(bb.x1+0.020,dy,t,transform=ax.transAxes,fontsize=fs-1.2,va="bottom",ha="left")

fig=plt.figure(figsize=(13.6,9.0))
gs=fig.add_gridspec(2,1,hspace=0.44,left=0.062,right=0.965,top=0.935,bottom=0.065)
gsT=gs[0].subgridspec(1,2,wspace=0.24)
gsB=gs[1].subgridspec(1,2,width_ratios=[1.36,0.64],wspace=0.26)

# (a) PCA
ax=fig.add_subplot(gsT[0,0])
Z=PCA(n_components=2).fit(StandardScaler().fit_transform(X))
P=Z.transform(StandardScaler().fit_transform(X))
for s_,mk in [('d1','o'),('d2','^'),('d3','s')]:
    m=ys==s_
    ax.scatter(P[m,0],P[m,1],c=C[s_],marker=mk,s=42,alpha=.9,edgecolor='none',
               label={'d1':'d1: OP nerve (n = 24)','d2':'d2: Blood/blister/choking (n = 28)','d3':'d3: Novichok A-series (n = 16)'}[s_])
ax.axhline(0,color='#BBB',lw=.8,ls='--'); ax.axvline(0,color='#BBB',lw=.8,ls='--')
ax.set_xlabel(f"PC1 ({Z.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({Z.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(loc='upper right'); ax.spines[['top','right']].set_visible(False)
ptitle(ax,'a','PCA of the 58-channel fingerprints')

# (b) hierarchical accuracy  -- CORRECTED
ax=fig.add_subplot(gsT[0,1])
lab=["Tier-1\nfamily","Tier-2\nd1 OP nerve","Tier-2\nd2 blood/blister/\nchoking","Tier-2\nd3 Novichok"]
acc=[67/68,13/24,19/28,7/16]; frac=["(67/68)","(13/24)","(19/28)","(7/16)"]
ch=[1/3,1/6,1/7,1/4]; col=["#E08A1E",C['d1'],C['d2'],C['d3']]
xs=np.arange(4)
ax.bar(xs,acc,color=col,width=.66,zorder=3)
for i,(a,f) in enumerate(zip(acc,frac)):
    ax.text(i,a+0.075,f"{a*100:.1f}%",ha='center',fontweight='bold',fontsize=11.5)
    ax.text(i,a+0.028,f,ha='center',fontsize=8.6,color="#555")
for i,c in enumerate(ch):
    ax.plot([i-.36,i+.36],[c,c],ls='--',lw=1.2,color='#333',zorder=4)
    ax.text(i+.39,c,f"{c*100:.1f}%",va='center',fontsize=8.2,color='#333')
ax.plot([],[],ls='--',lw=1.2,color='#333',label='chance level for that tier')
ax.set_xticks(xs); ax.set_xticklabels(lab,fontsize=8.8); ax.set_ylim(0,1.22)
ax.set_yticks([0,.2,.4,.6,.8,1.0]); ax.set_ylabel("LOO-CV accuracy")
ax.legend(loc='upper right',fontsize=8.2); ax.spines[['top','right']].set_visible(False)
ptitle(ax,'b','Hierarchical classification accuracy')

# (c) decision flow -- decon claims REMOVED
ax=fig.add_subplot(gsB[0,0]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
def _fit(t,limit):
    """shrink the font of text artist t until its width is under `limit` axes-units"""
    fig.canvas.draw()
    for _ in range(40):
        bb=t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        if bb.width<=limit: return bb.width
        t.set_fontsize(t.get_fontsize()-0.2); fig.canvas.draw()
    return bb.width

def box(x,y,w,h,txt,ec,fc,fs=9.0,bold=False,sub=None):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",
                 ec=ec,fc=fc,lw=1.3,zorder=2))
    t=ax.text(x,y+(0.026 if sub else 0),txt,ha='center',va='center',fontsize=fs,
            fontweight='bold' if bold else 'normal',color=ec if bold else '#111',zorder=3)
    _fit(t,w*0.90)
    if sub:
        t2=ax.text(x,y-0.032,sub,ha='center',va='center',fontsize=7.6,color='#444',zorder=3,
                   linespacing=1.35)
        _fit(t2,w*0.90)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,lw=1.0,color='#555',zorder=1))
box(.5,.93,.62,.10,"58-channel $\\Delta$E fingerprint","#2C6FB5","#EAF2FB")
arr(.5,.88,.5,.80)
box(.5,.745,.66,.105,"Tier-1 chemical-family assignment (98.5%)","#E08A1E","#FDF3E3")
for xc,s_,nm,sub in [(.165,'d1','d1: OP nerve agents','G-, V-series'),
                     (.5,'d2','d2: Blood / blister / choking','cyanide, mustard,\nphosgene, arsenical'),
                     (.835,'d3','d3: Novichok A-series','A-230/232/234/242')]:
    arr(.5,.69,xc,.592)
    box(xc,.505,.305,.160,nm,C[s_],"#FFFFFF",fs=8.6,bold=True,sub=sub)
    arr(xc,.425,.5,.30)
box(.5,.245,.80,.105,"Tier-2 compound identification (secondary, non-confirmatory)","#1B7F79","#E6F4F3",fs=8.8)
ax.text(.5,.085,"The family call narrows an unknown and indicates which confirmatory assay to prioritise.\n"
        "It is not a decontamination prescription: within d2 the treatments differ, and d1 and d3\n"
        "are both cleaved by nucleophilic attack at phosphorus.",ha='center',va='center',fontsize=7.9,color='#B03A2E',
        linespacing=1.45)
ptitle(ax,'c','Decision flow from the Tier-1 output')

# (d) Tier-1 confusion (unchanged)
ax=fig.add_subplot(gsB[0,1])
M=np.array([[24,0,0],[0,28,0],[0,1,15]],dtype=float); N=M.sum(1,keepdims=True)
im=ax.imshow(M/N,cmap='Blues',vmin=0,vmax=1)
for i in range(3):
    for j in range(3):
        v=M[i,j]/N[i,0]
        ax.text(j,i-0.09,f"{v:.2f}",ha='center',va='center',fontsize=11.5,color='white' if v>.5 else '#222')
        ax.text(j,i+0.14,f"({int(M[i,j])}/{int(N[i,0])})",ha='center',va='center',fontsize=8.6,color='white' if v>.5 else '#444')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(["d1: OP\nnerve","d2: Blood/\nblister/choking","d3: Novichok"],fontsize=8.8)
ax.set_yticklabels([f"d1  (n = 24)",f"d2  (n = 28)",f"d3  (n = 16)"],fontsize=8.8)
for t,c in zip(ax.get_xticklabels(),[C['d1'],C['d2'],C['d3']]): t.set_color(c)
for t,c in zip(ax.get_yticklabels(),[C['d1'],C['d2'],C['d3']]): t.set_color(c)
ax.set_xlabel("Predicted family"); ax.set_ylabel("True family")
fig.colorbar(im,ax=ax,fraction=0.040,pad=0.03,label="Proportion")
ptitle(ax,'d','Tier-1 confusion matrix (LOO-CV)')
ax.text(0.5,-0.30,"Overall Tier-1 accuracy 98.5% (67/68); the single error is A-242 at 500 $\\mu$M assigned to d2.",
        transform=ax.transAxes,ha='center',fontsize=8.4,color='#555')
fig.savefig("figures/newfig2.png",dpi=400,facecolor='white')
print("fig2 saved")
