# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, json
from sklearn.metrics import confusion_matrix, classification_report
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.9,
 "axes.labelsize":9.6,"xtick.labelsize":8.4,"ytick.labelsize":8.4,"legend.frameon":False,"legend.fontsize":8.2})
p17=np.load('results/classification/p17.npy',allow_pickle=True)
D="data/processed/integrated_17compounds.csv"
df=pd.read_csv(D); df=df[df['Concentration']>0].reset_index(drop=True); y=df['Agent'].values
order=['DMMP','GA','GB','GD','GF','VX','AC','CG','CK','HD','HN','L','PS','A-230','A-232','A-234','A-242']
SER={**{a:'d1' for a in order[:6]},**{a:'d2' for a in order[6:13]},**{a:'d3' for a in order[13:]}}
C={'d1':"#1F5FA9",'d2':"#C0392B",'d3':"#2E7D32"}
auc=json.load(open('results/classification/auc.json'))['per_agent']
def ptitle(ax,l,t,dy=1.04,fs=11.5):
    tx=ax.text(0,dy,f"({l})",transform=ax.transAxes,fontsize=fs,fontweight="bold",va="bottom",ha="left")
    ax.figure.canvas.draw()
    bb=tx.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.text(bb.x1+0.020,dy,t,transform=ax.transAxes,fontsize=fs-1.2,va="bottom",ha="left")
fig=plt.figure(figsize=(13.4,10.2))
gs=fig.add_gridspec(2,2,hspace=0.40,wspace=0.30,left=0.075,right=0.965,top=0.935,bottom=0.085)
R_D=0.6964   # solved so that width(a)+width(b) == width(c)+width(d)
_side=(1.0-R_D)/2.0
gsd=gs[1,1].subgridspec(1,3,width_ratios=[_side,R_D,_side],wspace=0.0)

# (a) confusion
ax=fig.add_subplot(gs[0,0])
M=confusion_matrix(y,p17,labels=order).astype(float); Mn=M/M.sum(1,keepdims=True)
im=ax.imshow(Mn,cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(17)); ax.set_yticks(range(17))
ax.set_xticklabels(order,rotation=90,fontsize=7.6); ax.set_yticklabels(order,fontsize=7.6)
for t,a in zip(ax.get_xticklabels(),order): t.set_color(C[SER[a]])
for t,a in zip(ax.get_yticklabels(),order): t.set_color(C[SER[a]])
for b in [5.5,12.5]:
    ax.axhline(b,color='#444',lw=1.1,ls='--'); ax.axvline(b,color='#444',lw=1.1,ls='--')
for i in range(17):
    if Mn[i,i]>0: ax.text(i,i,f"{Mn[i,i]:.2f}",ha='center',va='center',fontsize=6.6,
                          color='white' if Mn[i,i]>.5 else '#222')
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.colorbar(im,ax=ax,fraction=0.044,pad=0.02,label="Row-normalised proportion")
ptitle(ax,'a','Confusion matrix, 17-class LOO-CV')

# (b) per-family P/R/F1
ax=fig.add_subplot(gs[0,1])
r=classification_report(y,p17,labels=order,output_dict=True,zero_division=0)
fams=['d1','d2','d3']; W=0.26; xs=np.arange(3)
vals={m:[np.mean([r[a][m] for a in order if SER[a]==f]) for f in fams] for m in ['precision','recall','f1-score']}
for i,(m,lab,c) in enumerate([('precision','Precision','#4C72B0'),('recall','Recall','#DD8452'),('f1-score','F1-score','#55A868')]):
    b=ax.bar(xs+(i-1)*W,vals[m],width=W,color=c,label=lab,zorder=3)
    for rect,v in zip(b,vals[m]): ax.text(rect.get_x()+rect.get_width()/2,v+0.015,f"{v:.2f}",ha='center',fontsize=7.4)
ax.set_xticks(xs); ax.set_xticklabels(["d1\nOP nerve","d2\nBlood/blister/choking","d3\nNovichok A-series"],fontsize=8.6)
ax.set_ylim(0,0.92); ax.set_ylabel("Score (mean over compounds in family)"); ax.legend(loc='upper right')
ax.spines[['top','right']].set_visible(False)
ptitle(ax,'b','Per-family aggregate performance')

# (c) per-agent AUC
ax=fig.add_subplot(gs[1,0])
vals=[auc[a] for a in order]; cols=[C[SER[a]] for a in order]
ax.barh(np.arange(17)[::-1],vals,color=cols,height=.66,zorder=3)
for i,(a,v) in enumerate(zip(order,vals)):
    ax.text(v+0.006,16-i,f"{v:.3f}",va='center',fontsize=7.4)
mac=np.mean(vals)
ax.axvline(mac,ls='--',lw=1.2,color='#333',zorder=4)
ax.text(mac-0.008,-0.9,f"macro {mac:.3f}",ha='right',fontsize=8.0,color='#333')
ax.axvline(0.5,ls=':',lw=1.0,color='#999',zorder=2)
ax.set_yticks(np.arange(17)[::-1]); ax.set_yticklabels(order,fontsize=7.8)
for t,a in zip(ax.get_yticklabels(),order): t.set_color(C[SER[a]])
ax.set_xlim(0.4,1.06); ax.set_xlabel("One-vs-rest AUC (out-of-fold LOO decision scores)")
ax.spines[['top','right']].set_visible(False)
ptitle(ax,'c','Per-compound OvR AUC')

# (d) error structure
ax=fig.add_subplot(gsd[0,1])
err=[(y[i],p17[i]) for i in range(68) if p17[i]!=y[i]]
w=sum(1 for t,q in err if SER[t]==SER[q]); x=len(err)-w
b=ax.bar([0,1],[w,x],color=["#4C72B0","#C0392B"],width=.5,zorder=3)
for rect,v,lab in zip(b,[w,x],[f"{w} ({w/len(err)*100:.1f}%)",f"{x} ({x/len(err)*100:.1f}%)"]):
    ax.text(rect.get_x()+rect.get_width()/2,v+0.5,lab,ha='center',fontweight='bold',fontsize=10.5)
ax.set_xticks([0,1]); ax.set_xticklabels(["Within correct\nfamily","Crosses family\nboundary"],fontsize=8.8)
ax.set_ylim(0,len(err)+5); ax.set_ylabel(f"Number of errors (total {len(err)} of 68)")
ax.spines[['top','right']].set_visible(False)
cross=[f"{t} → {q}" for t,q in err if SER[t]!=SER[q]]
ax.text(1,x+3.0,"\n".join(cross),ha='center',fontsize=8.2,color="#C0392B")
ax.text(0.5,-0.16,"The family call survives even where compound identity fails.",transform=ax.transAxes,
        ha='center',fontsize=8.6,color='#444')
ptitle(ax,'d','Error structure of the flat 17-class model')
# ---- align bottom row to the top row -------------------------------------
# (a) is squeezed by its square aspect, so it sits inboard of the column edge.
# Put (c)'s left edge exactly at (a)'s, and (d)'s right edge exactly at (b)'s,
# which also equalises the a-b and c-d gaps.
fig.canvas.draw()
main=[x for x in fig.get_axes() if x.get_label()!='<colorbar>']
tp=sorted([x for x in main if x.get_position().y0>0.5],key=lambda x:x.get_position().x0)
bt=sorted([x for x in main if x.get_position().y0<=0.5],key=lambda x:x.get_position().x0)
axA,axB=tp; axC,axD=bt
pa,pb=axA.get_position(),axB.get_position()
pc,pd=axC.get_position(),axD.get_position()
axC.set_position([pa.x0, pc.y0, pc.width, pc.height])
axD.set_position([pb.x1-pd.width, pd.y0, pd.width, pd.height])
fig.canvas.draw()
pa,pb=axA.get_position(),axB.get_position()
pc,pd=axC.get_position(),axD.get_position()
print(f"  a {pa.x0:.4f}-{pa.x1:.4f} (w {pa.width:.4f})   b {pb.x0:.4f}-{pb.x1:.4f} (w {pb.width:.4f})")
print(f"  c {pc.x0:.4f}-{pc.x1:.4f} (w {pc.width:.4f})   d {pd.x0:.4f}-{pd.x1:.4f} (w {pd.width:.4f})")
print(f"  label (a) x = {pa.x0:.4f} ; label (c) x = {pc.x0:.4f}  -> aligned: {abs(pa.x0-pc.x0)<1e-6}")
print(f"  right edge b = {pb.x1:.4f} ; d = {pd.x1:.4f}          -> aligned: {abs(pb.x1-pd.x1)<1e-6}")
print(f"  gap a-b = {pb.x0-pa.x1:.4f} ; gap c-d = {pd.x0-pc.x1:.4f}")
print(f"  width a+b = {pa.width+pb.width:.4f} ; c+d = {pc.width+pd.width:.4f}")
OUT="figures/newfig6.png"
fig.savefig(OUT,dpi=400,facecolor='white')

# ---- trim the unused white band on the left --------------------------------
from PIL import Image
import numpy as _np
im=Image.open(OUT).convert("RGB"); a=_np.asarray(im)
nonwhite=(a<250).any(axis=2)           # any pixel that is not near-white
cols=_np.where(nonwhite.any(axis=0))[0]
pad=int(round(0.010*im.width))         # keep a small, even margin
left=max(0,cols[0]-pad)
im.crop((left,0,im.width,im.height)).save(OUT)
print(f"  cropped {left} px ({left/im.width*100:.1f}% of width) from the left; "
      f"{im.width}x{im.height} -> {im.width-left}x{im.height}")
print("fig6 saved")
