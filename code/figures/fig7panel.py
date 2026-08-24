# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, glob
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse
plt.rcParams.update({"font.family":"DejaVu Sans","axes.linewidth":0.9})
p3=np.load('results/classification/p3.npy',allow_pickle=True)
D="data/processed/integrated_17compounds.csv"
dd=pd.read_csv(D); dd=dd[dd['Concentration']>0].reset_index(drop=True)
order=['DMMP','GA','GB','GD','GF','VX','AC','CG','CK','HD','HN','L','PS','A-230','A-232','A-234','A-242']
SER={**{a:'d1' for a in order[:6]},**{a:'d2' for a in order[6:13]},**{a:'d3' for a in order[13:]}}
C={'d1':"#1F5FA9",'d2':"#C0392B",'d3':"#2E7D32"}
NAME={'d1':'d1 (OP)','d2':'d2 (B/B/C)','d3':'d3 (Novichok)'}
pred500={a:p3[(dd['Agent']==a)&(dd['Concentration']==500)][0] for a in order}
rgb=pd.concat([pd.read_csv(f) for f in glob.glob("data/raw/*.rgb_deltaE_data.csv")])
rgb['agent']=rgb['agent'].replace({'A':'A-230','B':'A-232','C':'A-234','D':'A-242'})
r5=rgb[rgb['conc']==500]
assert set(order)<=set(r5['agent'].unique())

def ptitle(ax,l,t,dy,fs=16.4,x=0.0):
    tx=ax.text(x,dy,f"({l})",transform=ax.transAxes,fontsize=fs,fontweight="bold",va="bottom",ha="left")
    ax.figure.canvas.draw()
    bb=tx.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.text(bb.x1+0.012,dy,t,transform=ax.transAxes,fontsize=fs-1.6,va="bottom",ha="left")

fig=plt.figure(figsize=(15.6,9.2))
gs=fig.add_gridspec(1,2,width_ratios=[1.14,1.0],wspace=0.09,left=0.035,right=0.985,top=0.945,bottom=0.030)
gsR=gs[1].subgridspec(2,1,height_ratios=[0.90,1.10],hspace=0.20)

# ---------------- (a) measured readout ----------------
ax=fig.add_subplot(gs[0]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
y0,dy=0.900,0.0518
ax.text(0.030,y0+0.062,"Target",fontsize=13.2,fontweight='bold',ha='left')
ax.text(0.420,y0+0.062,"Dual-illumination readout",fontsize=13.2,fontweight='bold',ha='center')
ax.text(0.760,y0+0.080,"Out-of-fold",fontsize=13.2,fontweight='bold',ha='center')
ax.text(0.760,y0+0.046,"Tier-1 call",fontsize=13.2,fontweight='bold',ha='center')
ax.text(0.935,y0+0.062,"Correct?",fontsize=13.2,fontweight='bold',ha='center')
x0,x1=0.185,0.655
for i,a in enumerate(order):
    y=y0-i*dy; c=C[SER[a]]
    ax.add_patch(Rectangle((0.026,y-0.019),0.010,0.038,color=c,zorder=3))
    ax.text(0.048,y,a,fontsize=12.6,fontweight='bold',color=c,va='center')
    for light,off in [('Normal',+0.0122),('UV',-0.0122)]:
        ax.text(x0-0.013,y+off,'N' if light=='Normal' else 'UV',fontsize=8.6,color='#666',
                va='center',ha='right')
        sub=r5[(r5['agent']==a)&(r5['light']==light)].sort_values('dye')
        xs=x0+np.arange(len(sub))*(x1-x0)/28.0
        cols=[(rw.R/255,rw.G/255,rw.B/255) for _,rw in sub.iterrows()]
        ax.scatter(xs,np.full(len(xs),y+off),s=46,c=cols,edgecolor='#DDD',linewidth=0.3,zorder=3)
    pr=pred500[a]
    if pr==SER[a]:
        ax.text(0.760,y,NAME[pr],fontsize=12.6,fontweight='bold',color=C[pr],ha='center',va='center')
        ax.text(0.935,y,"yes",fontsize=12.6,fontweight='bold',color="#2E7D32",ha='center',va='center')
    else:
        ax.add_patch(FancyBboxPatch((0.676,y-0.022),0.168,0.044,boxstyle="round,pad=0.004",
                     ec="#C0392B",fc="none",lw=1.5,zorder=4))
        ax.text(0.760,y,NAME[pr],fontsize=12.6,fontweight='bold',color=C[pr],ha='center',va='center')
        ax.text(0.935,y,"no",fontsize=12.6,fontweight='bold',color="#C0392B",ha='center',va='center')
axA=ax

# ---------------- (b) workflow ----------------
ax=fig.add_subplot(gsR[0]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
from matplotlib.patches import Polygon
steps=[("1","Collect\nsample"),("2","Add to\ndye array"),("3","Image under\nwhite + UV"),
       ("4","Run calibrated\nclassifier"),("5","Report\nfamily call")]
TEAL=["#BFE3E0","#A5D8D4","#8BCDC8","#71C2BC","#57B7B0"]
L,R=0.020,0.980; yb,yt=0.410,0.740; ym=(yb+yt)/2
n=len(steps); tip=0.030
w=(R-L+tip*(n-1))/n           # chevrons interlock by `tip`
for i,(num,txt) in enumerate(steps):
    xa=L+i*(w-tip); xb=xa+w
    pts=[(xa,yb),(xb-tip,yb),(xb,ym),(xb-tip,yt),(xa,yt)]
    if i>0: pts.append((xa+tip,ym))     # notch on the left for all but the first
    ax.add_patch(Polygon(pts,closed=True,fc=TEAL[i],ec="#1B7F79",lw=1.4,zorder=3+i))
    cx=xa+(w+ (tip if i>0 else 0))/2 - tip/2
    t=ax.text(cx,ym+0.052,txt,ha='center',va='center',fontsize=12.4,linespacing=1.35,zorder=10+i)
    fig.canvas.draw()
    for _ in range(40):
        bw=t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(
               ax.transAxes.inverted()).width
        if bw<=(w-2*tip)*1.02: break
        t.set_fontsize(t.get_fontsize()-0.2); fig.canvas.draw()
    ax.text(cx,ym-0.086,num,ha='center',va='center',fontsize=11.6,fontweight='bold',
            color="#0F5C57",zorder=10+i)
ax.add_patch(FancyBboxPatch((0.030,0.045),0.940,0.195,boxstyle="round,pad=0.010,rounding_size=0.02",
             ec="#9EC5DC",fc="#EAF3F9",lw=1.1))
ax.text(0.50,0.143,"A deployment concept only: device-level calibration, open-set negatives,\n"
        "matrix-spiked tests and blinded validation are all outstanding.",
        ha='center',va='center',fontsize=11.5,linespacing=1.5)
axB=ax

# ---------------- (c) status ----------------
ax=fig.add_subplot(gsR[1]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
rows=[("Current evidence","Solution-phase CWA panel at 10–500 $\\mu$M;\ninternal cross-validation only"),
      ("Strongest claim","Chemical-family screening, plus direct evidence\non the response pathway"),
      ("Not claimed","LOD, environmental matrix performance,\nfield validation, cost"),
      ("Next validation","Matrix spikes with pH and buffer capacity\ncontrolled; replicate lots; blinded tests")]
h=0.185; top=0.930
for i,(k,v) in enumerate(rows):
    y=top-i*(h+0.032)
    ax.add_patch(Rectangle((0.025,y-h),0.275,h,ec="#BBB",fc="#F2F2F2",lw=1.0))
    ax.add_patch(Rectangle((0.300,y-h),0.675,h,ec="#BBB",fc="white",lw=1.0))
    ax.text(0.1625,y-h/2,k,ha='center',va='center',fontsize=12.6,fontweight='bold')
    ax.text(0.320,y-h/2,v,ha='left',va='center',fontsize=11.5,linespacing=1.45)
axC2=ax
# ---- lift (b) so its top (and therefore its label) sits level with (a) ------
fig.canvas.draw()
pa,pb,pc=axA.get_position(),axB.get_position(),axC2.get_position()
gap=pb.y0-pc.y1                       # keep the b-c gap
shift=pa.y1-pb.y1
axB.set_position([pb.x0,pb.y0+shift,pb.width,pb.height])
pb=axB.get_position()
axC2.set_position([pc.x0,pc.y0,pc.width,(pb.y0-gap)-pc.y0])   # (c) grows into the freed space
fig.canvas.draw()
pa,pb,pc=axA.get_position(),axB.get_position(),axC2.get_position()

DELTA=0.014                            # label offset above the axes top, in figure units
ptitle(axA,'a','Measured array readout and Tier-1 call (500 $\\mu$M)',dy=1+DELTA/pa.height)
ptitle(axB,'b','Prototype workflow concept',                          dy=1+DELTA/pb.height)
ptitle(axC2,'c','Operational status and validation needs',            dy=1+DELTA/pc.height)
fig.canvas.draw()
ya=pa.y1+DELTA; yb=pb.y1+DELTA
print(f"  (a) axes top {pa.y1:.4f} -> label y {ya:.4f}")
print(f"  (b) axes top {pb.y1:.4f} -> label y {yb:.4f}   aligned: {abs(ya-yb)<1e-9}")
fig.savefig("figures/newfig7.png",dpi=400,facecolor='white'); print("fig7 saved")
