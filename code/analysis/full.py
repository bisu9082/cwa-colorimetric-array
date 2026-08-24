# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, json, itertools
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, cohen_kappa_score, balanced_accuracy_score
rng=np.random.default_rng(42)
D="data/processed/integrated_17compounds.csv"
df=pd.read_csv(D); feat=[c for c in df.columns if c.startswith('Dye')]
full=df.copy(); df=df[df['Concentration']>0].reset_index(drop=True)
X=df[feat].values; y=df['Agent'].values; conc=df['Concentration'].values
SER={**{a:'d1' for a in ['DMMP','GA','GB','GD','GF','VX']},
     **{a:'d2' for a in ['AC','CG','CK','HD','HN','L','PS']},
     **{a:'d3' for a in ['A-230','A-232','A-234','A-242']}}
ys=np.array([SER[a] for a in y]); agents=sorted(set(y))
R={}

def mk(): return OneVsRestClassifier(SVC(kernel='rbf',C=10.0,gamma='scale',random_state=42))
def loo(X,lab,ret_scores=False):
    pred=np.empty(len(lab),dtype=object); scores=[]
    cls=sorted(set(lab))
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler().fit(X[tr]); m=mk(); m.fit(sc.transform(X[tr]),lab[tr])
        Xt=sc.transform(X[te]); pred[te[0]]=m.predict(Xt)[0]
        if ret_scores:
            d=m.decision_function(Xt)[0]
            row=dict(zip(m.classes_,d)); scores.append([row.get(c,np.nan) for c in cls])
    return (pred,np.array(scores),cls) if ret_scores else pred

# ---------- Tier-1 and 17-class, leak-free ----------
p3=loo(X,ys); p17,S17,cls17=loo(X,y,ret_scores=True)
n3=int((p3==ys).sum()); n17=int((p17==y).sum())
R['tier1']=dict(n=n3,acc=n3/68); R['flat17']=dict(n=n17,acc=n17/68)
print(f"Tier-1 (in-fold)  {n3}/68 = {n3/68*100:.1f}%")
print(f"17-class (in-fold) {n17}/68 = {n17/68*100:.1f}%   [reported 43/68=63.2% used GLOBAL scaling]")
R['kappa17']=cohen_kappa_score(y,p17); R['bal17']=balanced_accuracy_score(y,p17)
R['kappa3']=cohen_kappa_score(ys,p3);  R['bal3']=balanced_accuracy_score(ys,p3)
print(f"  17-class kappa={R['kappa17']:.3f}  balanced acc={R['bal17']*100:.1f}%")
print(f"  Tier-1   kappa={R['kappa3']:.3f}  balanced acc={R['bal3']*100:.1f}%")

# error structure
err=[(y[i],p17[i]) for i in range(68) if p17[i]!=y[i]]
within=sum(1 for t,p in err if SER[t]==SER[p]); cross=len(err)-within
R['errors']=dict(total=len(err),within=within,cross=cross,
                 cross_list=[f"{t}->{p}" for t,p in err if SER[t]!=SER[p]])
print(f"  errors {len(err)}: within-series {within} ({within/len(err)*100:.1f}%), cross {cross} {R['errors']['cross_list']}")

# ---------- per-agent ----------
c3={a:int(((y==a)&(p3==ys)).sum()) for a in agents}
c17={a:int(((y==a)&(p17==y)).sum()) for a in agents}
R['per_agent']=dict(tier1=c3,flat17=c17)

# ---------- cluster bootstrap over agents ----------
def cboot(c,B=20000):
    ag=list(c); a=[]
    for _ in range(B):
        pk=rng.choice(ag,size=len(ag),replace=True)
        a.append(sum(c[x] for x in pk)/(4*len(ag)))
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
R['ci_tier1']=cboot(c3); R['ci_flat17']=cboot(c17)
print(f"\ncluster bootstrap (resample 17 agents, B=20000):")
print(f"  Tier-1   {n3/68*100:.1f}%  [{R['ci_tier1'][0]*100:.1f}, {R['ci_tier1'][1]*100:.1f}]")
print(f"  17-class {n17/68*100:.1f}%  [{R['ci_flat17'][0]*100:.1f}, {R['ci_flat17'][1]*100:.1f}]")

# ---------- hierarchical ----------
tier2_ok=np.zeros(68,dtype=bool)
for s in ['d1','d2','d3']:
    idx=np.where(ys==s)[0]
    pp=loo(X[idx],y[idx])
    tier2_ok[idx]=(pp==y[idx])
    print(f"  Tier-2 {s}: {int((pp==y[idx]).sum())}/{len(idx)}")
both=int(((p3==ys)&tier2_ok).sum())
R['hier']=dict(n=both,acc=both/68)
print(f"  hierarchical (BOTH tiers correct) = {both}/68 = {both/68*100:.1f}%")

# ---------- LOCO, in-fold ----------
loco={}
for c in sorted(set(conc)):
    te=conc==c; tr=~te
    sc=StandardScaler().fit(X[tr]); m=mk(); m.fit(sc.transform(X[tr]),y[tr])
    pr=m.predict(sc.transform(X[te])); loco[int(c)]=float((pr==y[te]).mean())
R['loco']=loco; R['loco_mean']=float(np.mean(list(loco.values())))
R['loco_sd']=float(np.std(list(loco.values()),ddof=1))
print(f"\nLOCO (in-fold): {[f'{k}uM {v*100:.1f}%' for k,v in loco.items()]}")
print(f"  mean {R['loco_mean']*100:.1f}%  SD {R['loco_sd']*100:.1f} pp  SE {R['loco_sd']/2*100:.1f} pp")

# ---------- per-concentration LOO ----------
R['per_conc']={int(c):dict(tier1=int((p3[conc==c]==ys[conc==c]).sum()),
                           flat17=int((p17[conc==c]==y[conc==c]).sum()),n=int((conc==c).sum()))
               for c in sorted(set(conc))}
print(f"per-conc LOO: {R['per_conc']}")
json.dump(R,open('results/classification/full1.json','w'),indent=1,default=float)
np.save('results/classification/S17.npy',S17); json.dump(cls17,open('results/classification/cls17.json','w'))
np.save('results/classification/p17.npy',p17.astype(str)); np.save('results/classification/p3.npy',p3.astype(str))
