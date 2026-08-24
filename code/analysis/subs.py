# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, json
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from collections import Counter
D="data/processed/integrated_17compounds.csv"
df=pd.read_csv(D); feat=[c for c in df.columns if c.startswith('Dye')]
df=df[df['Concentration']>0].reset_index(drop=True)
X=df[feat].values; y=df['Agent'].values
SER={**{a:'d1' for a in ['DMMP','GA','GB','GD','GF','VX']},
     **{a:'d2' for a in ['AC','CG','CK','HD','HN','L','PS']},
     **{a:'d3' for a in ['A-230','A-232','A-234','A-242']}}
ys=np.array([SER[a] for a in y])
dye_of=np.array([int(c.split('_')[0].replace('Dye','')) for c in feat])
def mk(): return OneVsRestClassifier(SVC(kernel='rbf',C=10.0,gamma='scale',random_state=42))

print("=== nested dye-subset (ANOVA-F on dye, selection inside each LOO fold) ===")
res={}
for k in [1,2,3,4,5,8,12,29]:
    ok=0; chosen=[]
    for tr,te in LeaveOneOut().split(X):
        F,_=f_classif(X[tr],ys[tr]); F=np.nan_to_num(F)
        score={d:F[dye_of==d].max() for d in range(1,30)}
        top=sorted(score,key=score.get,reverse=True)[:k]; chosen.append(tuple(sorted(top)))
        cols=np.isin(dye_of,top)
        sc=StandardScaler().fit(X[tr][:,cols]); m=mk(); m.fit(sc.transform(X[tr][:,cols]),ys[tr])
        ok+= m.predict(sc.transform(X[te][:,cols]))[0]==ys[te[0]]
    mode,cnt=Counter(chosen).most_common(1)[0]
    res[k]=dict(n=int(ok),acc=ok/68,modal=list(mode),stability=cnt/68)
    print(f"  k={k:>2}: {ok}/68 = {ok/68*100:.1f}%   modal set {list(mode)}  stability {cnt/68*100:.0f}%")
json.dump(res,open('results/classification/subsets.json','w'),indent=1)

print("\n=== per-agent OvR AUC from out-of-fold decision scores (17-class) ===")
S=np.load('results/classification/S17.npy'); cls=json.load(open('results/classification/cls17.json'))
aucs={}
for i,c in enumerate(cls):
    try: aucs[c]=roc_auc_score((y==c).astype(int),S[:,i])
    except Exception as e: aucs[c]=np.nan
for c in cls: print(f"   {c:<7} {aucs[c]:.3f}")
print(f"  macro (17 agents) = {np.nanmean(list(aucs.values())):.3f}")
# series-level AUC by max score within series
ser_auc={}
for s in ['d1','d2','d3']:
    idx=[i for i,c in enumerate(cls) if SER[c]==s]
    sc=S[:,idx].max(axis=1)
    ser_auc[s]=roc_auc_score((ys==s).astype(int),sc)
    print(f"  series {s} (max OvR score within series) = {ser_auc[s]:.3f}")
print(f"  macro (3 series) = {np.mean(list(ser_auc.values())):.3f}")
json.dump({'per_agent':aucs,'series':ser_auc},open('results/classification/auc.json','w'),indent=1,default=float)
