# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, json, sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed
D="data/processed/integrated_17compounds.csv"
df=pd.read_csv(D); feat=[c for c in df.columns if c.startswith('Dye')]
df=df[df['Concentration']>0].reset_index(drop=True)
X=df[feat].values; y=df['Agent'].values
SER={**{a:'d1' for a in ['DMMP','GA','GB','GD','GF','VX']},
     **{a:'d2' for a in ['AC','CG','CK','HD','HN','L','PS']},
     **{a:'d3' for a in ['A-230','A-232','A-234','A-242']}}
agents=sorted(set(y))
SC=[]
for tr,te in LeaveOneOut().split(X):
    sc=StandardScaler().fit(X[tr]); SC.append((sc.transform(X[tr]),sc.transform(X[te]),tr,te[0]))
from sklearn.multiclass import OneVsRestClassifier
def looacc(lab):
    ok=0
    for Xtr,Xte,tr,t0 in SC:
        m=OneVsRestClassifier(SVC(kernel='rbf',C=10.0,gamma='scale'),n_jobs=1)
        m.fit(Xtr,lab[tr]); ok += m.predict(Xte)[0]==lab[t0]
    return ok/68
NP=int(sys.argv[1]); which=sys.argv[2]
def one(seed):
    r=np.random.default_rng(seed)
    if which=='t1':
        mp=dict(zip(agents,r.permutation([SER[a] for a in agents])))
    else:
        mp=dict(zip(agents,r.permutation(agents)))
    return looacc(np.array([mp[a] for a in y]))
out=np.array(Parallel(n_jobs=2)(delayed(one)(7000+i) for i in range(NP)))
R=json.load(open('results/classification/full1.json'))
obs=R['tier1']['acc'] if which=='t1' else R['flat17']['acc']
p=(np.sum(out>=obs)+1)/(NP+1); z=(obs-out.mean())/out.std(ddof=1)
key='perm_tier1' if which=='t1' else 'perm_flat17'
R[key]=dict(obs=float(obs),null_mean=float(out.mean()),null_sd=float(out.std(ddof=1)),
            null_max=float(out.max()),z=float(z),p=float(p),n_perm=NP,level='compound-block')
json.dump(R,open('results/classification/full1.json','w'),indent=1,default=float)
print(f"{key}: obs {obs*100:.1f}%  null {out.mean()*100:.1f}+/-{out.std(ddof=1)*100:.1f}%  max {out.max()*100:.1f}%  z={z:.1f}  p={p:.4f}  (n={NP})")
