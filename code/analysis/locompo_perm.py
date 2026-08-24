import numpy as np, pandas as pd, json
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("data/processed/integrated_17compounds.csv")
feat=[c for c in df.columns if c.startswith('Dye')]
df=df[df['Concentration']>0].reset_index(drop=True)
X=df[feat].values; y=df['Agent'].values
SER={**{a:'d1' for a in ['DMMP','GA','GB','GD','GF','VX']},
     **{a:'d2' for a in ['AC','CG','CK','HD','HN','L','PS']},
     **{a:'d3' for a in ['A-230','A-232','A-234','A-242']}}
comps=sorted(set(y))
def mk(): return OneVsRestClassifier(SVC(kernel='rbf',C=10.0,gamma='scale',random_state=42))
def locompo(cmap):
    lab=np.array([cmap[a] for a in y]); ok=0
    for c in comps:
        te=(y==c); tr=~te
        if len(set(lab[tr]))<2: continue
        sc=StandardScaler().fit(X[tr]); m=mk(); m.fit(sc.transform(X[tr]),lab[tr])
        ok+=int((m.predict(sc.transform(X[te]))==lab[te]).sum())
    return ok
obs=locompo(SER)
print(f"observed LOCompO = {obs}/68 = {obs/68*100:.1f}%")
rng=np.random.default_rng(2026)
labs=[SER[c] for c in comps]; null=[]
for i in range(500):
    p=list(rng.permutation(labs)); null.append(locompo(dict(zip(comps,p)))/68*100)
null=np.array(null); mu,sd=null.mean(),null.std(ddof=1)
pv=(np.sum(null>=obs/68*100)+1)/(len(null)+1)
print(f"null = {mu:.1f} +/- {sd:.1f}%   max={null.max():.1f}%")
print(f"p = {pv:.4f}   z = {(obs/68*100-mu)/sd:.1f}")
json.dump(dict(obs_n=obs,obs=obs/68*100,null_mean=mu,null_sd=sd,null_max=float(null.max()),
               p=pv,z=(obs/68*100-mu)/sd,B=len(null)),open("/tmp/locompo_perm.json","w"),indent=1)
