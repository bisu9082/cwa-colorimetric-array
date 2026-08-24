import numpy as np, pandas as pd, json
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed
D="data/processed/integrated_17compounds.csv"
df=pd.read_csv(D); feat=[c for c in df.columns if c.startswith('Dye')]
df=df[df['Concentration']>0].reset_index(drop=True)
X=df[feat].values; y=df['Agent'].values
SC=[]
for tr,te in LeaveOneOut().split(X):
    sc=StandardScaler().fit(X[tr]); SC.append((sc.transform(X[tr]),sc.transform(X[te]),tr,te[0]))
def looacc(lab):
    ok=0
    for Xtr,Xte,tr,t0 in SC:
        m=OneVsRestClassifier(SVC(kernel='rbf',C=10.0,gamma='scale'),n_jobs=1)
        m.fit(Xtr,lab[tr]); ok+= m.predict(Xte)[0]==lab[t0]
    return ok/68
def one(s): return looacc(np.random.default_rng(s).permutation(y))
NP=120
out=np.array(Parallel(n_jobs=2)(delayed(one)(9000+i) for i in range(NP)))
R=json.load(open('results/classification/full1.json'))
obs=R['flat17']['acc']; p=(np.sum(out>=obs)+1)/(NP+1); z=(obs-out.mean())/out.std(ddof=1)
R['perm_flat17']=dict(obs=float(obs),null_mean=float(out.mean()),null_sd=float(out.std(ddof=1)),
   null_max=float(out.max()),z=float(z),p=float(p),n_perm=NP,level='sample (anti-conservative)')
json.dump(R,open('results/classification/full1.json','w'),indent=1,default=float)
np.save('results/classification/null17s.npy',out)
print(f"17-class SAMPLE-level: null {out.mean()*100:.1f}+/-{out.std(ddof=1)*100:.1f}%  max {out.max()*100:.1f}%  obs {obs*100:.1f}%  z={z:.1f}  p={p:.4f}  n={NP}")
