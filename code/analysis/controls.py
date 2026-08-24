import numpy as np, pandas as pd, itertools, json
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
df=pd.read_csv("data/processed/integrated_17compounds.csv")
feat=[c for c in df.columns if c.startswith('Dye')]
df=df[df['Concentration']>0].reset_index(drop=True)
X=df[feat].values; y=df['Agent'].values
SER={**{a:'d1' for a in ['DMMP','GA','GB','GD','GF','VX']},
     **{a:'d2' for a in ['AC','CG','CK','HD','HN','L','PS']},
     **{a:'d3' for a in ['A-230','A-232','A-234','A-242']}}
lab=np.array([SER[a] for a in y]); comps=sorted(set(y))
def locompo(Xu,C=10.0,g='scale'):
    ok=0
    for c in comps:
        te=(y==c); tr=~te
        sc=StandardScaler().fit(Xu[tr])
        m=OneVsRestClassifier(SVC(kernel='rbf',C=C,gamma=g,random_state=42))
        m.fit(sc.transform(Xu[tr]),lab[tr])
        ok+=int((m.predict(sc.transform(Xu[te]))==lab[te]).sum())
    return ok
print("[1] 전체 58채널 LOCompO         :", locompo(X), "/68")
# (14) magnitude-only
mag=X.mean(axis=1).reshape(-1,1)
print("[2] 총 dE 크기 1채널만 LOCompO  :", locompo(mag), "/68   (다수 baseline 28/68)")
# (15) 4-dye subset (Anthracene1, Pyrene2, MethylOrange8, BromophenolBlue16) x2 illum
sub=[f"Dye{i}_{j}" for i in [1,2,8,16] for j in "NU"]
Xs=df[sub].values
print("[3] 4염료(8채널) LOCompO        :", locompo(Xs), "/68")
# (16) hyperparameter window at LOCompO
res=[]
for C in [0.1,1,10,100,1000]:
    for g in ['scale',1e-3,1e-2,1e-1]:
        res.append((locompo(X,C,g),C,g))
res.sort(reverse=True)
print("[4] LOCompO 하이퍼파라미터 창   : 최대", res[0], " 최소", res[-1], " 보고설정(10,scale)=", locompo(X))
json.dump({"full":locompo(X),"magnitude_only":locompo(mag),"subset4":locompo(Xs),
           "hp_max":res[0][0],"hp_min":res[-1][0],"hp_argmax":str(res[0][1:])},open("/tmp/r3v.json","w"))
