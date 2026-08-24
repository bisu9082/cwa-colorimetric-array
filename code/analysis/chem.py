# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem, DataStructs
import json, itertools, numpy as np

# ---- A-series: PubChem/DFT-verified structures (Table S2) ----
AGENTS = {
 'DMMP': ('COP(C)(=O)OC','d1'),
 'GA'  : ('CCOP(=O)(C#N)N(C)C','d1'),
 'GB'  : ('CC(C)OP(C)(=O)F','d1'),
 'GD'  : ('CC(C)(C)C(C)OP(C)(=O)F','d1'),
 'GF'  : ('O=P(C)(F)OC1CCCCC1','d1'),
 'VX'  : ('CCOP(C)(=O)SCCN(C(C)C)C(C)C','d1'),
 'AC'  : ('C#N','d2'),
 'CG'  : ('O=C(Cl)Cl','d2'),
 'CK'  : ('ClC#N','d2'),
 'HD'  : ('ClCCSCCCl','d2'),
 'HN'  : ('ClCCN(CCCl)CCCl','d2'),
 'L'   : ('Cl/C=C/[As](Cl)Cl','d2'),
 'PS'  : ('ClC(Cl)(Cl)[N+](=O)[O-]','d2'),
 # CORRECTED A-series
 'A-230': ('CP(=O)(F)/N=C(/C)N(CC)CC','d3'),
 'A-232': ('FP(=O)(OC)/N=C(/C)N(CC)CC','d3'),
 'A-234': ('FP(=O)(OCC)/N=C(/C)N(CC)CC','d3'),
 'A-242': ('CP(=O)(F)N=C(N(CC)CC)N(CC)CC','d3'),
}
EXPECT_MW = {'A-230':194.19,'A-232':210.19,'A-234':224.22,'A-242':251.29}

rows={}
for k,(smi,ser) in AGENTS.items():
    m=Chem.MolFromSmiles(smi)
    assert m is not None, k
    rows[k]=dict(series=ser, smiles=smi,
        formula=rdMolDescriptors.CalcMolFormula(m),
        MW=round(Descriptors.MolWt(m),2),
        exact=round(Descriptors.ExactMolWt(m),4),
        XLogP=round(Crippen.MolLogP(m),3),
        TPSA=round(rdMolDescriptors.CalcTPSA(m),2),
        HBD=rdMolDescriptors.CalcNumHBD(m),
        HBA=rdMolDescriptors.CalcNumHBA(m),
        rotB=rdMolDescriptors.CalcNumRotatableBonds(m),
        heavy=m.GetNumHeavyAtoms())
print(f"{'agent':<7}{'formula':<16}{'MW':>8}{'exact':>11}{'XLogP':>8}{'TPSA':>8}{'HBD':>5}{'HBA':>5}  check")
for k,v in rows.items():
    chk=''
    if k in EXPECT_MW:
        d=abs(v['MW']-EXPECT_MW[k]); chk = f"TableS2 {EXPECT_MW[k]} -> {'OK' if d<0.1 else 'MISMATCH'}"
    print(f"{k:<7}{v['formula']:<16}{v['MW']:>8}{v['exact']:>11}{v['XLogP']:>8}{v['TPSA']:>8}{v['HBD']:>5}{v['HBA']:>5}  {chk}")

# Tanimoto (Morgan r=2, 2048)
gen=AllChem.GetMorganGenerator(radius=2,fpSize=2048)
fps={k:gen.GetFingerprint(Chem.MolFromSmiles(v['smiles'])) for k,v in rows.items()}
keys=list(rows)
sim={a:{b:round(DataStructs.TanimotoSimilarity(fps[a],fps[b]),3) for b in keys} for a in keys}
d3=['A-230','A-232','A-234','A-242']
print("\n=== d3 internal Tanimoto (corrected structures) ===")
for a in d3: print("  ",a, {b:sim[a][b] for b in d3 if b!=a})
off=[sim[a][b] for a,b in itertools.combinations(d3,2)]
print(f"  mean within-d3 = {np.mean(off):.3f}")
a242=[sim['A-242'][b] for b in d3 if b!='A-242']
print(f"  A-242 vs other A-series = {a242}  mean {np.mean(a242):.3f}")
oth=[sim[x][y] for x,y in itertools.combinations(['A-230','A-232','A-234'],2)]
print(f"  amidine trio internal   = {oth}  mean {np.mean(oth):.3f}")
json.dump({'agents':rows,'tanimoto':sim},open('results/classification/chem_corrected.json','w'),indent=1)
