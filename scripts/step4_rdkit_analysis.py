#!/usr/bin/env python3
"""
Step 4 Extension: RDKit-Based Cheminformatics Analysis
═══════════════════════════════════════════════════════

Objective: 17 CWA agents × 29 dyes의 분자 특성 기반 화학적 해석
  - Agent/Dye molecular descriptors (RDKit)
  - Agent-Dye interaction descriptor matrix
  - Correlation between molecular properties and ΔE sensor responses
  - Chemical interpretation of ML classification results
  - Publication-quality figures (Fig 5~6)

Author: AutoResearchClaw Pipeline v5.0
Date: 2026-04-12
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem, rdMolDescriptors
from rdkit.Chem import Draw, Lipinski, MolSurf
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "figures")
UPLOADS_DIR = os.path.join(BASE_DIR, "mnt", "uploads")

if not os.path.exists(RAW_DIR):
    RAW_DIR = UPLOADS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: SMILES DATABASE — 17 Agents + 29 Dyes
# ════════════════════════════════════════════════════════════════════════════

AGENTS = {
    # d1: Organophosphorus nerve agents
    'DMMP':  {'smiles': 'CP(=O)(OC)OC', 'series': 'd1',
              'fullname': 'Dimethyl methylphosphonate', 'type': 'OP simulant'},
    'GA':    {'smiles': 'CCOP(=O)(C#N)N(C)C', 'series': 'd1',
              'fullname': 'Tabun', 'type': 'G-series nerve agent'},
    'GB':    {'smiles': 'CC(C)OP(=O)(C)F', 'series': 'd1',
              'fullname': 'Sarin', 'type': 'G-series nerve agent'},
    'GD':    {'smiles': 'CC(OP(=O)(C)F)C(C)(C)C', 'series': 'd1',
              'fullname': 'Soman', 'type': 'G-series nerve agent'},
    'GF':    {'smiles': 'C1CCCCC1OP(=O)(C)F', 'series': 'd1',
              'fullname': 'Cyclosarin', 'type': 'G-series nerve agent'},
    'VX':    {'smiles': 'CCOP(=O)(C)SCCN(C(C)C)C(C)C', 'series': 'd1',
              'fullname': 'VX', 'type': 'V-series nerve agent'},

    # d2: Toxic industrial chemicals / blister / blood agents
    'AC':    {'smiles': 'C#N', 'series': 'd2',
              'fullname': 'Hydrogen cyanide', 'type': 'Blood agent'},
    'CG':    {'smiles': 'O=C(Cl)Cl', 'series': 'd2',
              'fullname': 'Phosgene', 'type': 'Choking agent'},
    'CK':    {'smiles': 'ClC#N', 'series': 'd2',
              'fullname': 'Cyanogen chloride', 'type': 'Blood agent'},
    'HD':    {'smiles': 'ClCCSCCCl', 'series': 'd2',
              'fullname': 'Sulfur mustard', 'type': 'Blister agent'},
    'HN':    {'smiles': 'ClCCN(CCCl)CCCl', 'series': 'd2',
              'fullname': 'Nitrogen mustard (HN-3)', 'type': 'Blister agent'},
    'L':     {'smiles': 'Cl/C=C/[As](Cl)Cl', 'series': 'd2',
              'fullname': 'Lewisite', 'type': 'Blister agent'},
    'PS':    {'smiles': 'O=[N+]([O-])C(Cl)(Cl)Cl', 'series': 'd2',
              'fullname': 'Chloropicrin', 'type': 'Lacrimator'},

    # d3: Novichok-class agents
    'A-230': {'smiles': 'CP(=O)(F)/N=C(\\C)/N(CC)CC', 'series': 'd3',
              'fullname': 'Novichok A-230', 'type': 'Novichok agent'},
    'A-232': {'smiles': 'CCOP(=O)(F)/N=C(\\C)/N(CC)CC', 'series': 'd3',
              'fullname': 'Novichok A-232', 'type': 'Novichok agent'},
    'A-234': {'smiles': 'CC(C)OP(=O)(F)/N=C(\\C)/N(CC)CC', 'series': 'd3',
              'fullname': 'Novichok A-234', 'type': 'Novichok agent'},
    'A-242': {'smiles': 'CC(C)(C)OP(=O)(F)/N=C(\\C)/N(CC)CC', 'series': 'd3',
              'fullname': 'Novichok A-242', 'type': 'Novichok agent'},
}

DYES = {
    'Dye1':  {'name': 'Anthracene', 'cas': '120-12-7',
              'smiles': 'c1ccc2cc3ccccc3cc2c1', 'solvent': 'THF',
              'class': 'PAH fluorophore'},
    'Dye2':  {'name': 'Pyrene', 'cas': '129-00-0',
              'smiles': 'c1cc2ccc3cccc4ccc(c1)c2c34', 'solvent': 'THF',
              'class': 'PAH fluorophore'},
    'Dye3':  {'name': 'Allura Red AC', 'cas': '25956-17-6',
              'smiles': 'COc1cc(C)c(cc1/N=N/c1c(O)ccc2cc(ccc12)S(=O)(=O)O)S(=O)(=O)O',
              'solvent': 'DW', 'class': 'Azo dye'},
    'Dye4':  {'name': 'Quinine', 'cas': '130-95-0',
              'smiles': 'COc1ccc2nccc(C(O)C3CC4CCN3CC4=C)c2c1', 'solvent': 'THF',
              'class': 'Alkaloid fluorophore'},
    'Dye5':  {'name': 'Rhodamine B', 'cas': '81-88-9',
              'smiles': 'CCN(CC)c1ccc2c(c1)Oc1cc(N(CC)CC)ccc1C2c1ccccc1C(=O)O',
              'solvent': 'EtOH', 'class': 'Xanthene dye'},
    'Dye6':  {'name': 'Methyl Blue', 'cas': '28983-56-4',
              'smiles': 'c1ccc(cc1)C(=C2C=CC(=[NH+]c3ccccc3)C=C2)c4ccc(cc4)N(c5ccccc5)c6ccc(cc6)S(=O)(=O)[O-]',
              'solvent': 'EtOH', 'class': 'Triarylmethane dye'},
    'Dye7':  {'name': 'Fluorescein', 'cas': '2321-07-5',
              'smiles': 'OC(=O)c1ccccc1-c1c2ccc(O)cc2oc2cc(O)ccc12',
              'solvent': 'THF', 'class': 'Xanthene dye'},
    'Dye8':  {'name': 'Methyl Orange', 'cas': '547-58-0',
              'smiles': 'CN(C)c1ccc(/N=N/c2ccc(cc2)S(=O)(=O)[O-])cc1',
              'solvent': 'DW', 'class': 'Azo indicator'},
    'Dye9':  {'name': 'Nile Red', 'cas': '7385-67-3',
              'smiles': 'CCN(CC)c1ccc2nc3c(ccc(=O)c3oc2c1)C',
              'solvent': 'DMSO', 'class': 'Oxazine dye'},
    'Dye10': {'name': 'Safranin O', 'cas': '477-73-6',
              'smiles': 'Cc1cc2nc3cc(C)c(N)cc3[n+]c2cc1N',
              'solvent': 'EtOH', 'class': 'Phenazine dye'},
    'Dye11': {'name': 'Toluidine Blue O', 'cas': '92-31-9',
              'smiles': 'Cc1cc2nc3ccc(N(C)C)cc3[s+]c2cc1N',
              'solvent': 'DW', 'class': 'Thiazine dye'},
    'Dye12': {'name': 'Nile Blue A', 'cas': '3625-57-8',
              'smiles': 'CCN(CC)c1ccc2nc3ccc(cc3[o+]c2c1)N',
              'solvent': 'DMSO', 'class': 'Oxazine dye'},
    'Dye13': {'name': 'Cresol Red', 'cas': '1733-12-6',
              'smiles': 'Cc1cc(C2(c3cc(C)c(O)cc3S2(=O)=O)C)cc(c1O)C',
              'solvent': 'DMSO', 'class': 'Sulfonephthalein indicator'},
    'Dye14': {'name': 'Eriochrome Black T', 'cas': '1787-61-7',
              'smiles': 'Oc1ccc2cc(cc(c2c1/N=N/c3cc4ccccc4cc3O)S(=O)(=O)[O-])[N+](=O)[O-]',
              'solvent': 'DMSO', 'class': 'Azo indicator'},
    'Dye15': {'name': 'Quinoline Yellow', 'cas': '8004-92-0',
              'smiles': 'OC(=O)c1cc(cc2ccc3ccccc3[n+]12)c1cc(C(=O)O)c2ccc3ccccc3[n+]12',
              'solvent': 'DW', 'class': 'Quinoline dye'},
    'Dye16': {'name': 'Bromophenol Blue', 'cas': '115-39-9',
              'smiles': 'OC1=CC(=C(c2cc(Br)c(O)c(Br)c2)c2ccccc2S1(=O)=O)Br',
              'solvent': 'DMSO', 'class': 'Sulfonephthalein indicator'},
    'Dye17': {'name': 'Neutral Red', 'cas': '553-24-2',
              'smiles': 'Cc1cc2nc3ccc(N(C)C)cc3[n+]c2cc1N',
              'solvent': 'DMSO', 'class': 'Phenazine dye'},
    'Dye18': {'name': 'Eosin Y', 'cas': '17372-87-1',
              'smiles': 'OC(=O)c1ccccc1-c1c2cc(Br)c(=O)c(Br)c2oc2c(Br)c([O-])c(Br)cc12',
              'solvent': 'DW', 'class': 'Xanthene dye'},
    'Dye19': {'name': 'Indigo carmine', 'cas': '860-22-0',
              'smiles': 'O=C1/C(=C2\\Nc3ccc(cc3C2=O)S(=O)(=O)[O-])/Nc2ccc(cc12)S(=O)(=O)[O-]',
              'solvent': 'DMSO', 'class': 'Indigoid dye'},
    'Dye20': {'name': 'Phenol Red', 'cas': '143-74-8',
              'smiles': 'OC1=CC=C(C(c2ccc(O)cc2)c2ccccc2S1(=O)=O)C=C1',
              'solvent': 'DMSO', 'class': 'Sulfonephthalein indicator'},
    'Dye21': {'name': 'Crystal Violet', 'cas': '548-62-9',
              'smiles': 'CN(C)c1ccc(cc1)C(=C1C=CC(=[N+](C)C)C=C1)c1ccc(cc1)N(C)C',
              'solvent': 'DW', 'class': 'Triarylmethane dye'},
    'Dye22': {'name': 'L-Glutathione (reduced)', 'cas': '70-18-8',
              'smiles': 'NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O',
              'solvent': 'DW', 'class': 'Thiol nucleophile'},
    'Dye23': {'name': '2,7-Dibromofluorene', 'cas': '16433-88-8',
              'smiles': 'Brc1ccc2c(c1)Cc1cc(Br)ccc1-2',
              'solvent': 'Toluene', 'class': 'Fluorene derivative'},
    'Dye24': {'name': 'Ethyl viologen dibromide', 'cas': '53721-12-3',
              'smiles': 'CC[n+]1ccc(-c2cc[n+](CC)cc2)cc1',
              'solvent': 'DW', 'class': 'Viologen redox indicator'},
    'Dye25': {'name': "4,4'-Dihydroxybenzophenone", 'cas': '611-99-4',
              'smiles': 'Oc1ccc(cc1)C(=O)c1ccc(O)cc1',
              'solvent': 'DMSO', 'class': 'Benzophenone'},
    'Dye26': {'name': '2,4-Dinitrophenylhydrazine', 'cas': '119-26-6',
              'smiles': 'NNc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]',
              'solvent': 'DMSO', 'class': 'Hydrazine reagent'},
    'Dye27': {'name': '2,5-Diaminobenzene-1,4-dithiol diHCl', 'cas': '75464-52-7',
              'smiles': 'Nc1cc(S)c(N)cc1S',
              'solvent': 'DMSO', 'class': 'Thiol/amine nucleophile'},
    'Dye28': {'name': '2-Hydrazinobenzothiazole', 'cas': '615-21-4',
              'smiles': 'NNc1nc2ccccc2s1',
              'solvent': 'DMSO', 'class': 'Hydrazine reagent'},
    'Dye29': {'name': 'Rhodamine 6G', 'cas': '989-38-8',
              'smiles': 'CCNc1cc2Oc3cc(NCC)c(C)cc3C(c2cc1C)c1ccccc1C(=O)OCC',
              'solvent': 'MeOH', 'class': 'Xanthene dye'},
}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: MOLECULAR DESCRIPTOR CALCULATION
# ════════════════════════════════════════════════════════════════════════════

DESCRIPTOR_LIST = [
    ('MW', Descriptors.MolWt),
    ('LogP', Crippen.MolLogP),
    ('TPSA', Descriptors.TPSA),
    ('HBA', Descriptors.NumHAcceptors),
    ('HBD', Descriptors.NumHDonors),
    ('RotBonds', Descriptors.NumRotatableBonds),
    ('AromaticRings', Descriptors.NumAromaticRings),
    ('HeavyAtoms', Descriptors.HeavyAtomCount),
    ('FractionCSP3', Descriptors.FractionCSP3),
    ('MR', Crippen.MolMR),  # Molar refractivity
    ('LabuteASA', MolSurf.LabuteASA),  # Labute ASA
    ('NumHeteroatoms', rdMolDescriptors.CalcNumHeteroatoms),
    ('NumRings', Descriptors.RingCount),
    ('BertzCT', Descriptors.BertzCT),  # Topological complexity
]


def compute_descriptors(smiles_dict, label_key='smiles'):
    """Compute RDKit descriptors for a dictionary of compounds."""
    records = []
    valid_names = []
    mols = {}

    for name, info in smiles_dict.items():
        smi = info[label_key] if isinstance(info, dict) else info
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  ⚠ Cannot parse SMILES for {name}: {smi}")
            # Create a minimal record with NaN
            record = {'Name': name}
            for desc_name, _ in DESCRIPTOR_LIST:
                record[desc_name] = np.nan
            records.append(record)
            valid_names.append(name)
            continue

        mols[name] = mol
        record = {'Name': name}
        for desc_name, desc_func in DESCRIPTOR_LIST:
            try:
                record[desc_name] = float(desc_func(mol))
            except:
                record[desc_name] = np.nan
        records.append(record)
        valid_names.append(name)

    df = pd.DataFrame(records).set_index('Name')
    return df, mols


def compute_fingerprints(mols_dict, fp_type='morgan', radius=2, nBits=1024):
    """Compute molecular fingerprints."""
    fps = {}
    for name, mol in mols_dict.items():
        if fp_type == 'morgan':
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            fp = gen.GetFingerprint(mol)
        fps[name] = fp
    return fps


def compute_tanimoto_matrix(fps):
    """Compute pairwise Tanimoto similarity matrix."""
    names = list(fps.keys())
    n = len(names)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[names[i]], fps[names[j]])
    return pd.DataFrame(sim_matrix, index=names, columns=names)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: AGENT-DYE INTERACTION DESCRIPTORS
# ════════════════════════════════════════════════════════════════════════════

def compute_interaction_descriptors(agent_desc, dye_desc):
    """
    Compute agent-dye interaction descriptors capturing complementarity:
      - ΔLogP: polarity mismatch (drives partitioning/binding)
      - ΔMW: size ratio
      - HBD-HBA complementarity: agent HBD × dye HBA + agent HBA × dye HBD
      - ΔTPSA: polar surface area difference
      - Combined nucleophilicity index
    """
    print("\n  Computing agent-dye interaction descriptors...")

    interactions = {}
    for agent_name in agent_desc.index:
        for dye_name in dye_desc.index:
            a = agent_desc.loc[agent_name]
            d = dye_desc.loc[dye_name]

            key = (agent_name, dye_name)
            interactions[key] = {
                'delta_LogP': abs(a['LogP'] - d['LogP']),
                'sum_LogP': a['LogP'] + d['LogP'],
                'HB_complementarity': a['HBD'] * d['HBA'] + a['HBA'] * d['HBD'],
                'delta_TPSA': abs(a['TPSA'] - d['TPSA']),
                'sum_TPSA': a['TPSA'] + d['TPSA'],
                'MW_ratio': a['MW'] / d['MW'] if d['MW'] > 0 else 0,
                'delta_MR': abs(a['MR'] - d['MR']),
                'complexity_product': a['BertzCT'] * d['BertzCT'],
                'heteroatom_sum': a['NumHeteroatoms'] + d['NumHeteroatoms'],
                'agent_MW': a['MW'],
                'agent_LogP': a['LogP'],
                'agent_TPSA': a['TPSA'],
                'dye_MW': d['MW'],
                'dye_LogP': d['LogP'],
                'dye_TPSA': d['TPSA'],
            }

    df = pd.DataFrame(interactions).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=['Agent', 'Dye'])
    print(f"  ✓ Computed {len(df)} agent-dye interaction pairs")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: LOAD SENSOR DATA AND CORRELATE
# ════════════════════════════════════════════════════════════════════════════

COMPOUND_NAME_MAP = {'A': 'A-230', 'B': 'A-232', 'C': 'A-234', 'D': 'A-242'}

def load_sensor_data():
    """Load ΔE sensor response data."""
    datasets = {}
    for prefix in ['d1', 'd2', 'd3']:
        normal = pd.read_csv(os.path.join(RAW_DIR, f"{prefix}.normal_deltaE_pivot.csv"))
        uv = pd.read_csv(os.path.join(RAW_DIR, f"{prefix}.uv_deltaE_pivot.csv"))
        if prefix == 'd3':
            normal['Agent'] = normal['Agent'].map(COMPOUND_NAME_MAP)
            uv['Agent'] = uv['Agent'].map(COMPOUND_NAME_MAP)
        dye_cols = [c for c in normal.columns if c.startswith('Dye')]
        normal_r = normal.rename(columns={c: f"{c}_N" for c in dye_cols})
        uv_r = uv.rename(columns={c: f"{c}_U" for c in dye_cols})
        merged = pd.merge(normal_r, uv_r, on=['Agent', 'Concentration'], how='outer')
        merged['Series'] = prefix
        datasets[prefix] = merged

    df_all = pd.concat(list(datasets.values()), ignore_index=True)
    return df_all


def compute_response_profile(df_all):
    """Compute mean ΔE response profile per agent (max concentration)."""
    max_conc = df_all[df_all['Concentration'] == df_all['Concentration'].max()]
    dye_cols_n = sorted([c for c in df_all.columns if c.startswith('Dye') and c.endswith('_N')])
    dye_cols_u = sorted([c for c in df_all.columns if c.startswith('Dye') and c.endswith('_U')])

    # Normal illumination response profile per agent
    profile_n = max_conc.groupby('Agent')[dye_cols_n].mean()
    profile_u = max_conc.groupby('Agent')[dye_cols_u].mean()

    # Combined (mean of N and U per dye)
    combined_profiles = {}
    for agent in profile_n.index:
        combined = {}
        for i in range(1, 30):
            dye_key = f'Dye{i}'
            val_n = profile_n.loc[agent, f'Dye{i}_N'] if f'Dye{i}_N' in profile_n.columns else 0
            val_u = profile_u.loc[agent, f'Dye{i}_U'] if f'Dye{i}_U' in profile_u.columns else 0
            combined[dye_key] = (val_n + val_u) / 2
        combined_profiles[agent] = combined

    profile_df = pd.DataFrame(combined_profiles).T
    return profile_df, profile_n, profile_u


def correlate_descriptors_with_responses(agent_desc, response_profile):
    """Correlate agent molecular descriptors with sensor ΔE response patterns."""
    print("\n  Correlating molecular descriptors with sensor responses...")

    common_agents = sorted(set(agent_desc.index) & set(response_profile.index))
    desc_sub = agent_desc.loc[common_agents].fillna(0)
    resp_sub = response_profile.loc[common_agents].fillna(0)

    correlations = {}
    for desc_col in desc_sub.columns:
        desc_vals = desc_sub[desc_col].values
        if np.std(desc_vals) < 1e-10:
            continue
        for dye_col in resp_sub.columns:
            resp_vals = resp_sub[dye_col].values
            if np.std(resp_vals) < 1e-10:
                continue
            r, p = stats.pearsonr(desc_vals, resp_vals)
            correlations[(desc_col, dye_col)] = {'r': r, 'p': p}

    corr_df = pd.DataFrame(correlations).T
    corr_df.index = pd.MultiIndex.from_tuples(corr_df.index, names=['Descriptor', 'Dye'])
    print(f"  ✓ Computed {len(corr_df)} descriptor-dye correlations")

    # Reshape to matrix
    desc_names = desc_sub.columns.tolist()
    dye_names = resp_sub.columns.tolist()
    r_matrix = np.zeros((len(desc_names), len(dye_names)))
    p_matrix = np.zeros((len(desc_names), len(dye_names)))

    for i, desc in enumerate(desc_names):
        for j, dye in enumerate(dye_names):
            key = (desc, dye)
            if key in correlations:
                r_matrix[i, j] = correlations[key]['r']
                p_matrix[i, j] = correlations[key]['p']

    r_df = pd.DataFrame(r_matrix, index=desc_names, columns=dye_names)
    p_df = pd.DataFrame(p_matrix, index=desc_names, columns=dye_names)

    return r_df, p_df, corr_df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: CHEMICAL INTERPRETATION OF ML RESULTS
# ════════════════════════════════════════════════════════════════════════════

def interpret_ml_results(agent_desc, dye_desc, response_profile, ml_results):
    """
    Integrate ML feature importance with chemical descriptors.
    Identify: WHY certain dyes are discriminative for certain agents.
    """
    print("\n" + "═"*70)
    print("  SECTION: CHEMICAL INTERPRETATION OF ML RESULTS")
    print("═"*70)

    interpretations = {}

    # 1. Top features from ML → which dyes? which illumination?
    feat_imp = ml_results.get('proposed_method', {}).get('feature_importances', {})
    if not feat_imp:
        print("  ⚠ No feature importances found in ML results")
        return {}

    top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"\n  Top 15 ML features and their chemical context:")

    for rank, (feat_name, importance) in enumerate(top_features, 1):
        # Parse feature name: e.g., "Dye21_U" → Dye21, UV illumination
        parts = feat_name.rsplit('_', 1)
        dye_key = parts[0]
        illum = 'UV' if parts[1] == 'U' else 'Normal'
        dye_info = DYES.get(dye_key, {})
        dye_name = dye_info.get('name', dye_key)
        dye_class = dye_info.get('class', 'Unknown')

        # Check which agents show strongest response on this dye
        if dye_key in response_profile.columns:
            responses = response_profile[dye_key].sort_values(ascending=False)
            top_responders = responses.head(3)
        else:
            top_responders = pd.Series()

        interpretations[feat_name] = {
            'rank': rank,
            'importance': importance,
            'dye_name': dye_name,
            'dye_class': dye_class,
            'illumination': illum,
            'top_responding_agents': top_responders.to_dict() if len(top_responders) > 0 else {},
        }

        resp_str = ', '.join([f"{a}({v:.1f})" for a, v in top_responders.items()]) if len(top_responders) > 0 else 'N/A'
        print(f"    {rank:2d}. {feat_name} (imp={importance:.4f})")
        print(f"        Dye: {dye_name} [{dye_class}] | Illumination: {illum}")
        print(f"        Top responders: {resp_str}")

    # 2. Classification performance by agent type → chemical logic
    det_metrics = ml_results.get('detailed_metrics', {})
    class_report = det_metrics.get('classification_report', {})

    print(f"\n  Classification performance by chemical category:")
    categories = {
        'G-series nerve': ['GA', 'GB', 'GD', 'GF'],
        'V-series nerve': ['VX'],
        'OP simulant': ['DMMP'],
        'Blood agents': ['AC', 'CK'],
        'Blister agents': ['HD', 'HN', 'L'],
        'Lacrimator': ['PS'],
        'Novichok': ['A-230', 'A-232', 'A-234', 'A-242'],
    }

    for cat, members in categories.items():
        f1_scores = []
        for m in members:
            if m in class_report:
                f1_scores.append(class_report[m].get('f1-score', 0))
        if f1_scores:
            mean_f1 = np.mean(f1_scores)
            print(f"    {cat:20s}: F1 = {mean_f1:.3f} (n={len(members)})")

    # 3. Dye class effectiveness analysis
    print(f"\n  Dye class contribution to classification:")
    dye_class_importance = {}
    for feat_name, imp in feat_imp.items():
        dye_key = feat_name.rsplit('_', 1)[0]
        dye_class = DYES.get(dye_key, {}).get('class', 'Unknown')
        if dye_class not in dye_class_importance:
            dye_class_importance[dye_class] = []
        dye_class_importance[dye_class].append(imp)

    for cls, imps in sorted(dye_class_importance.items(), key=lambda x: -np.sum(x[1])):
        print(f"    {cls:30s}: total_imp = {np.sum(imps):.4f} (n_features={len(imps)})")

    return interpretations


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: FIGURE GENERATION — Fig 5 & Fig 6
# ════════════════════════════════════════════════════════════════════════════

SERIES_COLORS = {'d1': '#2E75B6', 'd2': '#C00000', 'd3': '#548235'}
TYPE_MARKERS = {
    'G-series nerve agent': 'o', 'V-series nerve agent': 's', 'OP simulant': 'D',
    'Blood agent': '^', 'Blister agent': 'v', 'Choking agent': '<',
    'Lacrimator': '>', 'Novichok agent': 'P',
}


def generate_figure5(agent_desc, dye_desc, r_matrix, p_matrix, response_profile, agent_sim):
    """
    Fig 5: Cheminformatics-enhanced sensor array interpretation.
    (a) Agent molecular descriptor heatmap (clustered)
    (b) Descriptor-Dye Pearson correlation heatmap
    (c) Agent chemical space: descriptor-based vs sensor-based
    (d) Agent structural similarity dendrogram
    """
    print("  Generating Figure 5: Cheminformatics interpretation...")

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # ── Panel A: Agent descriptor heatmap ──
    ax_a = fig.add_subplot(gs[0, 0])
    desc_data = agent_desc.copy().fillna(0)
    desc_scaled = pd.DataFrame(
        StandardScaler().fit_transform(desc_data),
        index=desc_data.index, columns=desc_data.columns
    )

    # Color-code rows by series
    row_colors = []
    for agent in desc_scaled.index:
        series = AGENTS.get(agent, {}).get('series', 'd1')
        row_colors.append(SERIES_COLORS.get(series, 'gray'))

    sns.heatmap(desc_scaled, cmap='RdBu_r', center=0, ax=ax_a,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Z-score', 'shrink': 0.7},
                xticklabels=True, yticklabels=True)
    ax_a.set_title('(a) Agent Molecular Descriptor Profiles', fontweight='bold')
    ax_a.set_xlabel('Molecular Descriptor')
    ax_a.set_ylabel('Chemical Agent')
    ax_a.tick_params(axis='x', rotation=45, labelsize=7)
    ax_a.tick_params(axis='y', labelsize=8)

    # Add series color bar on left
    for i, color in enumerate(row_colors):
        ax_a.add_patch(plt.Rectangle((-0.8, i), 0.6, 1, color=color,
                                      transform=ax_a.transData, clip_on=False))

    # ── Panel B: Descriptor ↔ Dye correlation heatmap ──
    ax_b = fig.add_subplot(gs[0, 1])

    # Simplify dye labels
    dye_labels = [DYES.get(c, {}).get('name', c)[:15] for c in r_matrix.columns]
    mask_sig = p_matrix.values > 0.05  # mask non-significant

    sns.heatmap(r_matrix.values, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax_b, mask=mask_sig,
                xticklabels=dye_labels, yticklabels=r_matrix.index,
                linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Pearson r', 'shrink': 0.7})
    ax_b.set_title('(b) Descriptor–Dye Response Correlation\n(p<0.05 shown)', fontweight='bold')
    ax_b.set_xlabel('Dye')
    ax_b.set_ylabel('Descriptor')
    ax_b.tick_params(axis='x', rotation=90, labelsize=6)
    ax_b.tick_params(axis='y', labelsize=7)

    # ── Panel C: Dual chemical space comparison ──
    ax_c = fig.add_subplot(gs[1, 0])

    # Descriptor-based PCA
    desc_pca = PCA(n_components=2)
    X_desc = desc_pca.fit_transform(desc_scaled.fillna(0).values)

    # Sensor-based PCA
    common = sorted(set(desc_scaled.index) & set(response_profile.index))
    resp_scaled = StandardScaler().fit_transform(response_profile.loc[common].fillna(0).values)
    resp_pca = PCA(n_components=2)
    X_resp = resp_pca.fit_transform(resp_scaled)

    for i, agent in enumerate(common):
        series = AGENTS.get(agent, {}).get('series', 'd1')
        color = SERIES_COLORS.get(series, 'gray')
        agent_type = AGENTS.get(agent, {}).get('type', 'Unknown')
        marker = TYPE_MARKERS.get(agent_type, 'o')

        # Descriptor space
        ax_c.scatter(X_desc[i, 0], X_desc[i, 1], c=color, marker=marker,
                    s=80, edgecolors='black', linewidth=0.5, alpha=0.9, zorder=5)
        ax_c.annotate(agent, (X_desc[i, 0], X_desc[i, 1]),
                     fontsize=6, ha='center', va='bottom', xytext=(0, 5),
                     textcoords='offset points')

    ax_c.set_xlabel(f'PC1 ({desc_pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax_c.set_ylabel(f'PC2 ({desc_pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax_c.set_title('(c) Chemical Space: Molecular Descriptor PCA', fontweight='bold')
    ax_c.grid(alpha=0.3)

    # Legend for series
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E75B6', label='d1: Nerve agents', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C00000', label='d2: TIC/Blister', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#548235', label='d3: Novichok', markersize=8),
    ]
    ax_c.legend(handles=legend_elements, fontsize=7, loc='best')

    # ── Panel D: Tanimoto similarity dendrogram ──
    ax_d = fig.add_subplot(gs[1, 1])

    sim_vals = agent_sim.loc[common, common].values
    dist_matrix = 1 - sim_vals
    np.fill_diagonal(dist_matrix, 0)
    dist_condensed = squareform(dist_matrix)
    Z = linkage(dist_condensed, method='ward')

    # Color function by series
    dendrogram(Z, labels=common, ax=ax_d, leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.5 * max(Z[:, 2]))

    ax_d.set_title('(d) Structural Similarity Dendrogram\n(Tanimoto/Morgan FP)', fontweight='bold')
    ax_d.set_ylabel('Distance (1 - Tanimoto)')
    ax_d.tick_params(axis='x', labelsize=7)

    fig.suptitle('Figure 5: Cheminformatics-Enhanced Chemical Agent Characterization',
                 fontsize=14, fontweight='bold', y=1.01)

    fig_path = os.path.join(FIGURES_DIR, 'Fig5_cheminformatics.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {fig_path}")


def generate_figure6(agent_desc, dye_desc, response_profile, interpretations, ml_results):
    """
    Fig 6: ML–Chemistry integrated interpretation.
    (a) Feature importance mapped to dye chemical classes
    (b) Agent molecular property → ML classification performance
    (c) Interaction descriptor heatmap (agent × dye chemical property pairs)
    (d) Summary: mechanism of discrimination
    """
    print("  Generating Figure 6: ML–Chemistry integration...")

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    feat_imp = ml_results.get('proposed_method', {}).get('feature_importances', {})
    class_report = ml_results.get('detailed_metrics', {}).get('classification_report', {})

    # ── Panel A: Feature importance by dye class ──
    ax_a = fig.add_subplot(gs[0, 0])

    dye_class_data = {}
    for feat, imp in feat_imp.items():
        dye_key = feat.rsplit('_', 1)[0]
        illum = feat.rsplit('_', 1)[1]
        dye_class = DYES.get(dye_key, {}).get('class', 'Unknown')
        cat = f"{dye_class}"
        if cat not in dye_class_data:
            dye_class_data[cat] = {'Normal': 0, 'UV': 0, 'count': 0}
        if illum == 'N':
            dye_class_data[cat]['Normal'] += imp
        else:
            dye_class_data[cat]['UV'] += imp
        dye_class_data[cat]['count'] += 1

    classes = sorted(dye_class_data.keys(), key=lambda x: -(dye_class_data[x]['Normal'] + dye_class_data[x]['UV']))
    normal_vals = [dye_class_data[c]['Normal'] for c in classes]
    uv_vals = [dye_class_data[c]['UV'] for c in classes]

    y_pos = range(len(classes))
    ax_a.barh(y_pos, normal_vals, 0.4, label='Normal', color='#2E75B6', alpha=0.8)
    ax_a.barh([y + 0.4 for y in y_pos], uv_vals, 0.4, label='UV', color='#ED7D31', alpha=0.8)
    ax_a.set_yticks([y + 0.2 for y in y_pos])
    ax_a.set_yticklabels([c[:25] for c in classes], fontsize=7)
    ax_a.set_xlabel('Cumulative Feature Importance')
    ax_a.set_title('(a) ML Feature Importance by Dye Chemical Class', fontweight='bold')
    ax_a.legend(fontsize=8)
    ax_a.invert_yaxis()

    # ── Panel B: Agent MW/LogP/TPSA vs F1-score ──
    ax_b = fig.add_subplot(gs[0, 1])

    common_agents = sorted(set(agent_desc.index) & set(class_report.keys()))
    mws = [agent_desc.loc[a, 'MW'] for a in common_agents]
    logps = [agent_desc.loc[a, 'LogP'] for a in common_agents]
    tpsas = [agent_desc.loc[a, 'TPSA'] for a in common_agents]
    f1s = [class_report[a].get('f1-score', 0) for a in common_agents]

    scatter = ax_b.scatter(mws, f1s, c=logps, s=[t*3 + 30 for t in tpsas],
                          cmap='coolwarm', edgecolors='black', linewidth=0.5, alpha=0.8)
    for i, agent in enumerate(common_agents):
        ax_b.annotate(agent, (mws[i], f1s[i]), fontsize=6, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')

    cbar = plt.colorbar(scatter, ax=ax_b, shrink=0.7)
    cbar.set_label('LogP', fontsize=9)
    ax_b.set_xlabel('Molecular Weight (Da)')
    ax_b.set_ylabel('LOO F1-score')
    ax_b.set_title('(b) Molecular Properties vs Classification Performance\n(size=TPSA, color=LogP)',
                   fontweight='bold')
    ax_b.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='F1=0.5')
    ax_b.grid(alpha=0.2)

    # Correlation annotations
    r_mw, p_mw = stats.pearsonr(mws, f1s) if len(mws) > 2 else (0, 1)
    r_logp, p_logp = stats.pearsonr(logps, f1s) if len(logps) > 2 else (0, 1)
    ax_b.text(0.02, 0.98, f'r(MW, F1) = {r_mw:.2f} (p={p_mw:.3f})\n'
              f'r(LogP, F1) = {r_logp:.2f} (p={p_logp:.3f})',
              transform=ax_b.transAxes, fontsize=7, va='top',
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # ── Panel C: Sensor response vs descriptor PCA overlay ──
    ax_c = fig.add_subplot(gs[1, 0])

    # Sensor-based PCA
    common = sorted(set(agent_desc.index) & set(response_profile.index))
    resp_data = response_profile.loc[common].fillna(0)
    resp_scaled = StandardScaler().fit_transform(resp_data.values)
    pca_resp = PCA(n_components=2)
    X_resp = pca_resp.fit_transform(resp_scaled)

    for i, agent in enumerate(common):
        series = AGENTS.get(agent, {}).get('series', 'd1')
        color = SERIES_COLORS.get(series, 'gray')
        f1 = class_report.get(agent, {}).get('f1-score', 0)
        size = max(f1 * 200, 30)
        ax_c.scatter(X_resp[i, 0], X_resp[i, 1], c=color, s=size,
                    edgecolors='black', linewidth=0.5 if f1 > 0.5 else 1.5,
                    alpha=0.9, zorder=5,
                    linestyle='-' if f1 > 0.5 else '--')
        ax_c.annotate(agent, (X_resp[i, 0], X_resp[i, 1]),
                     fontsize=6, ha='center', va='bottom', xytext=(0, 5),
                     textcoords='offset points',
                     fontweight='bold' if f1 > 0.7 else 'normal',
                     color='black' if f1 > 0.5 else 'red')

    ax_c.set_xlabel(f'Sensor PC1 ({pca_resp.explained_variance_ratio_[0]*100:.1f}%)')
    ax_c.set_ylabel(f'Sensor PC2 ({pca_resp.explained_variance_ratio_[1]*100:.1f}%)')
    ax_c.set_title('(c) Sensor Response Space\n(size=F1, red labels=low F1)', fontweight='bold')
    ax_c.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E75B6', label='d1: Nerve agents', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C00000', label='d2: TIC/Blister', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#548235', label='d3: Novichok', markersize=8),
    ]
    ax_c.legend(handles=legend_elements, fontsize=7, loc='best')

    # ── Panel D: Summary mechanism text ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')

    # Compute key insights
    # Best classified agents
    best_agents = sorted([(a, class_report[a]['f1-score']) for a in common_agents], key=lambda x: -x[1])[:5]
    worst_agents = sorted([(a, class_report[a]['f1-score']) for a in common_agents], key=lambda x: x[1])[:5]

    # Dye class ranking
    top_dye_classes = sorted(dye_class_data.items(), key=lambda x: -(x[1]['Normal'] + x[1]['UV']))[:5]

    summary = (
        "CHEMICAL INTERPRETATION SUMMARY\n"
        "─" * 42 + "\n\n"
        "■ Best classified agents (LOO F1):\n"
    )
    for a, f1 in best_agents:
        atype = AGENTS.get(a, {}).get('type', '?')
        summary += f"  {a:8s} F1={f1:.2f}  [{atype}]\n"

    summary += "\n■ Most confused agents:\n"
    for a, f1 in worst_agents:
        atype = AGENTS.get(a, {}).get('type', '?')
        summary += f"  {a:8s} F1={f1:.2f}  [{atype}]\n"

    summary += "\n■ Top dye classes (importance):\n"
    for cls, data in top_dye_classes:
        total = data['Normal'] + data['UV']
        summary += f"  {cls[:28]:28s} {total:.3f}\n"

    summary += (
        "\n■ Key mechanistic insights:\n"
        "  1. Triarylmethane/xanthene dyes\n"
        "     → OP nerve agent discrimination\n"
        "     (nucleophilic addition to P=O)\n"
        "  2. Thiol/hydrazine reagents\n"
        "     → electrophilic agent detection\n"
        "     (HD, HN, L alkylation)\n"
        "  3. pH indicators\n"
        "     → acid/base gas discrimination\n"
        "     (AC/CG/CK/PS)\n"
        "  4. PAH fluorophores (UV mode)\n"
        "     → Novichok fingerprinting\n"
        "     (fluorescence quenching)"
    )

    ax_d.text(0.02, 0.98, summary, transform=ax_d.transAxes,
              fontsize=8, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F4F8',
                       edgecolor='#2E75B6', linewidth=1.5))

    fig.suptitle('Figure 6: ML–Chemistry Integrated Interpretation of Sensor Array Discrimination',
                 fontsize=14, fontweight='bold', y=1.01)

    fig_path = os.path.join(FIGURES_DIR, 'Fig6_ml_chemistry_integration.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {fig_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*70)
    print("█  STEP 4 EXTENSION: RDKit CHEMINFORMATICS ANALYSIS               █")
    print("█  17 Agents × 29 Dyes | Molecular Descriptor Integration         █")
    print("█"*70)

    # ── 1. Compute molecular descriptors ──
    print("\n" + "═"*70)
    print("  SECTION 1: MOLECULAR DESCRIPTORS")
    print("═"*70)

    print("\n  Computing agent descriptors...")
    agent_desc, agent_mols = compute_descriptors(AGENTS)
    print(f"  ✓ {len(agent_desc)} agents: {list(agent_desc.index)}")
    print(agent_desc[['MW', 'LogP', 'TPSA', 'HBA', 'HBD']].to_string())

    print("\n  Computing dye descriptors...")
    dye_desc, dye_mols = compute_descriptors(DYES)
    print(f"  ✓ {len(dye_desc)} dyes parsed")

    # ── 2. Fingerprints & similarity ──
    print("\n  Computing Morgan fingerprints...")
    agent_fps = compute_fingerprints(agent_mols)
    agent_sim = compute_tanimoto_matrix(agent_fps)
    print(f"  ✓ Agent similarity matrix: {agent_sim.shape}")

    dye_fps = compute_fingerprints(dye_mols)
    dye_sim = compute_tanimoto_matrix(dye_fps)
    print(f"  ✓ Dye similarity matrix: {dye_sim.shape}")

    # ── 3. Load sensor data ──
    print("\n" + "═"*70)
    print("  SECTION 2: SENSOR DATA INTEGRATION")
    print("═"*70)

    df_all = load_sensor_data()
    response_profile, resp_n, resp_u = compute_response_profile(df_all)
    print(f"  ✓ Response profile: {response_profile.shape} (agents × dyes)")

    # ── 4. Agent-Dye interaction descriptors ──
    interaction_desc = compute_interaction_descriptors(agent_desc, dye_desc)

    # ── 5. Descriptor–Response correlation ──
    print("\n" + "═"*70)
    print("  SECTION 3: DESCRIPTOR–RESPONSE CORRELATION")
    print("═"*70)

    r_matrix, p_matrix, corr_df = correlate_descriptors_with_responses(
        agent_desc, response_profile
    )

    # Strongest correlations
    print(f"\n  Top 10 strongest descriptor-dye correlations:")
    corr_sorted = corr_df.sort_values('r', key=abs, ascending=False).head(10)
    for idx, row in corr_sorted.iterrows():
        desc, dye = idx
        dye_name = DYES.get(dye, {}).get('name', dye)
        sig = '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
        print(f"    {desc:15s} × {dye_name:20s}: r={row['r']:.3f} (p={row['p']:.4f}) {sig}")

    # ── 6. Load ML results and interpret ──
    print("\n" + "═"*70)
    print("  SECTION 4: ML–CHEMISTRY INTEGRATION")
    print("═"*70)

    ml_results = None
    results_path = os.path.join(RESULTS_DIR, 'step4_full_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            ml_results = json.load(f)
        print("  ✓ ML results loaded")

    interpretations = interpret_ml_results(agent_desc, dye_desc, response_profile, ml_results)

    # ── 7. Generate Figures ──
    print("\n" + "═"*70)
    print("  SECTION 5: FIGURE GENERATION")
    print("═"*70)

    generate_figure5(agent_desc, dye_desc, r_matrix, p_matrix, response_profile, agent_sim)
    generate_figure6(agent_desc, dye_desc, response_profile, interpretations, ml_results)

    # ── 8. Save cheminformatics results ──
    print("\n" + "═"*70)
    print("  SECTION 6: SAVING CHEMINFORMATICS RESULTS")
    print("═"*70)

    chem_results = {
        'agent_descriptors': agent_desc.to_dict(),
        'dye_descriptors': dye_desc.to_dict(),
        'agent_similarity': agent_sim.to_dict(),
        'top_correlations': corr_sorted.reset_index().to_dict(orient='records'),
        'interpretations': interpretations,
        'dye_info': {k: {kk: vv for kk, vv in v.items() if kk != 'smiles'}
                     for k, v in DYES.items()},
        'agent_info': {k: {kk: vv for kk, vv in v.items() if kk != 'smiles'}
                       for k, v in AGENTS.items()},
    }

    chem_path = os.path.join(RESULTS_DIR, 'step4_cheminformatics.json')
    with open(chem_path, 'w', encoding='utf-8') as f:
        json.dump(chem_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ Cheminformatics results: {chem_path}")

    # Save agent descriptor table as CSV
    agent_desc.to_csv(os.path.join(RESULTS_DIR, 'agent_descriptors.csv'))
    dye_desc.to_csv(os.path.join(RESULTS_DIR, 'dye_descriptors.csv'))
    r_matrix.to_csv(os.path.join(RESULTS_DIR, 'descriptor_dye_correlation.csv'))
    print(f"  ✓ Descriptor CSVs saved")

    print("\n" + "█"*70)
    print("█  RDKit CHEMINFORMATICS ANALYSIS COMPLETE                         █")
    print("█"*70)


if __name__ == '__main__':
    main()
