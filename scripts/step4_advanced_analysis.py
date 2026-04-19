#!/usr/bin/env python3
"""
Step 4 Advanced: Maximized Analysis for Top-Tier Publication
════════════════════════════════════════════════════════════

Extensions beyond base ML + RDKit:
  A. SHAP Explainability (global + per-class beeswarm)
  B. Hierarchical Classification (Series → Compound, 2-tier)
  C. Concentration-Dependent Analysis + LOD estimation
  D. Cross-Reactivity Fingerprinting & Selectivity Index
  E. Dye-Agent Reaction Mechanism Proposal (functional group basis)
  F. Publication Figures 7 & 8

Author: AutoResearchClaw Pipeline v5.0
Date: 2026-04-12
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    LeaveOneOut, StratifiedKFold, cross_val_predict, cross_val_score
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score
)
import shap

warnings.filterwarnings('ignore')

# ─── Paths & Config ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "figures")
PROCESSED_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "processed")
UPLOADS_DIR = os.path.join(BASE_DIR, "mnt", "uploads")
if not os.path.exists(RAW_DIR):
    RAW_DIR = UPLOADS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

COMPOUND_MAP = {'A': 'A-230', 'B': 'A-232', 'C': 'A-234', 'D': 'A-242'}
SERIES_MAP = {
    'DMMP': 'd1', 'GA': 'd1', 'GB': 'd1', 'GD': 'd1', 'GF': 'd1', 'VX': 'd1',
    'AC': 'd2', 'CG': 'd2', 'CK': 'd2', 'HD': 'd2', 'HN': 'd2', 'L': 'd2', 'PS': 'd2',
    'A-230': 'd3', 'A-232': 'd3', 'A-234': 'd3', 'A-242': 'd3',
}
SERIES_LABELS = {'d1': 'Nerve Agents (OP)', 'd2': 'TIC/Blister', 'd3': 'Novichok'}
SERIES_COLORS = {'d1': '#2E75B6', 'd2': '#C00000', 'd3': '#548235'}

DYE_NAMES = {
    'Dye1': 'Anthracene', 'Dye2': 'Pyrene', 'Dye3': 'Allura Red AC',
    'Dye4': 'Quinine', 'Dye5': 'Rhodamine B', 'Dye6': 'Methyl Blue',
    'Dye7': 'Fluorescein', 'Dye8': 'Methyl Orange', 'Dye9': 'Nile Red',
    'Dye10': 'Safranin O', 'Dye11': 'Toluidine Blue O', 'Dye12': 'Nile Blue A',
    'Dye13': 'Cresol Red', 'Dye14': 'Eriochrome Black T', 'Dye15': 'Quinoline Yellow',
    'Dye16': 'Bromophenol Blue', 'Dye17': 'Neutral Red', 'Dye18': 'Eosin Y',
    'Dye19': 'Indigo carmine', 'Dye20': 'Phenol Red', 'Dye21': 'Crystal Violet',
    'Dye22': 'L-Glutathione', 'Dye23': '2,7-Dibromofluorene',
    'Dye24': 'Ethyl viologen', 'Dye25': "4,4'-DHBP",
    'Dye26': '2,4-DNPH', 'Dye27': '2,5-DAB-dithiol',
    'Dye28': '2-Hydrazino-BT', 'Dye29': 'Rhodamine 6G',
}

DYE_CLASSES = {
    'Dye1': 'PAH', 'Dye2': 'PAH', 'Dye3': 'Azo', 'Dye4': 'Alkaloid',
    'Dye5': 'Xanthene', 'Dye6': 'Triarylmethane', 'Dye7': 'Xanthene',
    'Dye8': 'Azo', 'Dye9': 'Oxazine', 'Dye10': 'Phenazine',
    'Dye11': 'Thiazine', 'Dye12': 'Oxazine', 'Dye13': 'Sulfonephthalein',
    'Dye14': 'Azo', 'Dye15': 'Quinoline', 'Dye16': 'Sulfonephthalein',
    'Dye17': 'Phenazine', 'Dye18': 'Xanthene', 'Dye19': 'Indigoid',
    'Dye20': 'Sulfonephthalein', 'Dye21': 'Triarylmethane', 'Dye22': 'Thiol',
    'Dye23': 'Fluorene', 'Dye24': 'Viologen', 'Dye25': 'Benzophenone',
    'Dye26': 'Hydrazine', 'Dye27': 'Thiol/Amine', 'Dye28': 'Hydrazine',
    'Dye29': 'Xanthene',
}

# Functional groups for mechanism proposal
AGENT_FG = {
    'DMMP': ['P=O', 'P-O-C'], 'GA': ['P=O', 'P-CN', 'P-N'],
    'GB': ['P=O', 'P-F', 'P-C'], 'GD': ['P=O', 'P-F', 'P-C'],
    'GF': ['P=O', 'P-F', 'P-C'], 'VX': ['P=O', 'P-S', 'P-O', 'N-alkyl'],
    'AC': ['C≡N', 'H-donor'], 'CG': ['C=O', 'C-Cl'],
    'CK': ['C≡N', 'C-Cl'], 'HD': ['S-alkyl', 'C-Cl'],
    'HN': ['N-alkyl', 'C-Cl'], 'L': ['As-Cl', 'C=C'],
    'PS': ['N-O', 'C-Cl'],
    'A-230': ['P=O', 'P-F', 'N=C', 'P-N'], 'A-232': ['P=O', 'P-F', 'N=C', 'P-N', 'P-O'],
    'A-234': ['P=O', 'P-F', 'N=C', 'P-N', 'P-O'], 'A-242': ['P=O', 'P-F', 'N=C', 'P-N', 'P-O'],
}

DYE_REACTIVE_FG = {
    'PAH': ['π-electron donor', 'fluorescence quenching'],
    'Xanthene': ['π-conjugation', 'nucleophilic O', 'fluorescence'],
    'Triarylmethane': ['cationic center', 'electrophilic C+'],
    'Azo': ['N=N chromophore', 'Lewis base N'],
    'Sulfonephthalein': ['pH-sensitive OH', 'SO₃⁻ anion'],
    'Phenazine': ['N-heterocycle', 'H-bond acceptor'],
    'Thiazine': ['S-heterocycle', 'cationic N+'],
    'Oxazine': ['O-heterocycle', 'solvatochromic'],
    'Thiol': ['SH nucleophile', 'redox active'],
    'Thiol/Amine': ['SH nucleophile', 'NH₂ nucleophile'],
    'Hydrazine': ['N-NH₂ nucleophile', 'carbonyl reactive'],
    'Alkaloid': ['fluorescence', 'N-base', 'OH donor'],
    'Quinoline': ['N-heterocycle', 'carboxylate'],
    'Fluorene': ['π-electron', 'halogen interaction'],
    'Viologen': ['redox mediator', 'electron acceptor'],
    'Benzophenone': ['carbonyl', 'phenol OH'],
    'Indigoid': ['H-bond donor/acceptor', 'redox'],
}


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════
def load_all_data():
    """Load data with all concentration levels."""
    datasets = {}
    for prefix in ['d1', 'd2', 'd3']:
        normal = pd.read_csv(os.path.join(RAW_DIR, f"{prefix}.normal_deltaE_pivot.csv"))
        uv = pd.read_csv(os.path.join(RAW_DIR, f"{prefix}.uv_deltaE_pivot.csv"))
        if prefix == 'd3':
            normal['Agent'] = normal['Agent'].map(COMPOUND_MAP)
            uv['Agent'] = uv['Agent'].map(COMPOUND_MAP)
        dye_cols = [c for c in normal.columns if c.startswith('Dye')]
        normal_r = normal.rename(columns={c: f"{c}_N" for c in dye_cols})
        uv_r = uv.rename(columns={c: f"{c}_U" for c in dye_cols})
        merged = pd.merge(normal_r, uv_r, on=['Agent', 'Concentration'], how='outer')
        merged['Series'] = prefix
        datasets[prefix] = merged

    df_all = pd.concat(list(datasets.values()), ignore_index=True)
    feature_cols = sorted([c for c in df_all.columns if c.startswith('Dye')])
    return df_all, feature_cols


# ════════════════════════════════════════════════════════════════
# A. SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════
def run_shap_analysis(X_scaled, y, feature_cols, compound_names):
    """SHAP-based model explainability."""
    print("\n" + "═"*70)
    print("  A. SHAP EXPLAINABILITY ANALYSIS")
    print("═"*70)

    # Train RF proxy for SHAP (tree-based SHAP is fast and exact)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)

    # Use TreeExplainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    # Global feature importance (mean |SHAP|)
    if isinstance(shap_values, list):
        # Multi-class: list of arrays, each (n_samples, n_features)
        shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # Multi-class: ndarray of shape (n_samples, n_features, n_classes)
        shap_abs_mean = np.abs(shap_values).mean(axis=(0, 2))
    else:
        shap_abs_mean = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.Series(shap_abs_mean, index=feature_cols).sort_values(ascending=False)

    print(f"\n  Global SHAP importance (top 15):")
    for feat, val in shap_importance.head(15).items():
        dye_key = feat.rsplit('_', 1)[0]
        illum = 'UV' if feat.endswith('_U') else 'Normal'
        dye_name = DYE_NAMES.get(dye_key, dye_key)
        print(f"    {feat:12s} ({dye_name:20s}, {illum:6s}): {val:.4f}")

    # Per-class SHAP analysis — which features matter for EACH compound?
    per_class_shap = {}
    if isinstance(shap_values, list):
        for cls_idx in range(len(compound_names)):
            cls_shap = np.abs(shap_values[cls_idx]).mean(axis=0)
            top_features = pd.Series(cls_shap, index=feature_cols).sort_values(ascending=False).head(5)
            per_class_shap[compound_names[cls_idx]] = top_features.to_dict()
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        for cls_idx in range(min(shap_values.shape[2], len(compound_names))):
            cls_shap = np.abs(shap_values[:, :, cls_idx]).mean(axis=0)
            top_features = pd.Series(cls_shap, index=feature_cols).sort_values(ascending=False).head(5)
            per_class_shap[compound_names[cls_idx]] = top_features.to_dict()

    print(f"\n  Per-class top discriminative features (SHAP):")
    for agent, feats in per_class_shap.items():
        top_feat = list(feats.keys())[0]
        dye_key = top_feat.rsplit('_', 1)[0]
        print(f"    {agent:8s}: {top_feat} ({DYE_NAMES.get(dye_key, dye_key)})")

    return shap_values, shap_importance, per_class_shap, rf


# ════════════════════════════════════════════════════════════════
# B. HIERARCHICAL CLASSIFICATION
# ════════════════════════════════════════════════════════════════
def run_hierarchical_classification(X_scaled, y, feature_cols, compound_names, df_nonzero):
    """Two-tier: Series (d1/d2/d3) → Compound within series."""
    print("\n" + "═"*70)
    print("  B. HIERARCHICAL CLASSIFICATION")
    print("═"*70)

    agents = df_nonzero['Agent'].values
    series_labels = np.array([SERIES_MAP[a] for a in agents])
    le_series = LabelEncoder()
    y_series = le_series.fit_transform(series_labels)

    results = {}

    # ── Tier 1: Series classification (3 classes) ──
    loo = LeaveOneOut()
    ovr_series = OneVsRestClassifier(SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
    y_pred_series = cross_val_predict(ovr_series, X_scaled, y_series, cv=loo)
    tier1_acc = accuracy_score(y_series, y_pred_series)
    tier1_f1 = f1_score(y_series, y_pred_series, average='macro')

    print(f"\n  Tier 1 — Series classification (3 classes):")
    print(f"    LOO Accuracy: {tier1_acc:.4f} ({int(tier1_acc*len(y_series))}/{len(y_series)})")
    print(f"    LOO F1-macro: {tier1_f1:.4f}")
    print(f"    {classification_report(y_series, y_pred_series, target_names=le_series.classes_)}")

    results['tier1'] = {
        'accuracy': float(tier1_acc), 'f1': float(tier1_f1),
        'classes': list(le_series.classes_),
    }

    # ── Tier 2: Within-series classification ──
    print(f"  Tier 2 — Within-series compound classification:")
    tier2_results = {}

    for series in ['d1', 'd2', 'd3']:
        mask = series_labels == series
        X_s = X_scaled[mask]
        agents_s = agents[mask]
        le_s = LabelEncoder()
        y_s = le_s.fit_transform(agents_s)
        n_classes = len(le_s.classes_)

        if n_classes <= 1:
            continue

        loo_s = LeaveOneOut()
        ovr_s = OneVsRestClassifier(SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
        y_pred_s = cross_val_predict(ovr_s, X_s, y_s, cv=loo_s)
        acc_s = accuracy_score(y_s, y_pred_s)
        f1_s = f1_score(y_s, y_pred_s, average='macro')

        print(f"\n    {series} ({SERIES_LABELS[series]}, {n_classes} compounds):")
        print(f"      LOO Accuracy: {acc_s:.4f} ({int(acc_s*len(y_s))}/{len(y_s)})")
        print(f"      LOO F1-macro: {f1_s:.4f}")
        print(f"      Classes: {list(le_s.classes_)}")

        tier2_results[series] = {
            'accuracy': float(acc_s), 'f1': float(f1_s),
            'n_classes': n_classes, 'classes': list(le_s.classes_),
        }

    results['tier2'] = tier2_results

    # ── Combined hierarchical accuracy ──
    # Simulate: series correct AND compound correct
    correct_count = 0
    for i in range(len(y)):
        series_correct = (y_pred_series[i] == y_series[i])
        if series_correct:
            true_series = le_series.inverse_transform([y_series[i]])[0]
            s_mask = series_labels == true_series
            s_indices = np.where(s_mask)[0]
            local_idx = np.where(s_indices == i)[0][0]

            X_s = X_scaled[s_mask]
            agents_s = agents[s_mask]
            le_s = LabelEncoder()
            y_s = le_s.fit_transform(agents_s)

            # LOO for this sample
            X_train = np.delete(X_s, local_idx, axis=0)
            y_train = np.delete(y_s, local_idx)
            X_test = X_s[local_idx:local_idx+1]
            y_test = y_s[local_idx]

            if len(np.unique(y_train)) > 1:
                ovr_t = OneVsRestClassifier(SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
                ovr_t.fit(X_train, y_train)
                pred_t = ovr_t.predict(X_test)[0]
                if pred_t == y_test:
                    correct_count += 1

    hierarchical_acc = correct_count / len(y)
    results['hierarchical_accuracy'] = float(hierarchical_acc)

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  HIERARCHICAL CLASSIFICATION SUMMARY     ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  Tier 1 (Series):     {tier1_acc:.4f} ({int(tier1_acc*len(y))}/{len(y)})      ║")
    for s, r in tier2_results.items():
        print(f"  ║  Tier 2 ({s}):         {r['accuracy']:.4f} ({r['n_classes']} classes)       ║")
    print(f"  ║  Combined accuracy:   {hierarchical_acc:.4f}                ║")
    print(f"  ║  Flat OvR SVM:        0.6324 (reference)       ║")
    print(f"  ╚══════════════════════════════════════════╝")

    return results


# ════════════════════════════════════════════════════════════════
# C. CONCENTRATION-DEPENDENT ANALYSIS + LOD
# ════════════════════════════════════════════════════════════════
def run_concentration_analysis(df_all, feature_cols):
    """Analyze sensitivity at each concentration level + estimate LOD."""
    print("\n" + "═"*70)
    print("  C. CONCENTRATION-DEPENDENT ANALYSIS & LOD ESTIMATION")
    print("═"*70)

    concentrations = sorted(df_all['Concentration'].unique())
    conc_nonzero = [c for c in concentrations if c > 0]

    results = {'per_concentration': {}, 'lod_estimates': {}}

    # Classification accuracy at each concentration
    print(f"\n  Classification accuracy by concentration:")
    for conc in conc_nonzero:
        df_c = df_all[df_all['Concentration'] == conc]
        X_c = df_c[feature_cols].fillna(0).values
        agents_c = df_c['Agent'].values
        le_c = LabelEncoder()
        y_c = le_c.fit_transform(agents_c)

        scaler = StandardScaler()
        X_c_scaled = scaler.fit_transform(X_c)

        loo = LeaveOneOut()
        ovr = OneVsRestClassifier(SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
        y_pred = cross_val_predict(ovr, X_c_scaled, y_c, cv=loo)
        acc = accuracy_score(y_c, y_pred)
        f1 = f1_score(y_c, y_pred, average='macro')

        results['per_concentration'][int(conc)] = {
            'accuracy': float(acc), 'f1': float(f1),
            'n_samples': len(y_c), 'n_classes': len(le_c.classes_),
        }
        print(f"    {conc:6.0f} µM: Acc={acc:.4f}, F1={f1:.4f} (n={len(y_c)}, classes={len(le_c.classes_)})")

    # LOD estimation: per-agent mean ΔE at each concentration
    print(f"\n  LOD estimation (3σ criterion):")
    baseline = df_all[df_all['Concentration'] == 0]
    dye_cols_n = [c for c in feature_cols if c.endswith('_N')]

    for agent in sorted(df_all['Agent'].unique()):
        baseline_agent = baseline[baseline['Agent'] == agent]
        if len(baseline_agent) == 0:
            continue

        # Baseline noise (σ)
        baseline_vals = baseline_agent[dye_cols_n].values.flatten()
        baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
        if len(baseline_vals) == 0:
            continue
        sigma = np.std(baseline_vals)
        threshold_3sigma = 3 * sigma

        # Mean response at each concentration
        conc_means = {}
        for conc in conc_nonzero:
            agent_conc = df_all[(df_all['Agent'] == agent) & (df_all['Concentration'] == conc)]
            if len(agent_conc) > 0:
                mean_response = agent_conc[dye_cols_n].mean().mean()
                conc_means[conc] = mean_response

        # Find lowest concentration where mean ΔE > 3σ
        lod = None
        for conc in sorted(conc_means.keys()):
            if conc_means[conc] > threshold_3sigma:
                lod = conc
                break

        results['lod_estimates'][agent] = {
            'baseline_sigma': float(sigma),
            'threshold_3sigma': float(threshold_3sigma),
            'lod_uM': float(lod) if lod else None,
            'conc_responses': {str(int(k)): float(v) for k, v in conc_means.items()},
        }

        lod_str = f"{lod:.0f} µM" if lod else ">500 µM"
        print(f"    {agent:8s}: σ={sigma:.2f}, 3σ={threshold_3sigma:.2f}, LOD ≈ {lod_str}")

    return results


# ════════════════════════════════════════════════════════════════
# D. CROSS-REACTIVITY & SELECTIVITY INDEX
# ════════════════════════════════════════════════════════════════
def run_selectivity_analysis(df_all, feature_cols):
    """Compute selectivity index for each dye toward each agent."""
    print("\n" + "═"*70)
    print("  D. CROSS-REACTIVITY & SELECTIVITY INDEX")
    print("═"*70)

    max_conc = df_all[df_all['Concentration'] == df_all[df_all['Concentration'] > 0]['Concentration'].max()]
    agents = sorted(max_conc['Agent'].unique())
    dye_cols_n = sorted([c for c in feature_cols if c.endswith('_N')])

    # Response matrix: agent × dye (Normal, max conc)
    response_matrix = max_conc.groupby('Agent')[dye_cols_n].mean()
    response_matrix.columns = [c.replace('_N', '') for c in response_matrix.columns]

    # Selectivity Index: SI(dye, agent_i) = ΔE(agent_i) / mean(ΔE(other agents))
    selectivity = pd.DataFrame(index=response_matrix.index, columns=response_matrix.columns)
    for dye in response_matrix.columns:
        for agent in response_matrix.index:
            own_response = response_matrix.loc[agent, dye]
            others = response_matrix.loc[response_matrix.index != agent, dye]
            mean_others = others.mean()
            if mean_others > 0.1:
                selectivity.loc[agent, dye] = own_response / mean_others
            else:
                selectivity.loc[agent, dye] = own_response / 0.1

    selectivity = selectivity.astype(float)

    # Top selective dye-agent pairs
    print(f"\n  Top 15 most selective dye-agent pairs (SI > 2.0):")
    flat = selectivity.stack().reset_index()
    flat.columns = ['Agent', 'Dye', 'SI']
    flat = flat.sort_values('SI', ascending=False)

    top_selective = flat.head(15)
    for _, row in top_selective.iterrows():
        dye_name = DYE_NAMES.get(row['Dye'], row['Dye'])
        print(f"    {row['Agent']:8s} × {dye_name:25s}: SI = {row['SI']:.2f}")

    # Cross-reactivity clusters
    print(f"\n  Agents with highest cross-reactivity (lowest selectivity):")
    mean_si = selectivity.mean(axis=1).sort_values()
    for agent, si in mean_si.head(5).items():
        print(f"    {agent:8s}: mean SI = {si:.2f}")

    return selectivity, response_matrix


# ════════════════════════════════════════════════════════════════
# E. REACTION MECHANISM PROPOSAL
# ════════════════════════════════════════════════════════════════
def propose_mechanisms(per_class_shap, selectivity):
    """Propose detection mechanisms based on functional group matching."""
    print("\n" + "═"*70)
    print("  E. PROPOSED DETECTION MECHANISMS")
    print("═"*70)

    mechanisms = {}

    for agent, feats in per_class_shap.items():
        agent_fgs = AGENT_FG.get(agent, [])
        top_feat = list(feats.keys())[0]
        dye_key = top_feat.rsplit('_', 1)[0]
        illum = 'UV' if top_feat.endswith('_U') else 'Normal'
        dye_class = DYE_CLASSES.get(dye_key, 'Unknown')
        dye_reactive = DYE_REACTIVE_FG.get(dye_class, [])
        dye_name = DYE_NAMES.get(dye_key, dye_key)

        # Propose mechanism
        mechanism = 'General interaction'
        if 'P-F' in agent_fgs and 'SH nucleophile' in dye_reactive:
            mechanism = 'Nucleophilic substitution at P-F by thiol'
        elif 'P-F' in agent_fgs and 'N-NH₂ nucleophile' in dye_reactive:
            mechanism = 'Hydrazinolysis of P-F bond'
        elif 'P=O' in agent_fgs and 'pH-sensitive OH' in dye_reactive:
            mechanism = 'Lewis acid-base interaction with P=O'
        elif 'C-Cl' in agent_fgs and 'SH nucleophile' in dye_reactive:
            mechanism = 'Thiol alkylation by C-Cl electrophile'
        elif 'C-Cl' in agent_fgs and 'cationic center' in dye_reactive:
            mechanism = 'Anion exchange / halide displacement'
        elif 'C≡N' in agent_fgs and 'cationic center' in dye_reactive:
            mechanism = 'CN⁻ coordination to cationic dye'
        elif 'C=O' in agent_fgs and 'N-NH₂ nucleophile' in dye_reactive:
            mechanism = 'Hydrazone formation with carbonyl'
        elif 'S-alkyl' in agent_fgs and 'SH nucleophile' in dye_reactive:
            mechanism = 'Thiol-disulfide exchange'
        elif 'As-Cl' in agent_fgs:
            mechanism = 'As(III) coordination to Lewis base'
        elif 'P-S' in agent_fgs and ('SH nucleophile' in dye_reactive or 'redox active' in dye_reactive):
            mechanism = 'Thioether interaction / redox modulation'
        elif 'N=C' in agent_fgs and 'pH-sensitive OH' in dye_reactive:
            mechanism = 'Amidine basicity alters pH indicator'
        elif 'N=C' in agent_fgs and 'fluorescence quenching' in dye_reactive:
            mechanism = 'Electron transfer quenching by amidine'
        elif illum == 'UV' and 'fluorescence' in ' '.join(dye_reactive):
            mechanism = f'Fluorescence modulation ({dye_class} under UV)'
        elif 'P=O' in agent_fgs:
            mechanism = f'Lewis acid-base (P=O ↔ {dye_class})'

        mechanisms[agent] = {
            'top_dye': dye_name,
            'dye_class': dye_class,
            'illumination': illum,
            'agent_functional_groups': agent_fgs,
            'dye_reactive_groups': dye_reactive,
            'proposed_mechanism': mechanism,
        }

        series = SERIES_MAP.get(agent, '?')
        print(f"    {agent:8s} [{series}] → {dye_name} ({dye_class}, {illum})")
        print(f"      Agent FGs: {', '.join(agent_fgs[:3])}")
        print(f"      Dye FGs:   {', '.join(dye_reactive[:2])}")
        print(f"      Mechanism: {mechanism}")
        print()

    return mechanisms


# ════════════════════════════════════════════════════════════════
# F. FIGURE 7: SHAP + Hierarchical + Concentration
# ════════════════════════════════════════════════════════════════
def generate_figure7(shap_values, X_scaled, feature_cols, compound_names,
                     hier_results, conc_results, y):
    """Fig 7: Advanced ML analysis."""
    print("  Generating Figure 7: Advanced ML analysis...")

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # ── Panel A: SHAP beeswarm (global) ──
    ax_a = fig.add_subplot(gs[0, 0])

    # Manual SHAP summary plot
    if isinstance(shap_values, list):
        shap_mean = np.mean(np.abs(np.array(shap_values)), axis=0)
        feat_imp = shap_mean.mean(axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (n_samples, n_features, n_classes)
        shap_mean = np.abs(shap_values).mean(axis=2)  # → (n_samples, n_features)
        feat_imp = shap_mean.mean(axis=0)
    else:
        shap_mean = np.abs(shap_values)
        feat_imp = shap_mean.mean(axis=0)
    top_idx = np.argsort(feat_imp)[-20:][::-1]

    feat_labels = []
    for idx in top_idx:
        fc = feature_cols[idx]
        dye_key = fc.rsplit('_', 1)[0]
        illum = '(UV)' if fc.endswith('_U') else '(N)'
        name = DYE_NAMES.get(dye_key, dye_key)[:12]
        feat_labels.append(f"{name} {illum}")

    y_pos = range(len(top_idx))
    colors = []
    for idx in top_idx:
        fc = feature_cols[idx]
        dye_key = fc.rsplit('_', 1)[0]
        dc = DYE_CLASSES.get(dye_key, 'Unknown')
        cmap = {'Xanthene': '#E74C3C', 'Sulfonephthalein': '#3498DB', 'Triarylmethane': '#2ECC71',
                'PAH': '#9B59B6', 'Azo': '#F39C12', 'Thiol': '#1ABC9C', 'Thiol/Amine': '#1ABC9C',
                'Hydrazine': '#E67E22', 'Thiazine': '#34495E', 'Phenazine': '#95A5A6'}
        colors.append(cmap.get(dc, '#BDC3C7'))

    ax_a.barh(y_pos, feat_imp[top_idx], color=colors, edgecolor='black', linewidth=0.3)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(feat_labels, fontsize=7)
    ax_a.set_xlabel('Mean |SHAP value|')
    ax_a.set_title('(a) SHAP Feature Importance (Top 20)\ncolored by dye class', fontweight='bold')
    ax_a.invert_yaxis()

    # ── Panel B: Hierarchical classification ──
    ax_b = fig.add_subplot(gs[0, 1])

    categories = ['Tier 1\n(Series)', 'Tier 2\n(d1: OP)', 'Tier 2\n(d2: TIC)',
                  'Tier 2\n(d3: Nov)', 'Combined\nHierarchical', 'Flat\nOvR SVM']
    t2 = hier_results.get('tier2', {})
    accs = [
        hier_results['tier1']['accuracy'],
        t2.get('d1', {}).get('accuracy', 0),
        t2.get('d2', {}).get('accuracy', 0),
        t2.get('d3', {}).get('accuracy', 0),
        hier_results['hierarchical_accuracy'],
        0.6324,
    ]
    n_classes = [3, 6, 7, 4, 17, 17]
    bar_colors = ['#2E75B6', '#2E75B6', '#C00000', '#548235', '#8E44AD', '#95A5A6']

    bars = ax_b.bar(range(len(categories)), accs, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_b.set_xticks(range(len(categories)))
    ax_b.set_xticklabels(categories, fontsize=8)
    ax_b.set_ylabel('LOO Accuracy')
    ax_b.set_title('(b) Hierarchical vs Flat Classification', fontweight='bold')
    ax_b.set_ylim(0, 1.05)

    for i, (bar, acc, nc) in enumerate(zip(bars, accs, n_classes)):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.1%}\n({nc}cls)', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax_b.axhline(1/17, color='gray', linestyle=':', alpha=0.5, label='Random (17-class)')
    ax_b.legend(fontsize=7)

    # ── Panel C: Concentration-dependent SIGNAL INTENSITY ──
    ax_c = fig.add_subplot(gs[1, 0])

    # Load full dataset to compute mean ΔE per series per concentration
    df_full = pd.read_csv(os.path.join(PROCESSED_DIR, 'integrated_17compounds.csv'))
    feat_cols_in_data = [c for c in df_full.columns if c.startswith('Dye')]
    conc_nonzero = sorted([c for c in df_full['Concentration'].unique() if c > 0])

    series_styles = {
        'd1': {'color': '#2E75B6', 'marker': 'o', 'label': 'd1 — Nerve Agents (OP)'},
        'd2': {'color': '#C00000', 'marker': 's', 'label': 'd2 — TIC / Blister'},
        'd3': {'color': '#548235', 'marker': 'D', 'label': 'd3 — Novichok'},
    }

    for series_key, style in series_styles.items():
        mean_signals = []
        std_signals = []
        for conc in conc_nonzero:
            mask = (df_full['Concentration'] == conc) & (df_full['Series'] == series_key)
            vals = df_full.loc[mask, feat_cols_in_data].values
            # Mean ΔE across all dye channels per sample, then mean/std across samples
            per_sample_mean = np.mean(vals, axis=1)  # mean across 58 features
            mean_signals.append(np.mean(per_sample_mean))
            std_signals.append(np.std(per_sample_mean))

        mean_signals = np.array(mean_signals)
        std_signals = np.array(std_signals)
        ax_c.errorbar(conc_nonzero, mean_signals, yerr=std_signals,
                      fmt=f'{style["marker"]}-', color=style['color'],
                      linewidth=2, markersize=8, capsize=4, capthick=1.5,
                      label=style['label'], alpha=0.9)
        ax_c.fill_between(conc_nonzero, mean_signals - std_signals,
                          mean_signals + std_signals, alpha=0.08, color=style['color'])

    ax_c.set_xlabel('Concentration (µM)', fontsize=10)
    ax_c.set_ylabel('Mean ΔE (across 58 channels)', fontsize=10)
    ax_c.set_title('(c) Signal Intensity vs Concentration by Series', fontweight='bold')
    ax_c.set_xscale('log')
    ax_c.legend(fontsize=7.5, loc='upper left')
    ax_c.grid(alpha=0.3)
    ax_c.set_xlim(7, 700)

    # Annotate dose-response trend
    ax_c.text(0.97, 0.05, 'Error bars: ±1 SD across agents',
              transform=ax_c.transAxes, fontsize=6.5, ha='right', va='bottom',
              style='italic', color='gray')

    # ── Panel D: LOD summary ──
    ax_d = fig.add_subplot(gs[1, 1])

    lod_data = conc_results['lod_estimates']
    agents_sorted = sorted(lod_data.keys())
    lod_vals = []
    agent_labels = []
    bar_colors_lod = []
    for a in agents_sorted:
        lod = lod_data[a].get('lod_uM')
        if lod is not None:
            lod_vals.append(lod)
        else:
            lod_vals.append(600)  # placeholder for >500
        agent_labels.append(a)
        series = SERIES_MAP.get(a, 'd1')
        bar_colors_lod.append(SERIES_COLORS.get(series, 'gray'))

    ax_d.barh(range(len(agent_labels)), lod_vals, color=bar_colors_lod,
              edgecolor='black', linewidth=0.3)
    ax_d.set_yticks(range(len(agent_labels)))
    ax_d.set_yticklabels(agent_labels, fontsize=8)
    ax_d.set_xlabel('Estimated LOD (µM, 3σ criterion)')
    ax_d.set_title('(d) Limit of Detection by Agent', fontweight='bold')
    ax_d.axvline(10, color='red', linestyle='--', alpha=0.5, label='10 µM')
    ax_d.axvline(50, color='orange', linestyle='--', alpha=0.5, label='50 µM')
    ax_d.legend(fontsize=7)
    ax_d.invert_yaxis()

    fig.suptitle('Figure 7: Advanced ML Analysis — SHAP, Hierarchical Classification, and Sensitivity',
                 fontsize=14, fontweight='bold', y=1.01)

    fig_path = os.path.join(FIGURES_DIR, 'Fig7_advanced_ml.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {fig_path}")


# ════════════════════════════════════════════════════════════════
# G. FIGURE 8: Selectivity + Mechanism
# ════════════════════════════════════════════════════════════════
def generate_figure8(selectivity, mechanisms, per_class_shap, response_matrix):
    """Fig 8: Selectivity fingerprint + proposed mechanisms."""
    print("  Generating Figure 8: Selectivity & mechanism...")

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── Panel A: Selectivity heatmap ──
    ax_a = fig.add_subplot(gs[0, :])

    sel_display = selectivity.copy()
    sel_display.columns = [DYE_NAMES.get(c, c)[:12] for c in sel_display.columns]
    sel_clipped = sel_display.clip(upper=5)

    sns.heatmap(sel_clipped, cmap='YlOrRd', ax=ax_a, linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Selectivity Index (SI)', 'shrink': 0.5},
                xticklabels=True, yticklabels=True)
    ax_a.set_title('(a) Selectivity Index Heatmap (SI = ΔE_agent / mean ΔE_others, Normal, 500 µM)',
                   fontweight='bold')
    ax_a.set_xlabel('Dye')
    ax_a.set_ylabel('Agent')
    ax_a.tick_params(axis='x', rotation=90, labelsize=6)
    ax_a.tick_params(axis='y', labelsize=8)

    # ── Panel B: Per-class SHAP fingerprint ──
    ax_b = fig.add_subplot(gs[1, 0])

    # Create heatmap of per-class top SHAP features
    all_agents = sorted(per_class_shap.keys())
    all_feats = set()
    for feats in per_class_shap.values():
        all_feats.update(feats.keys())
    all_feats = sorted(all_feats)

    shap_matrix = np.zeros((len(all_agents), len(all_feats)))
    for i, agent in enumerate(all_agents):
        for j, feat in enumerate(all_feats):
            shap_matrix[i, j] = per_class_shap.get(agent, {}).get(feat, 0)

    # Only show top 20 features
    feat_sums = shap_matrix.sum(axis=0)
    top_feat_idx = np.argsort(feat_sums)[-20:][::-1]

    feat_labels_b = []
    for idx in top_feat_idx:
        fc = all_feats[idx]
        dye_key = fc.rsplit('_', 1)[0]
        illum = '(UV)' if fc.endswith('_U') else '(N)'
        feat_labels_b.append(f"{DYE_NAMES.get(dye_key, dye_key)[:10]} {illum}")

    sns.heatmap(shap_matrix[:, top_feat_idx], cmap='Reds', ax=ax_b,
                xticklabels=feat_labels_b, yticklabels=all_agents,
                linewidths=0.3, linecolor='white',
                cbar_kws={'label': '|SHAP|', 'shrink': 0.7})
    ax_b.set_title('(b) Per-Agent SHAP Fingerprint (Top 20 Features)', fontweight='bold')
    ax_b.set_xlabel('Feature (Dye × Illumination)')
    ax_b.set_ylabel('Agent')
    ax_b.tick_params(axis='x', rotation=90, labelsize=6)
    ax_b.tick_params(axis='y', labelsize=7)

    # ── Panel C: Mechanism summary ──
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.axis('off')

    mech_text = "PROPOSED DETECTION MECHANISMS\n" + "─" * 44 + "\n\n"

    # Group by mechanism type
    mech_groups = {}
    for agent, info in mechanisms.items():
        mech = info['proposed_mechanism']
        if mech not in mech_groups:
            mech_groups[mech] = []
        mech_groups[mech].append(agent)

    for mech, agents in sorted(mech_groups.items(), key=lambda x: -len(x[1])):
        agent_str = ', '.join(agents)
        mech_text += f"■ {mech}\n"
        mech_text += f"  → {agent_str}\n\n"

    mech_text += "─" * 44 + "\n"
    mech_text += "Key: Dye reactive groups determine\n"
    mech_text += "selectivity via functional group\n"
    mech_text += "complementarity with agent electro-\n"
    mech_text += "philic/nucleophilic centers."

    ax_c.text(0.02, 0.98, mech_text, transform=ax_c.transAxes,
              fontsize=7.5, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E7',
                       edgecolor='#D4A017', linewidth=1.5))

    fig.suptitle('Figure 8: Selectivity Fingerprinting and Proposed Detection Mechanisms',
                 fontsize=14, fontweight='bold', y=1.01)

    fig_path = os.path.join(FIGURES_DIR, 'Fig8_selectivity_mechanism.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {fig_path}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    start_time = time.time()
    print("\n" + "█"*70)
    print("█  STEP 4 ADVANCED: MAXIMIZED ANALYSIS                             █")
    print("█  SHAP + Hierarchical + LOD + Selectivity + Mechanism             █")
    print("█"*70)

    # Load data
    df_all, feature_cols = load_all_data()
    df_nonzero = df_all[df_all['Concentration'] > 0].copy()
    X = df_nonzero[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df_nonzero['Agent'])
    compound_names = le.classes_
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # A. SHAP
    shap_values, shap_importance, per_class_shap, rf_model = \
        run_shap_analysis(X_scaled, y, feature_cols, compound_names)

    # B. Hierarchical
    hier_results = run_hierarchical_classification(X_scaled, y, feature_cols, compound_names, df_nonzero)

    # C. Concentration
    conc_results = run_concentration_analysis(df_all, feature_cols)

    # D. Selectivity
    selectivity, response_matrix = run_selectivity_analysis(df_all, feature_cols)

    # E. Mechanisms
    mechanisms = propose_mechanisms(per_class_shap, selectivity)

    # F. Figures
    print("\n" + "═"*70)
    print("  FIGURE GENERATION")
    print("═"*70)
    generate_figure7(shap_values, X_scaled, feature_cols, compound_names,
                     hier_results, conc_results, y)
    generate_figure8(selectivity, mechanisms, per_class_shap, response_matrix)

    # Save all results
    print("\n" + "═"*70)
    print("  SAVING ADVANCED RESULTS")
    print("═"*70)

    advanced_results = {
        'shap_global_importance': shap_importance.to_dict(),
        'shap_per_class': per_class_shap,
        'hierarchical': hier_results,
        'concentration': conc_results,
        'selectivity_top_pairs': {f"{k[0]}_x_{k[1]}": v for k, v in selectivity.stack().sort_values(ascending=False).head(30).items()},
        'mechanisms': mechanisms,
    }

    path = os.path.join(RESULTS_DIR, 'step4_advanced_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(advanced_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ Advanced results: {path}")

    selectivity.to_csv(os.path.join(RESULTS_DIR, 'selectivity_matrix.csv'))
    print(f"  ✓ Selectivity matrix CSV saved")

    elapsed = time.time() - start_time

    print(f"\n" + "█"*70)
    print(f"█  ADVANCED ANALYSIS COMPLETE ({elapsed:.1f}s)                            █")
    print(f"█"*70)


if __name__ == '__main__':
    main()
