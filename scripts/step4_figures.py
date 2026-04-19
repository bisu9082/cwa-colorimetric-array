#!/usr/bin/env python3
"""
Step 4: Figure Generation v4
Changes from v3:
 Fig1 : Panel B narrower (width_ratio 0.6:1.0), B-C closer (wspace 0.04),
         Panel A x-tick fontsize 13 (matches y-tick)
 Fig3 : d2 label corrected → 'Blood / Blister / Choking'  (no TIC)
 Fig5 : Panel A colored strips removed (agent names only),
         panel gap narrowed (wspace 0.22),
         Panel C GF → lower-left; all connector lines removed; legend TIC fixed
 Fig6 : All connector lines removed throughout;
         Panel B DMMP/HD lower-left, GD slightly right;
         Panel C per-agent manual fine-tuning per user spec
 Fig8 : Panel A/B x-tick and y-tick fontsize increased
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import LeaveOneOut
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "raw")
RESULTS_DIR   = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "results")
PROCESSED_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        14,
    'axes.titlesize':   16,
    'axes.labelsize':   14,
    'xtick.labelsize':  13,
    'ytick.labelsize':  13,
    'legend.fontsize':  12,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

SERIES_COLORS = {'d1': '#2E75B6', 'd2': '#C00000', 'd3': '#548235'}
# ★ Corrected d2 label — no TIC
SERIES_LABELS = {
    'd1': 'd1: Nerve agents (G/V-series)',
    'd2': 'd2: Blood / Blister / Choking',
    'd3': 'd3: Novichok (A-series)',
}
COLORS = {'proposed': '#2E75B6', 'baseline': '#C00000', 'ablation': '#548235',
          'accent': '#ED7D31'}
COMPOUND_NAME_MAP = {'A': 'A-230', 'B': 'A-232', 'C': 'A-234', 'D': 'A-242'}

DYE_NAMES = {
    'Dye1':'Anthracene',     'Dye2':'Pyrene',          'Dye3':'Acridine Or.',
    'Dye4':'Quinine',        'Dye5':'Rhodamine B',     'Dye6':'Methyl Blue',
    'Dye7':'Fluorescein',    'Dye8':'Methyl Orange',   'Dye9':'Nile Red',
    'Dye10':'Safranin O',    'Dye11':'Toluidine BlO',  'Dye12':'Nile Blue A',
    'Dye13':'Coumarin 6',    'Dye14':'Eriochrome BT',  'Dye15':'Quinoline Y',
    'Dye16':'Bromophenol B', 'Dye17':'Brilliant Grn',  'Dye18':'Eosin Y',
    'Dye19':'Ethyl Violet',  'Dye20':'Phenol Red',     'Dye21':'Crystal Vlt',
    'Dye22':'L-Glutathione', 'Dye23':'Cysteine',       'Dye24':'Congo Red',
    'Dye25':"4,4'-DHBP",    'Dye26':'Biphenyl-diol',  'Dye27':'2,5-DAB-dithiol',
    'Dye28':'3,4-DABenzamide','Dye29':'Rhodamine 6G',
}
DYE_CLASSES = {
    'Dye1':'PAH','Dye2':'PAH',
    'Dye3':'Acridine','Dye4':'Quinoline','Dye5':'Xanthene',
    'Dye6':'Triarylmethane','Dye7':'Xanthene','Dye8':'Azo',
    'Dye9':'Oxazine','Dye10':'Phenazine','Dye11':'Thiazine',
    'Dye12':'Oxazine','Dye13':'Coumarin','Dye14':'Azo',
    'Dye15':'Quinoline','Dye16':'Sulfonephthalein','Dye17':'Triarylmethane',
    'Dye18':'Xanthene','Dye19':'Triarylmethane','Dye20':'Sulfonephthalein',
    'Dye21':'Triarylmethane','Dye22':'Thiol','Dye23':'Thiol/Amine',
    'Dye24':'Azo','Dye25':'Biphenol','Dye26':'Biphenol',
    'Dye27':'Thiol/Amine','Dye28':'Hydrazine','Dye29':'Xanthene',
}
CLASS_COLORS = {
    'PAH':'#9B59B6','Azo':'#F39C12','Triarylmethane':'#2ECC71',
    'Xanthene':'#E74C3C','Sulfonephthalein':'#3498DB','Thiazine':'#34495E',
    'Oxazine':'#16A085','Thiol':'#1ABC9C','Thiol/Amine':'#1ABC9C',
    'Quinoline':'#8E44AD','Acridine':'#D35400','Phenazine':'#95A5A6',
    'Biphenol':'#7F8C8D','Coumarin':'#27AE60','Hydrazine':'#E67E22',
}

# ─── Manual text-offset tables (no connector lines) ───────────────────────────
# Fig5 Panel C  (PCA coord units ≈ ±6)
_F5C_OVR = {
    'GF': dict(dx=-0.18, dy=-0.18, ha='right', va='top'),
    'PS': dict(dx= 0.15, dy=-0.38, ha='left',  va='top'),   # ★ PS 아래로
}
_F5C_DEF = dict(dx=0.15, dy=0.12, ha='left', va='bottom')

# Fig6 Panel B  (MW units ≈ 30–280, F1 units 0–1)
_F6B_OVR = {
    'DMMP': dict(dx=-7,  dy=-0.04,  ha='right',  va='top'),
    'HD':   dict(dx=-3,  dy=-0.028, ha='right',  va='top'),    # ★ lower-left, closer
    'GD':   dict(dx= 7,  dy= 0.00,  ha='left',   va='center'),
    'L':    dict(dx= 0,  dy=-0.032, ha='center', va='top'),    # ★ below, closer
    'GB':   dict(dx=-3,  dy=-0.028, ha='right',  va='top'),    # ★ lower-left, closer
    'GA':   dict(dx= 6,  dy= 0.00,  ha='left',   va='center'), # ★ right of marker
}
_F6B_DEF = dict(dx=5, dy=0.02, ha='left', va='bottom')

# Fig6 Panel C  (Sensor PCA units ≈ x: -5..13,  y: -5..6)
_F6C_OVR = {
    'GD':    dict(dx= 0.0,  dy=-0.45, ha='center', va='top'),
    'L':     dict(dx= 0.25, dy= 0.50, ha='left',   va='bottom'),
    'CG':    dict(dx= 0.25, dy= 0.50, ha='left',   va='bottom'),
    'GA':    dict(dx= 0.25, dy= 0.50, ha='left',   va='bottom'),
    'DMMP':  dict(dx= 0.25, dy= 0.50, ha='left',   va='bottom'),
    'HN':    dict(dx= 0.55, dy= 0.00, ha='left',   va='center'),  # ★ right of marker
    'CK':    dict(dx= 0.55, dy= 0.20, ha='left',   va='bottom'),
    'AC':    dict(dx= 0.55, dy= 0.20, ha='left',   va='bottom'),
    'A-230': dict(dx= 0.25, dy= 0.50, ha='left',   va='bottom'),
    'A-232': dict(dx= 0.25, dy= 0.50, ha='left',   va='bottom'),
    'A-234': dict(dx= 0.25, dy= 0.50, ha='left',   va='bottom'),
    'A-242': dict(dx= 0.25, dy= 0.55, ha='left',   va='bottom'),
    'PS':    dict(dx=-0.40, dy= 0.00, ha='right',  va='center'),  # ★ left of marker
    'HD':    dict(dx= 0.00, dy= 0.45, ha='center', va='bottom'),  # ★ above marker
    'GF':    dict(dx= 0.40, dy=-0.35, ha='left',   va='top'),     # ★ "gv" = GF: down + right
}
_F6C_DEF = dict(dx=0.25, dy=0.25, ha='left', va='bottom')


def _put_label(ax, x, y, text, override_table, default, fontsize=11,
               fontweight='bold', color='black'):
    """Place a text label with NO connector line, using per-agent overrides."""
    cfg = {**default, **override_table.get(text, {})}
    ax.text(x + cfg['dx'], y + cfg['dy'], text,
            fontsize=fontsize, fontweight=fontweight,
            ha=cfg['ha'], va=cfg['va'], color=color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# ─── Data I/O ─────────────────────────────────────────────────────────────────
def load_data():
    dfs = []
    for prefix in ['d1', 'd2', 'd3']:
        n = pd.read_csv(os.path.join(RAW_DIR, f"{prefix}.normal_deltaE_pivot.csv"))
        u = pd.read_csv(os.path.join(RAW_DIR, f"{prefix}.uv_deltaE_pivot.csv"))
        if prefix == 'd3':
            for col in ['Agent']:
                if col in n.columns: n[col] = n[col].map(lambda x: COMPOUND_NAME_MAP.get(x, x))
                if col in u.columns: u[col] = u[col].map(lambda x: COMPOUND_NAME_MAP.get(x, x))
        n_cols = [c for c in n.columns if c.startswith('Dye')]
        u_cols = [c for c in u.columns if c.startswith('Dye')]
        merged = pd.merge(n[['Agent','Concentration']+n_cols],
                          u[['Agent','Concentration']+u_cols],
                          on=['Agent','Concentration'], suffixes=('_N','_U'))
        merged['Series'] = prefix
        dfs.append(merged)
    df_all    = pd.concat(dfs, ignore_index=True)
    df_nonzero= df_all[df_all['Concentration'] > 0].copy()
    feature_cols = [c for c in df_all.columns if '_N' in c or '_U' in c]
    le = LabelEncoder()
    y  = le.fit_transform(df_nonzero['Agent'])
    X  = df_nonzero[feature_cols].fillna(0).values
    return df_all, df_nonzero, X, y, feature_cols, le

def load_results():
    p = os.path.join(RESULTS_DIR, 'step4_full_results.json')
    return json.load(open(p)) if os.path.exists(p) else {}

def load_advanced():
    p = os.path.join(RESULTS_DIR, 'step4_advanced_results.json')
    return json.load(open(p)) if os.path.exists(p) else {}

def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {path}")

def _legend_handles():
    return [Patch(color=SERIES_COLORS[s], label=SERIES_LABELS[s]) for s in ('d1','d2','d3')]


# ════════════════════════════════════════════════════════════════════════════
# FIG 1  — Wide landscape, B narrower, B-C closer, A x-tick font larger
# ════════════════════════════════════════════════════════════════════════════
def fig1(df_all, feature_cols):
    print("  Fig 1 …")
    normal_cols = [c for c in feature_cols if c.endswith('_N')]
    uv_cols     = [c for c in feature_cols if c.endswith('_U')]
    df500 = df_all[df_all['Concentration'] == 500]

    fig = plt.figure(figsize=(28, 14))
    # ★ 3-column: B(left) | C(center) | empty spacer matching colorbar area of A
    gs  = gridspec.GridSpec(2, 3,
                            hspace=0.38, wspace=0.04,
                            height_ratios=[1.1, 1.0],
                            width_ratios=[0.58, 0.96, 0.22])   # spacer ≈ colorbar width → C ends at x-axis right tick

    agents_ordered = df500.sort_values(['Series','Agent'])['Agent'].unique().tolist()
    series_order   = [df_all[df_all['Agent']==ag]['Series'].iloc[0]
                      for ag in agents_ordered]

    # ── (a) ΔE Fingerprint Heatmap — top, full width ──────────────────────
    ax_a = fig.add_subplot(gs[0, :])

    heat_rows = []
    for ag in agents_ordered:
        sub = df500[df500['Agent'] == ag]
        heat_rows.append(sub[feature_cols].mean().values if not sub.empty
                         else np.zeros(len(feature_cols)))
    heat_data = np.array(heat_rows)

    xtick_labels = []
    for fc in feature_cols:
        num   = fc.rsplit('_',1)[0].replace('Dye','')
        illum = 'U' if fc.endswith('_U') else 'N'
        xtick_labels.append(f"{num}{illum}")

    im_a = ax_a.imshow(heat_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax_a.set_yticks(range(len(agents_ordered)))
    ax_a.set_yticklabels(agents_ordered, fontsize=13)
    ax_a.set_xticks(range(0, len(feature_cols), 2))
    ax_a.set_xticklabels(xtick_labels[::2], fontsize=13, rotation=90)   # ★ x-tick 13
    ax_a.set_xlabel('Dye Channel  (N = Normal,  U = UV)', fontsize=14)
    ax_a.set_ylabel('Agent', fontsize=14)
    ax_a.set_title('(a)  ΔE Fingerprint Heatmap  (500 µM)',
                   fontweight='bold', fontsize=16)
    for idx in range(1, len(series_order)):
        if series_order[idx] != series_order[idx-1]:
            ax_a.axhline(idx-0.5, color='black', linewidth=2.0)
    cbar_a = plt.colorbar(im_a, ax=ax_a, shrink=0.55, pad=0.01)
    cbar_a.set_label('ΔE', fontsize=13)
    cbar_a.ax.tick_params(labelsize=12)

    # ── (b) Dose–Response Heatmap: agents × concentrations — bottom-left ──
    ax_b = fig.add_subplot(gs[1, 0])
    concs = sorted([c for c in df_all['Concentration'].unique() if c > 0])

    heat_b = []
    for ag in agents_ordered:
        row = []
        for c in concs:
            sub = df_all[(df_all['Agent']==ag) & (df_all['Concentration']==c)]
            row.append(sub[normal_cols].mean().mean() if not sub.empty else 0.0)
        heat_b.append(row)
    heat_b = np.array(heat_b)

    im_b = ax_b.imshow(heat_b, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax_b.set_yticks(range(len(agents_ordered)))
    ax_b.set_yticklabels(agents_ordered, fontsize=12)
    ax_b.set_xticks(range(len(concs)))
    ax_b.set_xticklabels([f'{int(c)}' if c == int(c) else f'{c}'
                           for c in concs], fontsize=12, rotation=45, ha='right')
    ax_b.set_xlabel('Concentration (µM)', fontsize=14)
    ax_b.set_ylabel('Agent', fontsize=14)
    ax_b.set_title('(b)  Dose–Response Heatmap  (Normal, mean ΔE)',
                   fontweight='bold', fontsize=15)
    for idx in range(1, len(series_order)):
        if series_order[idx] != series_order[idx-1]:
            ax_b.axhline(idx-0.5, color='black', linewidth=1.8)
    cbar_b = plt.colorbar(im_b, ax=ax_b, shrink=0.75, pad=0.01)
    cbar_b.set_label('Mean ΔE', fontsize=13)
    cbar_b.ax.tick_params(labelsize=12)

    # ── (c) Normal vs UV scatter — bottom-right ────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    n_mean = df500[normal_cols].mean()
    u_mean = df500[uv_cols].mean()
    ax_c.scatter(n_mean.values, u_mean.values, alpha=0.7,
                 c=COLORS['proposed'], edgecolors='black', linewidth=0.6, s=70)
    mv = max(n_mean.max(), u_mean.max()) * 1.08
    ax_c.plot([0, mv],[0, mv],'--', color='gray', alpha=0.55, linewidth=1.5)
    ax_c.set_xlabel('Normal ΔE  (mean)', fontsize=14)
    ax_c.set_ylabel('UV ΔE  (mean)', fontsize=14)
    ax_c.set_title('(c)  Normal vs UV Response', fontweight='bold', fontsize=15)
    ax_c.tick_params(labelsize=13)
    ax_c.grid(alpha=0.3)

    save_fig(fig, 'Fig1_heatmap_concentration.png')


# ════════════════════════════════════════════════════════════════════════════
# FIG 2  — unchanged
# ════════════════════════════════════════════════════════════════════════════
def fig2(results_dict, X, y, feature_cols):
    print("  Fig 2 …")
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_sc, y); importances = rf.feature_importances_

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    plt.subplots_adjust(hspace=0.60, wspace=0.38, bottom=0.06)

    ax = axes[0,0]
    if results_dict:
        pm = results_dict['proposed_method']
        bm = results_dict['baseline_models']
        ab = results_dict['ablation_studies']
        labels, accs, stds, clrs = [], [], [], []
        labels.append('OvR SVM\n(Proposed)'); accs.append(pm['overall_accuracy']); stds.append(pm['overall_accuracy_std']); clrs.append(COLORS['proposed'])
        bname_map = {'fixed_29dye':'RF (Normal only)','random_selection':'Random Features','single_best_dye':'Single Best Dye','standard_svm':'Standard SVM','svm_ovr':'Standard SVM','rf_normal':'RF (Normal only)'}
        for k,d in bm.items(): labels.append(bname_map.get(k, k.replace('_',' '))); accs.append(d['accuracy_mean']); stds.append(d['accuracy_std']); clrs.append(COLORS['baseline'])
        aname_map = {'no_ovr':'−OvR Decomp.','no_feature_selection':'−Feature Select','fixed_k10':'Fixed k=10','normal_only':'Normal Only','ensemble_instead':'Ensemble Voting'}
        for k,d in ab.items(): labels.append(aname_map.get(k, k.replace('_',' '))); accs.append(d['accuracy_mean']); stds.append(d['accuracy_std']); clrs.append(COLORS['ablation'])
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, accs, yerr=stds, capsize=5, color=clrs, alpha=0.85, edgecolor='black', linewidth=0.6, error_kw={'elinewidth':1.5})
        ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=11, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylim(0, 1.08); ax.set_ylabel('Accuracy', fontsize=13)
        ax.set_title('(a)  Classification Accuracy Comparison', fontweight='bold', fontsize=15)
        ax.grid(axis='y', alpha=0.3); ax.axhline(1/17, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
        for bar,acc,std in zip(bars,accs,stds):
            ax.text(bar.get_x()+bar.get_width()/2, max(acc+std+0.015,0.04), f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.legend([Patch(color=COLORS['proposed']),Patch(color=COLORS['baseline']),Patch(color=COLORS['ablation'])],['Proposed','Baseline','Ablation'],fontsize=10,loc='upper right')

    ax = axes[0,1]
    top_k = 15; top_idx = np.argsort(importances)[-top_k:][::-1]
    feat_labels,feat_colors = [],[]
    for i in top_idx:
        fc = feature_cols[i]; dkey = fc.rsplit('_',1)[0]; illum = 'UV' if fc.endswith('_U') else 'N'
        feat_labels.append(f"{DYE_NAMES.get(dkey,dkey)} ({illum})")
        feat_colors.append('#ED7D31' if fc.endswith('_U') else COLORS['proposed'])
    ax.barh(range(top_k), importances[top_idx], color=feat_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_k)); ax.set_yticklabels(feat_labels, fontsize=11)
    ax.set_xlabel('Importance', fontsize=13); ax.set_title('(b)  RF Feature Importance (Top 15)', fontweight='bold', fontsize=15)
    ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
    ax.legend([Patch(color=COLORS['proposed']),Patch(color='#ED7D31')],['Normal','UV'],fontsize=11,loc='lower right')

    ax = axes[1,0]
    sel = SelectKBest(f_classif,k='all'); sel.fit(X_sc,y); f_scores = sel.scores_
    n_mask = np.array([c.endswith('_N') for c in feature_cols]); u_mask = ~n_mask
    ax.bar(np.where(n_mask)[0], f_scores[n_mask], label='Normal', color=COLORS['proposed'], alpha=0.7, width=0.9)
    ax.bar(np.where(u_mask)[0], f_scores[u_mask], label='UV',     color='#ED7D31',           alpha=0.7, width=0.9)
    ax.set_xlabel('Feature Index',fontsize=13); ax.set_ylabel('ANOVA F-score',fontsize=13)
    ax.set_title('(c)  Discriminative Power (ANOVA F-score)',fontweight='bold',fontsize=15); ax.legend(fontsize=11); ax.grid(axis='y',alpha=0.3)

    ax = axes[1,1]
    if results_dict:
        pm_acc = results_dict['proposed_method']['overall_accuracy']
        abl_ab = results_dict['ablation_studies']
        abl_labels,abl_deltas = [],[]
        aname_map2 = {'no_ovr':'−OvR Decomposition','no_feature_selection':'−Feature Selection','fixed_k10':'Fixed k=10','normal_only':'Normal Illumination Only','ensemble_instead':'Ensemble Instead of SVM'}
        for k,d in abl_ab.items(): abl_labels.append(aname_map2.get(k,k)); abl_deltas.append((d['accuracy_mean']-pm_acc)*100)
        clrs_d = ['#C00000' if d<0 else '#548235' for d in abl_deltas]
        ax.barh(range(len(abl_labels)), abl_deltas, color=clrs_d, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(abl_labels))); ax.set_yticklabels(abl_labels, fontsize=11)
        ax.axvline(0, color='black', linewidth=1.2)
        ax.set_xlabel('Accuracy Change vs Proposed (%)',fontsize=13)
        ax.set_title('(d)  Ablation Study: Component Impact',fontweight='bold',fontsize=15); ax.grid(axis='x',alpha=0.3)
        x_range = max(abs(d) for d in abl_deltas)*1.5; ax.set_xlim(-x_range, x_range)
        for i,d in enumerate(abl_deltas):
            if d>=0: ax.text(-0.3, i, f'+{d:.2f}%', va='center', ha='right', fontsize=11, fontweight='bold')
            else:    ax.text(d-0.3, i, f'{d:.2f}%',  va='center', ha='right', fontsize=11, fontweight='bold')

    save_fig(fig, 'Fig2_ml_performance.png')


# ════════════════════════════════════════════════════════════════════════════
# FIG 3  — d2 label corrected (no TIC)
# ════════════════════════════════════════════════════════════════════════════
def fig3(X, y, feature_cols, le, df_nonzero):
    print("  Fig 3 …")
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    compound_names = le.classes_
    series_map = {row['Agent']: row['Series']
                  for _, row in df_nonzero.drop_duplicates('Agent').iterrows()}

    pca = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X_sc)
    perp = min(30, X_sc.shape[0]-1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sc)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.38)

    for ax_idx, (ax, X_emb, title) in enumerate(zip(
            axes[:2],
            [X_pca, X_tsne],
            ['(a)  PCA Chemical Space', '(b)  t-SNE Chemical Space'])):
        for cls_idx, name in enumerate(compound_names):
            mask = y == cls_idx
            series = series_map.get(name, 'd1')
            ax.scatter(X_emb[mask,0], X_emb[mask,1],
                       color=SERIES_COLORS[series], alpha=0.75, s=60,
                       edgecolors='black', linewidth=0.4)
        xlab = (f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
                if ax_idx == 0 else 't-SNE 1')
        ylab = (f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
                if ax_idx == 0 else 't-SNE 2')
        ax.set_xlabel(xlab, fontsize=14); ax.set_ylabel(ylab, fontsize=14)
        ax.set_title(title, fontweight='bold', fontsize=16); ax.grid(alpha=0.3)
        loc = 'best' if ax_idx == 0 else 'lower right'   # ★ (b) legend lower-right
        ax.legend(handles=_legend_handles(), fontsize=12, loc=loc,
                  framealpha=0.9, edgecolor='gray')

    # ── (c) PCA Feature Loadings ──
    ax = axes[2]
    loadings = pca.components_[:2].T   # (58, 2)
    strength  = np.abs(loadings[:,0]) + np.abs(loadings[:,1])
    top_n     = 22
    top_idx   = np.argsort(strength)[-top_n:][::-1]

    N_COLOR = '#2E75B6'; U_COLOR = '#ED7D31'
    arrow_scale = 2.8
    all_lx = [loadings[i,0]*arrow_scale for i in range(len(feature_cols))]
    all_ly = [loadings[i,1]*arrow_scale for i in range(len(feature_cols))]
    xl = max(abs(min(all_lx)), abs(max(all_lx)))*1.35
    yl = max(abs(min(all_ly)), abs(max(all_ly)))*1.35
    ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl)

    for idx in range(len(feature_cols)):
        fc  = feature_cols[idx]; col = U_COLOR if fc.endswith('_U') else N_COLOR
        lw, al = (1.8, 0.85) if idx in top_idx else (0.8, 0.25)
        ax.annotate('', xy=(loadings[idx,0]*arrow_scale, loadings[idx,1]*arrow_scale),
                    xytext=(0,0), arrowprops=dict(arrowstyle='->', color=col, lw=lw, alpha=al))

    label_scale = arrow_scale*1.18
    candidates = []
    for idx in top_idx:
        fc = feature_cols[idx]; dkey = fc.rsplit('_',1)[0]
        num = dkey.replace('Dye',''); illum = 'U' if fc.endswith('_U') else 'N'
        lx = loadings[idx,0]*label_scale; ly = loadings[idx,1]*label_scale
        candidates.append({'lx':lx,'ly':ly,'lbl':f'#{num}({illum})',
                            'col': U_COLOR if illum=='U' else N_COLOR})

    PROX = 0.28; used = [False]*len(candidates); groups = []
    for i,c in enumerate(candidates):
        if used[i]: continue
        grp = [c]; used[i] = True
        for j,c2 in enumerate(candidates):
            if j==i or used[j]: continue
            if np.hypot(c['lx']-c2['lx'], c['ly']-c2['ly']) < PROX:
                grp.append(c2); used[j] = True
        groups.append(grp)

    for grp in groups:
        cx = np.clip(np.mean([g['lx'] for g in grp]), -xl*0.92, xl*0.92)
        cy = np.clip(np.mean([g['ly'] for g in grp]), -yl*0.92, yl*0.92)
        combined = ', '.join([g['lbl'] for g in grp])
        col = U_COLOR if sum(1 for g in grp if '(U)' in g['lbl']) > len(grp)/2 else N_COLOR
        ax.text(cx, cy, combined, fontsize=9.5, ha='center', va='center',
                color=col, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    ax.axhline(0, color='gray', linewidth=0.7); ax.axvline(0, color='gray', linewidth=0.7)
    ax.set_xlabel('PC1 Loading', fontsize=14); ax.set_ylabel('PC2 Loading', fontsize=14)
    ax.set_title('(c)  PCA Feature Loadings\n(top contributors labeled)',
                 fontweight='bold', fontsize=15); ax.grid(alpha=0.25)
    ax.legend([Line2D([0],[0],color=N_COLOR,lw=2.5), Line2D([0],[0],color=U_COLOR,lw=2.5)],
              ['Normal','UV'], fontsize=13, loc='upper right', framealpha=0.9, edgecolor='gray')

    save_fig(fig, 'Fig3_chemical_space.png')


# ════════════════════════════════════════════════════════════════════════════
# FIG 4  — unchanged (good as-is)
# ════════════════════════════════════════════════════════════════════════════
def fig4(X, y, feature_cols, le, results_dict):
    print("  Fig 4 …")
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    ovr = OneVsRestClassifier(SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
    y_pred = cross_val_predict(ovr, X_sc, y, cv=LeaveOneOut())
    cm = confusion_matrix(y, y_pred); compound_names = le.classes_

    fig = plt.figure(figsize=(20, 9))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.12)

    ax_a = fig.add_subplot(gs[0])
    cm_norm = np.nan_to_num(cm.astype(float)/cm.sum(axis=1)[:,np.newaxis])
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax_a,
                xticklabels=compound_names, yticklabels=compound_names,
                cbar_kws={'label':'Normalized Count','shrink':0.8},
                linewidths=0.4, linecolor='white', vmin=0, vmax=1, annot_kws={'size':9})
    ax_a.set_xlabel('Predicted', fontweight='bold', fontsize=14)
    ax_a.set_ylabel('True',      fontweight='bold', fontsize=14)
    ax_a.set_title('(a)  LOO Confusion Matrix — 17 Compounds', fontweight='bold', fontsize=16)
    ax_a.tick_params(axis='x', rotation=45, labelsize=12)
    ax_a.tick_params(axis='y', rotation=0,  labelsize=12)
    for b in [6, 13]:
        ax_a.axhline(b, color='black', linewidth=1.5)
        ax_a.axvline(b, color='black', linewidth=1.5)

    ax_b = fig.add_subplot(gs[1])
    ax_b.axis('off')
    ax_b.set_title('(b)  Key Results Summary', fontweight='bold', fontsize=16, pad=10)
    if results_dict:
        pm = results_dict.get('proposed_method', {})
        st = results_dict.get('statistical_analysis', {})
        rows = [
            ('LOO Accuracy',     f"{pm.get('loo_accuracy',0):.1%}"),
            ('LOO F1-macro',     f"{pm.get('loo_f1',0):.4f}"),
            ('LOO Precision',    f"{pm.get('loo_precision',0):.4f}"),
            ('3-fold CV Acc.',   f"{pm.get('overall_accuracy',0):.1%} ± {pm.get('overall_accuracy_std',0):.1%}"),
            ('Optimal k',        f"{pm.get('k_optimal','?')} / 58 features"),
            ('Classifier',       'OvR SVM (RBF, C=10)'),
            ('ANOVA F',          f"{st.get('anova',{}).get('f_statistic',0):.2f}"),
            ('p-value',          f"{st.get('anova',{}).get('p_value',0):.2e}"),
            ("Cohen's f",        f"{st.get('effect_size',{}).get('cohens_f',0):.3f}"),
            ('Bootstrap 95% CI', f"[{st.get('bootstrap_ci',{}).get('ci_lower',0):.3f}, {st.get('bootstrap_ci',{}).get('ci_upper',0):.3f}]"),
        ]
        tbl = ax_b.table(cellText=rows, colLabels=['Metric','Value'],
                         cellLoc='center', loc='center', bbox=[0.02,0.05,0.96,0.88])
        tbl.auto_set_font_size(False); tbl.set_fontsize(12.5)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#AAAAAA')
            if r==0:    cell.set_facecolor('#2E75B6'); cell.set_text_props(color='white', fontweight='bold')
            elif r in (1,2): cell.set_facecolor('#BDD7EE')
            elif r%2==0: cell.set_facecolor('#F5F5F5')
            else:        cell.set_facecolor('white')

    save_fig(fig, 'Fig4_confusion_stats.png')


# ════════════════════════════════════════════════════════════════════════════
# FIG 5  — Panel D removed, 3-panel horizontal;
#           Panel A: NO color strips, agent names only;
#           Panel C: GF lower-left, NO connector lines, legend no TIC
# ════════════════════════════════════════════════════════════════════════════
def fig5(df_nonzero, feature_cols):
    print("  Fig 5 …")
    p_adesc = os.path.join(RESULTS_DIR, 'agent_descriptors.csv')
    p_corr  = os.path.join(RESULTS_DIR, 'descriptor_dye_correlation.csv')
    if not os.path.exists(p_adesc):
        print("    ⚠ Cheminformatics CSVs not found; skipping Fig 5"); return

    agent_desc = pd.read_csv(p_adesc, index_col=0)
    corr_df    = pd.read_csv(p_corr,  index_col=0)
    df500 = df_nonzero[df_nonzero['Concentration'] == 500]
    resp_profile = df500.groupby('Agent')[feature_cols].mean()

    fig = plt.figure(figsize=(28, 9))
    gs  = gridspec.GridSpec(1, 3, wspace=0.22)   # ★ tighter gap

    # ── (a) Agent Descriptor Heatmap ─────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    desc_data = agent_desc.select_dtypes(include=[np.number]).fillna(0)
    # ★ Fill NaN after Z-score (zero-variance cols → NaN → show as 0 = neutral white)
    _scaled_arr = StandardScaler().fit_transform(desc_data)
    _scaled_arr = np.nan_to_num(_scaled_arr, nan=0.0)
    desc_scaled = pd.DataFrame(_scaled_arr, index=desc_data.index, columns=desc_data.columns)
    agents_ordered_a = sorted(
        desc_scaled.index,
        key=lambda a: {'d1':0,'d2':1,'d3':2}.get(
            df_nonzero[df_nonzero['Agent']==a]['Series'].iloc[0]
            if (df_nonzero['Agent']==a).any() else 'd1', 0))
    desc_ordered = desc_scaled.loc[agents_ordered_a] if all(
        a in desc_scaled.index for a in agents_ordered_a) else desc_scaled

    im_a = ax_a.imshow(desc_ordered.values, aspect='auto', cmap='RdBu_r',
                        vmin=-2.5, vmax=2.5, interpolation='nearest')
    ax_a.set_yticks(range(len(desc_ordered.index)))
    ax_a.set_yticklabels(desc_ordered.index, fontsize=13)   # ★ agent names only, no strips
    ax_a.set_xticks(range(len(desc_ordered.columns)))
    ax_a.set_xticklabels(desc_ordered.columns, fontsize=11, rotation=45, ha='right')
    ax_a.set_xlabel('Molecular Descriptor', fontsize=14)
    ax_a.set_ylabel('Agent', fontsize=14)
    ax_a.set_title('(a)  Agent Molecular Descriptor Profiles (Z-score)',
                   fontweight='bold', fontsize=15)
    # ★ Color strips REMOVED — only agent name text remains (via ytick labels above)
    plt.colorbar(im_a, ax=ax_a, shrink=0.65, pad=0.01).set_label('Z-score', fontsize=13)

    # ── (b) Descriptor-Dye Correlation Heatmap ───────────────────────────
    ax_b = fig.add_subplot(gs[1])
    if corr_df is not None and not corr_df.empty:
        dye_short = [c[:10] for c in corr_df.columns]
        im_b = ax_b.imshow(corr_df.values, aspect='auto', cmap='RdBu_r',
                            vmin=-1, vmax=1, interpolation='nearest')
        ax_b.set_xticks(range(len(dye_short)))
        ax_b.set_xticklabels(dye_short, fontsize=10, rotation=90)
        ax_b.set_yticks(range(len(corr_df.index)))
        ax_b.set_yticklabels(corr_df.index, fontsize=12)
        ax_b.set_title('(b)  Descriptor–Dye Response Correlation\n(Pearson r, p<0.05)',
                        fontweight='bold', fontsize=15)
        ax_b.set_xlabel('Dye', fontsize=14); ax_b.set_ylabel('Descriptor', fontsize=14)
        plt.colorbar(im_b, ax=ax_b, shrink=0.65, pad=0.01).set_label('Pearson r', fontsize=13)

    # ── (c) Molecular Descriptor PCA Space — GF lower-left, no lines ─────
    ax_c = fig.add_subplot(gs[2])
    common = sorted(set(desc_ordered.index) & set(resp_profile.index))
    if len(common) >= 3:
        X_desc = desc_ordered.loc[common].fillna(0).values
        pca_d  = PCA(n_components=2)
        X_d    = pca_d.fit_transform(StandardScaler().fit_transform(X_desc))
        series_list = [df_nonzero[df_nonzero['Agent']==ag]['Series'].iloc[0]
                       if (df_nonzero['Agent']==ag).any() else 'd1'
                       for ag in common]

        for i, ag in enumerate(common):
            ax_c.scatter(X_d[i,0], X_d[i,1],
                         c=SERIES_COLORS.get(series_list[i],'gray'), s=110,
                         edgecolors='black', linewidth=0.6, zorder=5)

        # ★ Manual labels, NO connector lines
        for i, ag in enumerate(common):
            col = SERIES_COLORS.get(series_list[i], 'gray')
            _put_label(ax_c, X_d[i,0], X_d[i,1], ag,
                       _F5C_OVR, _F5C_DEF, fontsize=12, color=col)

        ax_c.set_xlabel(f'PC1 ({pca_d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
        ax_c.set_ylabel(f'PC2 ({pca_d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
        ax_c.set_title('(c)  Molecular Descriptor PCA Space', fontweight='bold', fontsize=15)
        ax_c.grid(alpha=0.3); ax_c.tick_params(labelsize=13)
        # ★ Legend: use SERIES_LABELS (no TIC)
        for s in ('d1','d2','d3'):
            ax_c.plot([],[], 'o', color=SERIES_COLORS[s],
                      label=SERIES_LABELS[s], markersize=10)
        ax_c.legend(fontsize=11, loc='best')

    save_fig(fig, 'Fig5_cheminformatics.png')


# ════════════════════════════════════════════════════════════════════════════
# FIG 6  — All connector lines removed; Panel B/C manual positioning
# ════════════════════════════════════════════════════════════════════════════
def fig6(df_nonzero, feature_cols, results_dict):
    print("  Fig 6 …")
    p_adesc = os.path.join(RESULTS_DIR, 'agent_descriptors.csv')
    if not os.path.exists(p_adesc):
        print("    ⚠ Agent descriptor CSV not found; skipping Fig 6"); return

    agent_desc = pd.read_csv(p_adesc, index_col=0)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df_nonzero[feature_cols].fillna(0).values)
    le     = LabelEncoder(); y = le.fit_transform(df_nonzero['Agent'])
    rf     = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_sc, y)
    importances = rf.feature_importances_
    feat_imp = dict(zip(feature_cols, importances))
    class_report = results_dict.get('detailed_metrics',{}).get('classification_report',{})

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(2, 2, hspace=0.44, wspace=0.42)

    # ── (a) Feature importance by dye class ──────────────────────────────
    ax_a = fig.add_subplot(gs[0,0])
    dye_class_data = {}
    for feat, imp in feat_imp.items():
        dkey = feat.rsplit('_',1)[0]; illum = feat.rsplit('_',1)[1]
        dc   = DYE_CLASSES.get(dkey,'Unknown')
        if dc not in dye_class_data: dye_class_data[dc] = {'N':0,'U':0}
        dye_class_data[dc]['N' if illum=='N' else 'U'] += imp
    classes = sorted(dye_class_data, key=lambda x: -(dye_class_data[x]['N']+dye_class_data[x]['U']))
    y_pos  = np.arange(len(classes))
    ax_a.barh(y_pos+0.2, [dye_class_data[c]['N'] for c in classes], 0.38, label='Normal', color='#2E75B6', alpha=0.85)
    ax_a.barh(y_pos-0.2, [dye_class_data[c]['U'] for c in classes], 0.38, label='UV',     color='#ED7D31', alpha=0.85)
    ax_a.set_yticks(y_pos); ax_a.set_yticklabels(classes, fontsize=13)
    ax_a.set_xlabel('Cumulative Feature Importance', fontsize=14)
    ax_a.set_title('(a)  Feature Importance by Dye Class', fontweight='bold', fontsize=16)
    ax_a.legend(fontsize=13); ax_a.invert_yaxis(); ax_a.grid(axis='x', alpha=0.3)
    ax_a.tick_params(labelsize=13)

    # ── (b) MW / LogP vs LOO F1 — NO connector lines, manual positioning ──
    ax_b = fig.add_subplot(gs[0,1])
    if class_report:
        common = sorted(set(agent_desc.index) & set(class_report.keys()))
        if len(common) > 2:
            mws   = [agent_desc.loc[a,'MW']   for a in common if 'MW'   in agent_desc.columns]
            logps = [agent_desc.loc[a,'LogP']  for a in common if 'LogP' in agent_desc.columns]
            tpsas = [agent_desc.loc[a,'TPSA']  for a in common if 'TPSA' in agent_desc.columns]
            f1s   = [class_report[a].get('f1-score',0) for a in common]
            n     = min(len(mws),len(logps),len(tpsas),len(f1s))
            mws,logps,tpsas,f1s = mws[:n],logps[:n],tpsas[:n],f1s[:n]
            sc = ax_b.scatter(mws, f1s, c=logps,
                              s=[max(t*3+30,20) for t in tpsas],
                              cmap='coolwarm', edgecolors='black', linewidth=0.6,
                              alpha=0.85, zorder=5)
            # ★ NO connector lines — manual positioning only
            for i, ag in enumerate(common[:n]):
                _put_label(ax_b, mws[i], f1s[i], ag,
                           _F6B_OVR, _F6B_DEF, fontsize=11)
            cb = plt.colorbar(sc, ax=ax_b, shrink=0.7)
            cb.set_label('LogP', fontsize=13)
    ax_b.set_xlabel('Molecular Weight (Da)', fontsize=14)
    ax_b.set_ylabel('LOO F1-score', fontsize=14)
    ax_b.set_title('(b)  Mol. Properties vs Classification F1\n(size=TPSA, color=LogP)',
                   fontweight='bold', fontsize=15)
    ax_b.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax_b.grid(alpha=0.2); ax_b.tick_params(labelsize=13)

    # ── (c) Sensor PCA — NO connector lines, manual fine-tuned positions ──
    ax_c = fig.add_subplot(gs[1,0])
    df500   = df_nonzero[df_nonzero['Concentration']==500]
    resp_pr = df500.groupby('Agent')[feature_cols].mean()
    common2 = sorted(set(agent_desc.index) & set(resp_pr.index))
    if len(common2) >= 3:
        X_r  = StandardScaler().fit_transform(resp_pr.loc[common2].fillna(0).values)
        pca2 = PCA(n_components=2); Xr2 = pca2.fit_transform(X_r)
        f1_vals = [class_report.get(a,{}).get('f1-score',0) for a in common2]
        sc2 = ax_c.scatter(Xr2[:,0], Xr2[:,1],
                            c=f1_vals, cmap='RdYlGn', vmin=0, vmax=1,
                            s=130, edgecolors='black', linewidth=0.8, zorder=5)
        # ★ NO connector lines — manual fine-tuning per user spec
        for i, ag in enumerate(common2):
            col = 'red' if f1_vals[i] < 0.4 else 'black'
            _put_label(ax_c, Xr2[i,0], Xr2[i,1], ag,
                       _F6C_OVR, _F6C_DEF, fontsize=11, color=col)
        cb2 = plt.colorbar(sc2, ax=ax_c, shrink=0.7)
        cb2.set_label('LOO F1-score', fontsize=13)
    ax_c.set_xlabel('Sensor PC1', fontsize=14); ax_c.set_ylabel('Sensor PC2', fontsize=14)
    ax_c.set_title('(c)  Sensor Response Space (color=F1)', fontweight='bold', fontsize=16)
    ax_c.grid(alpha=0.3); ax_c.tick_params(labelsize=13)

    # ── (d) Dye-class importance heatmap ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1,1])
    dc_n = {c: dye_class_data[c]['N'] for c in classes}
    dc_u = {c: dye_class_data[c]['U'] for c in classes}
    mat  = np.array([[dc_n[c], dc_u[c]] for c in classes])
    im_d = ax_d.imshow(mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax_d.set_xticks([0,1]); ax_d.set_xticklabels(['Normal','UV'], fontsize=14)
    ax_d.set_yticks(range(len(classes))); ax_d.set_yticklabels(classes, fontsize=13)
    ax_d.set_title('(d)  Dye-Class Importance\nNormal vs UV Illumination',
                   fontweight='bold', fontsize=15)
    ax_d.set_xlabel('Illumination', fontsize=14)
    for i, c in enumerate(classes):
        for j, val in enumerate([dc_n[c], dc_u[c]]):
            ax_d.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=10.5,
                      fontweight='bold', color='white' if val>mat.max()*0.6 else 'black')
    plt.colorbar(im_d, ax=ax_d, shrink=0.7).set_label('Importance', fontsize=13)

    save_fig(fig, 'Fig6_ml_chemistry_integration.png')


# ════════════════════════════════════════════════════════════════════════════
# FIG 7  — d2 label corrected, rest unchanged
# ════════════════════════════════════════════════════════════════════════════
def fig7(df_all, feature_cols):
    print("  Fig 7 …")
    adv = load_advanced()
    if not adv: print("    ⚠ Advanced results not found; skipping Fig 7"); return

    hier      = adv.get('hierarchical',{})
    shap_glob = adv.get('shap_global_importance',{})

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.38)

    ax_a = fig.add_subplot(gs[0,0])
    shap_sorted = sorted(shap_glob.items(), key=lambda x: -x[1])[:20]
    feat_labels, feat_vals, bar_colors_a = [], [], []
    for fc,val in shap_sorted:
        dkey = fc.rsplit('_',1)[0]; illum = 'UV' if fc.endswith('_U') else 'N'
        feat_labels.append(f"{DYE_NAMES.get(dkey,dkey)} ({illum})")
        feat_vals.append(val)
        bar_colors_a.append(CLASS_COLORS.get(DYE_CLASSES.get(dkey,'Unknown'),'#BDC3C7'))
    ax_a.barh(range(len(feat_vals)), feat_vals, color=bar_colors_a, edgecolor='black', linewidth=0.4)
    ax_a.set_yticks(range(len(feat_labels))); ax_a.set_yticklabels(feat_labels, fontsize=12)
    ax_a.set_xlabel('Mean |SHAP value|', fontsize=14)
    ax_a.set_title('(a)  SHAP Feature Importance (Top 20)\ncolored by dye class',
                   fontweight='bold', fontsize=15)
    ax_a.invert_yaxis(); ax_a.grid(axis='x', alpha=0.3)
    seen = {DYE_CLASSES.get(fc.rsplit('_',1)[0],'Unknown'): CLASS_COLORS.get(DYE_CLASSES.get(fc.rsplit('_',1)[0],'Unknown'),'#BDC3C7') for fc,_ in shap_sorted}
    # ★ Legend: x축 바로 위에 위치 (lower right)
    ax_a.legend(handles=[Patch(color=c, label=k) for k, c in seen.items()],
                fontsize=12, loc='lower right', ncol=2, framealpha=0.9)

    ax_b = fig.add_subplot(gs[0,1])
    t2  = hier.get('tier2',{})
    cats = ['Tier 1\n(Series,\n3 cls)','Tier 2\nd1: OP\n(6 cls)','Tier 2\nd2: TIC\n(7 cls)',
            'Tier 2\nd3: Nov\n(4 cls)','Combined\nHier.','Flat\nOvR SVM']
    accs = [hier.get('tier1',{}).get('accuracy',0),
            t2.get('d1',{}).get('accuracy',0), t2.get('d2',{}).get('accuracy',0),
            t2.get('d3',{}).get('accuracy',0), hier.get('hierarchical_accuracy',0), 0.6324]
    bars_b = ax_b.bar(range(len(cats)), accs,
                       color=['#2E75B6','#4BACC6','#4BACC6','#4BACC6','#8E44AD','#95A5A6'],
                       edgecolor='black', linewidth=0.6)
    ax_b.set_xticks(range(len(cats))); ax_b.set_xticklabels(cats, fontsize=12)
    ax_b.set_ylim(0, 1.18); ax_b.set_ylabel('LOO Accuracy', fontsize=14)
    ax_b.set_title('(b)  Hierarchical vs Flat Classification', fontweight='bold', fontsize=15)
    ax_b.axhline(1/17, color='gray', linestyle=':', alpha=0.55, linewidth=1.2, label='Random (17-class)')
    ax_b.legend(fontsize=12); ax_b.grid(axis='y', alpha=0.3)
    for bar,acc in zip(bars_b,accs):
        ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.035, f'{acc:.1%}',
                  ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax_c = fig.add_subplot(gs[1,0])
    df_full  = df_all.copy()
    feat_sub = [c for c in feature_cols if c in df_full.columns]
    concs    = sorted([c for c in df_full['Concentration'].unique() if c > 0])
    series_sty = {
        'd1': {'color':'#2E75B6','marker':'o','label': SERIES_LABELS['d1']},
        'd2': {'color':'#C00000','marker':'s','label': SERIES_LABELS['d2']},   # ★ corrected
        'd3': {'color':'#548235','marker':'D','label': SERIES_LABELS['d3']},
    }
    for s, sty in series_sty.items():
        means, stds = [], []
        for conc in concs:
            mask = (df_full['Concentration']==conc) & (df_full['Series']==s)
            vals = df_full.loc[mask, feat_sub].values
            per_samp = np.mean(vals, axis=1)
            means.append(np.mean(per_samp)); stds.append(np.std(per_samp))
        means, stds = np.array(means), np.array(stds)
        ax_c.errorbar(concs, means, yerr=stds, fmt=f"{sty['marker']}-",
                      color=sty['color'], linewidth=2.2, markersize=9,
                      capsize=5, capthick=1.8, label=sty['label'])
        ax_c.fill_between(concs, means-stds, means+stds, alpha=0.08, color=sty['color'])
    ax_c.set_xlabel('Concentration (µM)', fontsize=14); ax_c.set_ylabel('Mean ΔE  (±1 SD)', fontsize=14)
    ax_c.set_title('(c)  Signal Intensity vs Concentration', fontweight='bold', fontsize=15)
    ax_c.set_xscale('log'); ax_c.legend(fontsize=12, loc='upper left'); ax_c.grid(alpha=0.3)
    ax_c.tick_params(labelsize=13)
    ax_c.text(0.97,0.04,'Error bars: ±1 SD across agents',
              transform=ax_c.transAxes, fontsize=11, ha='right', style='italic', color='gray')

    ax_d = fig.add_subplot(gs[1,1])
    df500 = df_full[df_full['Concentration']==500]
    agent_means = df500.groupby('Agent')[feat_sub].mean().mean(axis=1).sort_values(ascending=True)
    colors_d = [SERIES_COLORS.get(df_full[df_full['Agent']==ag]['Series'].iloc[0],'gray')
                for ag in agent_means.index]
    ax_d.barh(range(len(agent_means)), agent_means.values,
              color=colors_d, edgecolor='black', linewidth=0.4)
    ax_d.set_yticks(range(len(agent_means))); ax_d.set_yticklabels(agent_means.index, fontsize=13)
    ax_d.set_xlabel('Mean ΔE across all channels (500 µM)', fontsize=14)
    ax_d.set_title('(d)  Signal Strength by Agent\n(500 µM, all 58 channels)',
                   fontweight='bold', fontsize=15)
    ax_d.grid(axis='x', alpha=0.3); ax_d.tick_params(labelsize=13)
    for i,v in enumerate(agent_means.values):
        ax_d.text(v+0.02, i, f'{v:.2f}', va='center', fontsize=11, fontweight='bold')

    save_fig(fig, 'Fig7_advanced_ml.png')


# ════════════════════════════════════════════════════════════════════════════
# FIG 8  — Panel A/B x-tick and y-tick fonts larger
# ════════════════════════════════════════════════════════════════════════════
def fig8():
    print("  Fig 8 …")
    adv   = load_advanced()
    p_sel = os.path.join(RESULTS_DIR, 'selectivity_matrix.csv')
    if not adv or not os.path.exists(p_sel):
        print("    ⚠ Selectivity matrix not found; skipping Fig 8"); return

    sel_df    = pd.read_csv(p_sel, index_col=0)
    per_class = adv.get('shap_per_class',{})
    mechanisms= adv.get('mechanisms',{})

    fig = plt.figure(figsize=(28, 20))
    gs  = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.42,
                            width_ratios=[2, 1.2, 1.0])

    # ── (a) Selectivity heatmap ───────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0,:])
    sel_disp = sel_df.copy()
    sel_disp.columns = [DYE_NAMES.get(c,c)[:11] for c in sel_disp.columns]
    sns.heatmap(sel_disp.clip(upper=5), cmap='YlOrRd', ax=ax_a,
                linewidths=0.25, linecolor='white',
                cbar_kws={'label':'Selectivity Index (SI, capped at 5)','shrink':0.45},
                xticklabels=True, yticklabels=True, annot=False)
    ax_a.set_title('(a)  Selectivity Index Heatmap  (SI = ΔE_agent / mean ΔE_others, Normal, 500 µM)',
                   fontweight='bold', fontsize=17)
    ax_a.set_xlabel('Dye', fontsize=15); ax_a.set_ylabel('Agent', fontsize=15)
    ax_a.tick_params(axis='x', rotation=70, labelsize=15)   # ★ x-tick 15
    ax_a.tick_params(axis='y', labelsize=14, rotation=0)    # ★ y-tick 14

    # ── (b) Per-agent SHAP fingerprint ───────────────────────────────────
    ax_b = fig.add_subplot(gs[1,0])
    all_agents = sorted(per_class.keys())
    all_feats  = sorted(set(f for fd in per_class.values() for f in fd))
    shap_mat   = np.array([[per_class.get(ag,{}).get(f,0) for f in all_feats]
                            for ag in all_agents])
    top_f_idx  = np.argsort(shap_mat.sum(axis=0))[-20:][::-1]
    feat_lbs   = [f"{DYE_NAMES.get(all_feats[i].rsplit('_',1)[0], all_feats[i].rsplit('_',1)[0])[:9]} "
                  f"({'(UV)' if all_feats[i].endswith('_U') else '(N)'})"
                  for i in top_f_idx]
    sns.heatmap(shap_mat[:,top_f_idx], cmap='Reds', ax=ax_b,
                xticklabels=feat_lbs, yticklabels=all_agents,
                linewidths=0.2, linecolor='white',
                cbar_kws={'label':'|SHAP|','shrink':0.7})
    ax_b.set_title('(b)  Per-Agent SHAP Fingerprint\n(Top 20 Features)',
                   fontweight='bold', fontsize=16)
    ax_b.set_xlabel('Feature  (Dye × Illumination)', fontsize=14)
    ax_b.set_ylabel('Agent', fontsize=14)
    ax_b.tick_params(axis='x', rotation=75, labelsize=14)   # ★ x-tick 14
    ax_b.tick_params(axis='y', labelsize=13, rotation=0)

    # ── (c) Mechanism category bar chart ─────────────────────────────────
    ax_c = fig.add_subplot(gs[1,1])
    mech_groups = {}
    for agent, info in mechanisms.items():
        m = info.get('proposed_mechanism','Unknown')
        if   'Lewis acid'      in m: key = 'Lewis Acid–Base'
        elif 'Nucleophilic'    in m: key = 'Nucleophilic Sub.'
        elif 'Electron transfer' in m: key = 'Electron Transfer'
        elif 'Anion exchange'  in m or 'halide' in m.lower(): key = 'Anion Exchange'
        elif 'coordination'    in m.lower(): key = 'Coordination'
        elif 'Fluorescence'    in m: key = 'Fluorescence Mod.'
        else:                        key = 'Other'
        mech_groups.setdefault(key,[]).append(agent)

    mech_sorted  = sorted(mech_groups.items(), key=lambda x: -len(x[1]))
    mech_names   = [m for m,_ in mech_sorted]
    mech_counts  = [len(ag) for _,ag in mech_sorted]
    mech_agents  = [', '.join(sorted(ag)) for _,ag in mech_sorted]
    mech_palette = ['#2E75B6','#C00000','#548235','#ED7D31','#8E44AD','#1ABC9C','#95A5A6']
    ax_c.barh(range(len(mech_names)), mech_counts,
              color=mech_palette[:len(mech_names)], edgecolor='black', linewidth=0.6)
    ax_c.set_yticks(range(len(mech_names)))
    ax_c.set_yticklabels(mech_names, fontsize=15)          # ★ y-tick 15
    ax_c.set_xlabel('Number of Agents', fontsize=14)
    ax_c.set_title('(c)  Detection Mechanism Categories', fontweight='bold', fontsize=16)
    ax_c.set_xlim(0, max(mech_counts)*1.55); ax_c.grid(axis='x', alpha=0.3)
    ax_c.tick_params(axis='x', labelsize=14)               # ★ x-tick 14
    ax_c.tick_params(axis='y', labelsize=15)               # ★ y-tick 15
    for i,(cnt,ag_str) in enumerate(zip(mech_counts, mech_agents)):
        ax_c.text(cnt+0.08, i, f'{cnt}  ({ag_str})', va='center', fontsize=11, fontweight='bold')

    save_fig(fig, 'Fig8_selectivity_mechanism.png')


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("\n"+"═"*70)
    print("  FIGURE GENERATION v4  —  All 8 Figures")
    print("═"*70)
    df_all, df_nonzero, X, y, feature_cols, le = load_data()
    results_dict = load_results()
    fig1(df_all, feature_cols)
    fig2(results_dict, X, y, feature_cols)
    fig3(X, y, feature_cols, le, df_nonzero)
    fig4(X, y, feature_cols, le, results_dict)
    fig5(df_nonzero, feature_cols)
    fig6(df_nonzero, feature_cols, results_dict)
    fig7(df_all, feature_cols)
    fig8()
    print("\n"+"═"*70)
    print("  ALL FIGURES COMPLETE")
    print("═"*70)

if __name__ == '__main__':
    main()
