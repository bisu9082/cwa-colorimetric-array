#!/usr/bin/env python3
"""
Generate FigS1: 17-agent LOO confusion matrix for SI
Saves as confusion_matrix_17agents.png (+ .pdf) in manuscript folder
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "processed")
RESULTS_DIR   = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "data", "results")
OUT_DIR       = os.path.join(BASE_DIR, "mnt", "## research", "deltaE", "manuscript")

# ── Load data ──────────────────────────────────────────────────────────────
import pandas as pd
p_proc = os.path.join(PROCESSED_DIR, 'integrated_17compounds.csv')
df = pd.read_csv(p_proc)
df = df.rename(columns={'Agent': 'agent', 'Concentration': 'concentration', 'Series': 'series'})

with open(os.path.join(RESULTS_DIR, 'step4_full_results.json')) as f:
    full_res = json.load(f)

k_opt = full_res.get('proposed_method', {}).get('k_optimal', 58)

# Feature columns
feature_cols = [c for c in df.columns if c not in ['agent', 'concentration', 'series', 'replicate', 'Series', 'Agent', 'Concentration']]
# Apply feature selection (top-k)
from sklearn.feature_selection import SelectKBest, f_classif
le = LabelEncoder()
y  = le.fit_transform(df['agent'])
X  = df[feature_cols].values

sel = SelectKBest(f_classif, k=min(k_opt, len(feature_cols)))
X_sel = sel.fit_transform(X, y)
selected_cols = [feature_cols[i] for i in sel.get_support(indices=True)]

# Series color map for annotation lines
d1_agents = ['DMMP','GA','GB','GD','GF','VX']
d2_agents = ['AC','CG','CK','HD','HN','L','PS']
d3_agents = ['A-230','A-232','A-234','A-242']

# Sort by series for cleaner visualization
all_classes = list(le.classes_)
ordered_names = [a for a in d1_agents if a in all_classes] + \
                [a for a in d2_agents if a in all_classes] + \
                [a for a in d3_agents if a in all_classes]
# build reorder index
order_idx = [all_classes.index(n) for n in ordered_names]
compound_names = ordered_names

# ── LOO-CV prediction ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_sel)
ovr    = OneVsRestClassifier(SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
print("Running LOO-CV (this takes a moment) …")
y_pred = cross_val_predict(ovr, X_sc, y, cv=LeaveOneOut())

cm      = confusion_matrix(y, y_pred)
cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1)[:, np.newaxis])
# Reorder rows and columns by series grouping (d1 → d2 → d3)
cm_norm = cm_norm[np.ix_(order_idx, order_idx)]

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))

# Series group colors for tick labels
series_color = {}
for name in compound_names:
    if name in d1_agents:  series_color[name] = '#1f77b4'   # blue
    elif name in d2_agents: series_color[name] = '#d62728'  # red
    else:                   series_color[name] = '#2ca02c'  # green

sns.heatmap(
    cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
    xticklabels=compound_names, yticklabels=compound_names,
    cbar_kws={'label': 'Normalized Count', 'shrink': 0.75},
    linewidths=0.4, linecolor='white',
    vmin=0, vmax=1, annot_kws={'size': 10}
)

# Series separator lines
n_d1 = len([a for a in d1_agents if a in all_classes])
n_d2 = len([a for a in d2_agents if a in all_classes])
b1, b2 = n_d1, n_d1 + n_d2
for b in [b1, b2]:
    ax.axhline(b, color='black', linewidth=2.0)
    ax.axvline(b, color='black', linewidth=2.0)

ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
ax.set_ylabel('True Label',      fontweight='bold', fontsize=14)
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.tick_params(axis='y', rotation=0,  labelsize=12)

# Color tick labels by series
for tick_label in ax.get_xticklabels():
    txt = tick_label.get_text()
    tick_label.set_color(series_color.get(txt, 'black'))
for tick_label in ax.get_yticklabels():
    txt = tick_label.get_text()
    tick_label.set_color(series_color.get(txt, 'black'))

# Legend for series
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', label='Nerve agents (d1)'),
    Patch(facecolor='#d62728', label='Blood/blister/choking agents (d2)'),
    Patch(facecolor='#2ca02c', label='Novichok A-series (d3)'),
]
ax.legend(handles=legend_elements, loc='upper right',
          fontsize=11, framealpha=0.9,
          bbox_to_anchor=(1.22, 1.14))

plt.tight_layout()

# Save
out_png = os.path.join(OUT_DIR, 'confusion_matrix_17agents.png')
out_pdf = os.path.join(OUT_DIR, 'confusion_matrix_17agents.pdf')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close(fig)
