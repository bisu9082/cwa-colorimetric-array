# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":8.2,
    "axes.linewidth":0.9,"axes.labelsize":8.6,
    "xtick.labelsize":7.8,"ytick.labelsize":7.8,
    "xtick.major.width":0.9,"ytick.major.width":0.9,
    "legend.frameon":False,"legend.fontsize":7.4,
})
DASH = "–"                      # en dash for P–F, LC–HRMS
C_NEG, C_POS = "#C0392B", "#2E7D32"
C_BLUE, C_GREY, C_ORNG = "#1F4E79", "#8C8C8C", "#E07B39"

def panel_title(ax, lab, title, dy=1.05, fs=9.2):
    t = ax.text(0.0, dy, f"({lab})", transform=ax.transAxes,
                fontsize=fs, fontweight="bold", va="bottom", ha="left")
    fig = ax.figure; fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.text(bb.x1 + 0.025, dy, title, transform=ax.transAxes,
            fontsize=fs-0.6, va="bottom", ha="left")

fig = plt.figure(figsize=(7.28, 6.15))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.40,
                      left=0.148, right=0.972, top=0.928, bottom=0.092)

# ─────────────── (a) targeted search outcome ───────────────
ax = fig.add_subplot(gs[0, 0])
labels = ["Covalent\nadducts", "MS$^2$ adduct\nfragments", "Non-covalent\nintact clusters",
          "Agent hydrolysis\nproducts", "Intact\nagents", "Intact\ndyes"]
searched = np.array([216, 103, 72, 33, 30, 32])
detected = np.array([0, 0, 0, 13, 9, 8])
y = np.arange(len(labels))[::-1]
ax.barh(y, searched, height=0.60, color="#DCDCDC", edgecolor="#8C8C8C", lw=0.7, zorder=2)
for yy, s, d in zip(y, searched, detected):
    if d:
        ax.barh(yy, d, height=0.60, color=C_POS, edgecolor="none", zorder=3)
        ax.text(s + 6, yy, f"{d} / {s}", va="center", ha="left", fontsize=7.1, color="#333333")
    else:
        ax.plot(2.0, yy, marker="X", ms=6.4, mew=0, color=C_NEG, zorder=4)
        ax.text(s + 6, yy, f"0 / {s}", va="center", ha="left", fontsize=7.1,
                color=C_NEG, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.3)
ax.set_xlim(0, 292); ax.set_xlabel("Number of species")
ax.spines[["top", "right"]].set_visible(False)
hh = [Rectangle((0,0),1,1,fc="#DCDCDC",ec="#8C8C8C",lw=0.7), Rectangle((0,0),1,1,fc=C_POS),
      plt.Line2D([],[],ls="none",marker="X",ms=6.0,color=C_NEG)]
ax.legend(hh, ["searched", "detected", "zero detected"], loc="lower right",
          bbox_to_anchor=(1.03, -0.035), handlelength=1.1, handleheight=0.85,
          borderpad=0.2, labelspacing=0.30)
panel_title(ax, "a", "Targeted search outcome")

# ─────────────── (b) detection power ───────────────
ax = fig.add_subplot(gs[0, 1])
dyes = ["Rhodamine B / 6G", "Quinine", "2-hydrazino-\nbenzothiazole", "2,4-DNPH", "Anthracene"]
lo = np.array([0.0016, 0.0062, 0.0078, 11.65, 36.09])
hi = np.array([0.0019, 0.1195, 0.0106, 20.22, 65.08])
ok = [True, True, True, False, False]
y = np.arange(len(dyes))[::-1]
ax.axvspan(1.0, 300, color=C_NEG, alpha=0.07, zorder=1)
ax.axvline(1.0, color=C_NEG, ls="--", lw=1.0, zorder=2)
for yy, l, h_, k in zip(y, lo, hi, ok):
    c = C_BLUE if k else C_GREY
    ax.plot([l, h_], [yy, yy], color=c, lw=3.6, solid_capstyle="round", zorder=3,
            alpha=1.0 if k else 0.6)
    ax.plot([l, h_], [yy, yy], "o", ms=3.6, color=c, zorder=4)
ax.text(1.55, 3.55, "1% conversion", fontsize=6.9, color=C_NEG, va="center", rotation=90)
ax.text(26, 0.60, "inconclusive:\npoor ESI response", fontsize=6.8, color="#666666",
        ha="center", va="center", linespacing=1.25)
ax.set_xscale("log"); ax.set_xlim(6e-4, 300)
ax.set_yticks(y); ax.set_yticklabels(dyes, fontsize=7.3)
ax.set_xlabel("Minimum detectable conversion (%)")
ax.spines[["top", "right"]].set_visible(False)
panel_title(ax, "b", "Detection power of the negative")

# ─────────────── (c) covalent vs non-covalent ───────────────
ax = fig.add_subplot(gs[1, 0])
mz_obs, mz_cov = 574.3517, 556.3411
ax.vlines(mz_obs, 0, 148, color=C_POS, lw=3.4, zorder=3)
ax.plot(mz_obs, 148, "o", ms=5.0, color=C_POS, zorder=4)
ax.plot(mz_cov, 0, marker="X", ms=7.0, mew=0, color=C_NEG, zorder=4, clip_on=False)
ax.annotate("", xy=(mz_obs, 96), xytext=(mz_cov, 96),
            arrowprops=dict(arrowstyle="<->", lw=0.95, color="#404040"))
ax.text((mz_obs + mz_cov) / 2, 101, f"{DASH}H$_2$O   18.011 Da", ha="center", va="bottom",
        fontsize=7.1, color="#404040")
ax.text(mz_obs, 156, "[Quinine + A-242\nhydrolysate + H]$^+$", ha="center", va="bottom",
        fontsize=7.2, color=C_POS, fontweight="bold", linespacing=1.25)
ax.text(mz_obs + 1.4, 140, "574.3517\n$\\Delta$ +0.27 ppm\n148 scans", ha="left", va="top",
        fontsize=7.0, color=C_POS, linespacing=1.3)
ax.text(mz_cov, 26, "covalent ester\n556.3411\n0 scans", ha="center", va="bottom",
        fontsize=7.0, color=C_NEG, linespacing=1.3)
ax.set_xlim(548, 586); ax.set_ylim(0, 200)
ax.set_xlabel("$m/z$"); ax.set_ylabel("Scans detected")
ax.set_yticks([0, 50, 100, 150, 200])
ax.spines[["top", "right"]].set_visible(False)
panel_title(ax, "c", "Observed ion is an exact mass sum")

# ─────────────── (d) descriptor record: no separation ───────────────
ax = fig.add_subplot(gs[1, 1])
# converged-geometry Mayer P-F bond orders (Table S13); VX excluded (P-S, different bond type)
A = [("A-242",1.0428),("A-232",1.0525),("A-230",1.0609),("A-234",1.0862)]
G = [("GD",1.0509),("GB",1.0544),("GF",1.0554)]
yA, yG = 0.66, 0.26
ax.axvspan(min(v for _,v in G), max(v for _,v in G), color=C_BLUE, alpha=.10, zorder=1)
ax.hlines(yA, min(v for _,v in A), max(v for _,v in A), color=C_ORNG, lw=2.2, alpha=.40, zorder=2)
ax.hlines(yG, min(v for _,v in G), max(v for _,v in G), color=C_BLUE, lw=2.2, alpha=.40, zorder=2)
for k,v in A:
    ax.plot(v, yA, "o", ms=9, color=C_ORNG, mec="#7A3E12", mew=.9, zorder=4)
    ax.annotate(k,(v,yA),textcoords="offset points",xytext=(0,14),ha="center",
                fontsize=7.8,fontweight="bold",color="#5A2D0C")
# G-agents are within 0.005 of each other: stagger the labels with leader lines
for i,(k,v) in enumerate(G):
    ax.plot(v, yG, "s", ms=8, color=C_BLUE, mec="#12314F", mew=.9, zorder=4)
    ty = yG-0.115 if i%2==0 else yG-0.205
    ax.plot([v,v],[yG-0.035,ty+0.028],lw=.8,color="#8AA8C0",zorder=3)
    ax.annotate(k,(v,ty),ha="center",va="center",fontsize=7.8,fontweight="bold",color="#12314F")
ax.set_yticks([yA,yG]); ax.set_yticklabels(["A-series","G-agents"],fontsize=8.6,fontweight="bold")
for t,c in zip(ax.get_yticklabels(),[C_ORNG,C_BLUE]): t.set_color(c)
ax.tick_params(axis='y',length=0)
ax.text(0.5,1.00,"the G-agent range lies inside the A-series range",transform=ax.transAxes,
        ha="center",va="top",fontsize=7.5,color="#555")
ax.set_xlim(1.036,1.092); ax.set_ylim(-0.02,0.92)
ax.set_xlabel("Mayer P\u2013F bond order (converged geometry)")
ax.spines[["top","right","left"]].set_visible(False)
panel_title(ax, "d", "Descriptor separates neither family")

fig.savefig("figures/newfig5.png", dpi=600, facecolor="white")
print("saved")
