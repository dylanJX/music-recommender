"""Generate the final AUC progression chart with all 13 submissions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

charts_dir = Path(__file__).parent / "charts"
charts_dir.mkdir(exist_ok=True)

submissions = [
    "hw4.py\nbaseline",
    "Rule 1\n(hw4)",
    "Rule 2\n(Hybrid CF)",
    "Rule 3\n(Pop Boost)",
    "LGBM v1\n(naive)",
    "LGBM v2\n(hard neg)",
    "LGBM v3\n(binary)",
    "LGBM v4\n(40k rank)",
    "LGBM v5\n(40k cls)",
    "Ensemble\n(R2+LGBM)",
    "Ensemble v2\n(opt wts)",
    "LGBM v6\n(interact)",
    "Optimized\nRule 2",
]
aucs_all = [0.851, 0.857, 0.863, 0.859, 0.675, 0.762, 0.768, 0.769, 0.801,
            0.875, 0.828, 0.804, 0.903]

# Color code: blue=heuristic, orange=LGBM, green=ensemble, gold=final
marker_colors = [
    "#3498db",  # hw4 baseline (heuristic)
    "#3498db",  # rule1 (heuristic)
    "#3498db",  # rule2 (heuristic)
    "#3498db",  # rule3 (heuristic)
    "#e67e22",  # lgbm v1
    "#e67e22",  # lgbm v2
    "#e67e22",  # lgbm v3
    "#e67e22",  # lgbm v4
    "#e67e22",  # lgbm v5
    "#2ecc71",  # ensemble
    "#2ecc71",  # ensemble v2
    "#e67e22",  # lgbm v6
    "#FFD700",  # optimized rule 2 (STAR)
]

fig, ax = plt.subplots(figsize=(15, 7))
x = np.arange(len(submissions))

# Line connecting all points
ax.plot(x, aucs_all, color="#555555", linewidth=1.4, zorder=1, linestyle="-")

# Individual markers (all except the last)
for xi, (auc, col) in enumerate(zip(aucs_all[:-1], marker_colors[:-1])):
    ax.scatter(xi, auc, color=col, s=90, zorder=3, edgecolors="white", linewidths=1.2)
    offset = 0.006 if auc > 0.76 else -0.012
    va = "bottom" if auc > 0.76 else "top"
    ax.text(xi, auc + offset, f"{auc:.3f}", ha="center", va=va, fontsize=8, fontweight="bold")

# Star marker for the final result
ax.scatter(x[-1], aucs_all[-1], color="#FFD700", s=300, zorder=5,
           marker="*", edgecolors="#B8860B", linewidths=1.0)
ax.text(x[-1], aucs_all[-1] + 0.008, f"{aucs_all[-1]:.3f}", ha="center", va="bottom",
        fontsize=10, fontweight="bold", color="#B8860B")

# Rank 1 threshold line
ax.axhline(y=0.890, color="#e74c3c", linestyle="--", linewidth=2.0,
           label="Rank 1 threshold (0.890)", zorder=2)

# Previous best line
ax.axhline(y=0.875, color="#8e44ad", linestyle="--", linewidth=1.2,
           label="Previous best: Ensemble (0.875)", zorder=2, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(submissions, fontsize=8)
ax.set_ylabel("Kaggle ROC AUC", fontsize=12)
ax.set_title("AUC Progression Across All 13 Submissions", fontsize=14, fontweight="bold")
ax.set_ylim(0.60, 0.92)
ax.grid(axis="y", alpha=0.3)

# Legend with color-coded categories
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Heuristic rules'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', markersize=10, label='LightGBM variants'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Ensemble'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700', markersize=14, label='Final: Optimized Rule 2 (0.903)'),
    Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2.0, label='Rank 1 threshold (0.890)'),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

# Shade LGBM region
ax.axvspan(3.5, 8.5, alpha=0.05, color="#e67e22")
ax.text(6, 0.615, "Standalone LightGBM variants", ha="center", fontsize=8, color="#c0392b", style="italic")

plt.tight_layout()
plt.savefig(charts_dir / "auc_progression_final.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved auc_progression_final.png")
