from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RT_FILE = Path("results/residual_analysis/mixedlm_results.csv")
QWEN_FILE = Path("results/qwen3/qwen_position_regression_results.csv")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

POSITION_ORDER = ["K-3", "K-2", "K-1", "K", "K+1", "K+2", "K+3", "K_END"]
POSITION_X = {p: i for i, p in enumerate(POSITION_ORDER)}
COLORS = {"CD": "#69B3A2", "RD": "#B05A6E"}
DODGE = {"CD": -0.07, "RD": 0.07}

rt = pd.read_csv(RT_FILE)
qwen = pd.read_csv(QWEN_FILE)

# A = Human RT; B = Qwen surprisal; C = Qwen representation distance
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# A. Human reading time
ax = axes[0]
for dist in ["CD", "RD"]:
    dat = rt[rt["DistType"] == dist].copy()
    dat["X"] = dat["Position"].map(POSITION_X)
    dat = dat.dropna(subset=["X", "Beta", "CILow", "CIHigh"]).sort_values("X")
    x = dat["X"].to_numpy(float) + DODGE[dist]
    y = dat["Beta"].to_numpy(float)
    yerr = np.vstack([y - dat["CILow"].to_numpy(float), dat["CIHigh"].to_numpy(float) - y])
    ax.errorbar(x, y, yerr=yerr, color=COLORS[dist], marker="o", markersize=6, linewidth=2, capsize=3.5, label=dist)

    yrange = max(dat["CIHigh"].max() - dat["CILow"].min(), 0.01)
    for xv, hi, sig in zip(x, dat["CIHigh"], dat["Significant_FDR_Within"].fillna(False)):
        if sig: ax.text(xv, hi + yrange * 0.025, "*", ha="center", va="bottom", fontsize=14, color=COLORS[dist])

ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
ax.set_title("Human reading time\n")
ax.set_ylabel("Estimated difference in residualized log RT")
ax.text(-0.12, 1.06, "A", transform=ax.transAxes, fontsize=15, fontweight="bold")

# B/C. Qwen
panels = [
    ("DeltaSurprisal", "Contextual surprisal", "Estimated difference in contextual surprisal", "B"),
    ("RepresentationDistance", "Output-layer representation distance", "Cosine distance", "C")
]

for ax, (outcome, title, ylabel, panel) in zip(axes[1:], panels):
    plot_data = qwen[qwen["Outcome"] == outcome].copy()

    for dist in ["CD", "RD"]:
        dat = plot_data[plot_data["DistType"] == dist].copy()
        dat["X"] = dat["Position"].map(POSITION_X)
        dat = dat.dropna(subset=["X", "Estimate", "CILow", "CIHigh"]).sort_values("X")
        x = dat["X"].to_numpy(float) + DODGE[dist]
        y = dat["Estimate"].to_numpy(float)
        yerr = np.vstack([y - dat["CILow"].to_numpy(float), dat["CIHigh"].to_numpy(float) - y])
        ax.errorbar(x, y, yerr=yerr, color=COLORS[dist], marker="o", markersize=6, linewidth=2, capsize=3.5)

        yrange = max(plot_data["CIHigh"].max() - plot_data["CILow"].min(), 0.001)
        for xv, hi, sig in zip(x, dat["CIHigh"], dat["Significant_FDR_Within"].fillna(False)):
            if sig: ax.text(xv, hi + yrange * 0.025, "*", ha="center", va="bottom", fontsize=14, color=COLORS[dist])

    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_title(title+"\n")
    ax.set_ylabel(ylabel)
    ax.text(-0.12, 1.06, panel, transform=ax.transAxes, fontsize=15, fontweight="bold")

# Common formatting
for ax in axes:
    ax.set_xticks(range(len(POSITION_ORDER)), POSITION_ORDER)
    ax.set_xlabel("Analysis position")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

# One centered figure-level legend
handles = [
    plt.Line2D([0], [0], color=COLORS["CD"], marker="o", linewidth=2, markersize=6, label="CD - Original"),
    plt.Line2D([0], [0], color=COLORS["RD"], marker="o", linewidth=2, markersize=6, label="RD - Original")
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.2)
fig.savefig(OUT / "Fig_2.svg", format="svg", bbox_inches="tight")
plt.show()
plt.close(fig)

# Copy other figures
import shutil

files = {
    "results/qwen3/cosine_decay/cosine_distance_results.svg": OUT / "Fig_5.svg",
    "results/distortion_surface_controls/distortion_surface_controls.svg": OUT / "Fig_1.svg",
    "results/residual_analysis/decay/main_distance_results.svg": OUT / "Fig_3.svg",
    "results/qwen3/surprisal_decay/surprisal_distance_results.svg": OUT / "Fig_4.svg",
    
}

for src, dst in files.items():
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")



