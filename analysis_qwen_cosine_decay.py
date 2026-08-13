# 1. packages and constants
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from patsy import build_design_matrices
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm import tqdm

# 2. constances
FILE = "results/qwen3/qwen_word_effects.csv"
OUT = "results/qwen3/cosine_decay"
os.makedirs(OUT, exist_ok=True)

STORY = "Story"
OUTCOME = "RepresentationDistance"
DIST_TYPES = ["CD", "RD"]

MIN_DISTANCE = 0
MAX_DISTANCE = 10
MIN_TRAJECTORIES_PER_DISTANCE = 10

EARLY_DISTANCES = [1, 2, 3]
LATE_DISTANCES = [8, 9, 10]

N_BOOT = 1000
RANDOM_SEED = 42

EFFECT_COLORS = {"CD": "#69B3A2", "RD": "#B05A6E"}

COLORS = {"CD": "#69B3A2", "RD": "#B05A6E"}
BAR_COLORS = {"CD": "#B8DCD4", "RD": "#DDB7C1"}

MODEL_LABELS = {
    "constant": "Constant", "linear": "Linear", "quadratic": "Quadratic",
    "exponential": "Exponential", "power_law": "Power law",
    "log_normal": "Log-normal", "log_cauchy": "Log-Cauchy",
    "zipf_alekseev": "Zipf-Alekseev", "broken_stick": "Broken stick",
    "exp_plus_exp": "Exp. + exp.", "exp_plus_power": "Exp. + power",
}

MODEL_ORDER = [
    "log_normal", "log_cauchy", "zipf_alekseev", "broken_stick",
    "exponential", "power_law", "exp_plus_exp", "exp_plus_power",
    "linear", "quadratic", "constant",
]

# 3. functions
def fixed_design_row(fit, row):
    design_info = fit.model.data.design_info
    matrix = build_design_matrices([design_info], pd.DataFrame([row]), return_type="dataframe")[0]
    return matrix.reindex(columns=fit.params.index, fill_value=0.0).to_numpy(float).ravel()

def contrast_stats(fit, contrast):
    result = fit.t_test(np.asarray(contrast, float)[np.newaxis, :])
    estimate = float(np.asarray(result.effect).squeeze())
    standard_error = float(np.asarray(result.sd).squeeze())
    t_value = float(np.asarray(result.tvalue).squeeze())
    p_value = float(np.asarray(result.pvalue).squeeze())
    ci_low, ci_high = np.asarray(result.conf_int(alpha=0.05), float).reshape(-1, 2)[0]
    return estimate, standard_error, t_value, p_value, float(ci_low), float(ci_high)

def distance_label(distance):
    return "K" if int(distance) == 0 else f"K+{int(distance)}"

def curve_constant(x, baseline):
    return np.full_like(np.asarray(x, float), baseline)

def curve_linear(x, baseline, slope):
    x = np.asarray(x, float)
    return baseline + slope * x

def curve_quadratic(x, baseline, slope, quadratic):
    x = np.asarray(x, float)
    return baseline + slope * x + quadratic * x**2

def curve_exponential(x, baseline, amplitude, rate):
    x = np.asarray(x, float)
    return baseline + amplitude * np.exp(-rate * x)

def curve_power_law(x, baseline, amplitude, alpha):
    x = np.asarray(x, float)
    return baseline + amplitude * (x + 1.0)**(-alpha)

def curve_log_normal(x, baseline, amplitude, mu, sigma):
    log_x = np.log(np.asarray(x, float) + 1.0)
    return baseline + amplitude * np.exp(-0.5 * ((log_x - mu) / sigma)**2)

def curve_log_cauchy(x, baseline, amplitude, mu, sigma):
    log_x = np.log(np.asarray(x, float) + 1.0)
    z = log_x - mu
    return baseline + amplitude * sigma**2 / (sigma**2 + z**2)

def curve_zipf_alekseev(x, baseline, amplitude, alpha, beta):
    x = np.asarray(x, float)
    log_x = np.log(x + 1.0)
    return baseline + amplitude * np.exp(-alpha * log_x - beta * log_x**2)

def curve_broken_stick(x, baseline, slope_1, slope_2, knot):
    x = np.asarray(x, float)
    return baseline + slope_1 * x + (slope_2 - slope_1) * np.maximum(x - knot, 0.0)

def curve_exp_plus_exp(x, baseline, amplitude, rate_1, rate_2, exp_weight):
    x = np.asarray(x, float)
    shape = exp_weight * np.exp(-rate_1 * x) + (1.0 - exp_weight) * np.exp(-rate_2 * x)
    return baseline + amplitude * shape

def curve_exp_plus_power(x, baseline, amplitude, rate, alpha, power_weight):
    x = np.asarray(x, float)
    shape = power_weight * (x + 1.0)**(-alpha) + (1.0 - power_weight) * np.exp(-rate * x)
    return baseline + amplitude * shape

def curve_bounds(name, x):
    
    if name == "constant": 
        return [-np.inf], [np.inf]
    
    if name == "linear": 
        return [-np.inf] * 2, [np.inf] * 2
    
    if name == "quadratic": 
        return [-np.inf] * 3, [np.inf] * 3
    
    if name in {"exponential", "power_law"}: 
        return [-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 10.0]
    
    if name in {"log_normal", "log_cauchy"}:
        log_x = np.log(np.asarray(x, float) + 1.0)
        return [-np.inf, -np.inf, log_x.min() - 3.0, 0.05], [np.inf, np.inf, log_x.max() + 3.0, 5.0]
    
    if name == "zipf_alekseev": 
        return [-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, 10.0, 10.0]
    
    if name == "broken_stick": 
        return [-np.inf, -np.inf, -np.inf, float(x[1])], [np.inf, np.inf, np.inf, float(x[-2])]
    
    return [-np.inf, -np.inf, 1e-6, 1e-6, 0.0], [np.inf, np.inf, 10.0, 10.0, 1.0]

def curve_starts(name, x, y):
    baseline = float(y[-1])
    amplitude = float(y[0] - baseline)
    scale = max(float(np.ptp(y)), 1e-3)
    if abs(amplitude) < scale * 0.05: amplitude = scale
    
    slope, intercept = np.polyfit(x, y, 1)
    
    if name == "constant": 
        return [[float(np.mean(y))]]
    
    if name == "linear": 
        return [[float(intercept), float(slope)]]
    
    if name == "quadratic":
        quadratic, slope, intercept = np.polyfit(x, y, 2)
        return [[float(intercept), float(slope), float(quadratic)]]
    
    if name in {"exponential", "power_law"}: 
        return [[baseline, amplitude, rate] for rate in (0.1, 0.5, 1.0, 2.0)]
    
    log_x = np.log(np.asarray(x, float) + 1.0)
    peak_mu = float(log_x[np.argmax(np.abs(y - baseline))])
    mus = {peak_mu, float(log_x.min()), float(np.median(log_x))}

    if name in {"log_normal", "log_cauchy"}:
        return [[baseline, float(y[np.argmin(np.abs(log_x - mu))] - baseline), mu, sigma] for mu in mus for sigma in (0.3, 0.8, 1.5)]

    if name == "zipf_alekseev": 
        return [[baseline, amplitude, alpha, beta] for alpha in (0.1, 0.5, 1.0) for beta in (0.0, 0.1, 0.5)]
    
    if name == "broken_stick": 
        return [[float(intercept), float(slope), float(slope), float(knot)] for knot in np.quantile(x, [0.3, 0.5, 0.7])]
    
    if name == "exp_plus_exp": 
        return [[baseline, amplitude, r1, r2, weight] for r1, r2 in ((0.1, 1.0), (0.2, 2.0), (0.5, 5.0)) for weight in (0.25, 0.5, 0.75)]
    
    return [[baseline, amplitude, rate, alpha, weight] for rate, alpha in ((0.1, 0.5), (0.5, 1.0), (1.0, 2.0)) for weight in (0.25, 0.5, 0.75)]

def nearest_psd(matrix):
    matrix = (np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    floor = max(float(np.max(np.abs(values))) * 1e-10, 1e-12)
    values = np.clip(values, floor, None)
    return (vectors * values) @ vectors.T

def fit_curve_model(name, x, y, effect_cov):
    function = CURVE_SPECS[name]["function"]
    parameter_names = CURVE_SPECS[name]["parameters"]
    n, k = len(x), len(parameter_names)
    if n <= k + 1: raise ValueError(f"{name} needs more than {k + 1} distance points for AICc")
    safe_cov = nearest_psd(effect_cov)
    lower, upper = curve_bounds(name, x)
    best, errors = None, []
    for p0 in curve_starts(name, x, y):
        params, param_cov = curve_fit(function, x, y, p0=p0, bounds=(lower, upper), sigma=safe_cov, absolute_sigma=True, method="trf", max_nfev=100000, x_scale="jac")
        predicted = function(x, *params)
        residual = y - predicted
        gls_chisq = float(residual @ np.linalg.solve(safe_cov, residual))
        if best is None or gls_chisq < best["GLSChiSquare"]:
            best = {"params": params, "param_cov": param_cov, "predicted": predicted, "GLSChiSquare": gls_chisq}
            
    if best is None: raise RuntimeError("; ".join(errors[:3]))
    logdet = float(np.linalg.slogdet(safe_cov)[1])
    minus_2_loglik = float(n * np.log(2 * np.pi) + logdet + best["GLSChiSquare"])
    aic = minus_2_loglik + 2 * k
    best["Minus2LogLik"] = minus_2_loglik
    best["AIC"] = aic
    best["AICc"] = aic + (2 * k * (k + 1)) / (n - k - 1)
    best["BIC"] = minus_2_loglik + k * np.log(n)
    best["RawRMSE"] = float(np.sqrt(np.mean((y - best["predicted"])**2)))
    best["ParamCovCondition"] = float(np.linalg.cond(best["param_cov"]))
    return best

def calculate_curve_metrics(model_name, params, x_min, x_max):
    function = CURVE_SPECS[model_name]["function"]
    predict = lambda value: float(function(np.array([value], dtype=float), *params)[0])
    if model_name == "log_normal":
        baseline, amplitude, mu, sigma = params
        half_width = sigma * np.sqrt(2 * np.log(2))
        return {
            "Baseline": baseline,
            "Amplitude": amplitude,
            "PeakDistance": np.exp(mu) - 1,
            "PeakEffect": baseline + amplitude,
            "HalfMaximumLow": np.exp(mu - half_width) - 1,
            "HalfMaximumHigh": np.exp(mu + half_width) - 1,
            f"EffectAtKPlus{int(x_min)}": predict(x_min),
            f"EffectAtKPlus{int(x_max)}": predict(x_max),
            f"LocalComponentAtKPlus{int(x_max)}": predict(x_max) - baseline,
        }
    if model_name == "linear":
        intercept, slope = params
        zero_crossing = -intercept / slope if abs(slope) > 1e-10 else np.nan
        return {
            "InterceptAtK": intercept,
            "SlopePerWord": slope,
            f"EffectAtKPlus{int(x_min)}": predict(x_min),
            f"EffectAtKPlus{int(x_max)}": predict(x_max),
            f"TotalChangeKPlus{int(x_min)}ToKPlus{int(x_max)}": predict(x_max) - predict(x_min),
            "ZeroCrossingDistance": zero_crossing,
        }
    raise ValueError(f"No derived metrics defined for {model_name}")

def p_text(p):
    if p < 0.001: return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")

def panel_label(ax, label, x=-0.05, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold")

def make_qwen_main_figure(trajectory_label, top_ylabel, boundary_ylabel, persistence_ylabel, output_stem):
    distance_interaction = direct_wald[
        direct_wald["Term"].astype(str).str.contains("C(Distance)", regex=False) &
        direct_wald["Term"].astype(str).str.contains("DistType") &
        direct_wald["Term"].astype(str).str.contains(":")
    ].iloc[0]
    boundary_interaction = direct_wald[
        direct_wald["Term"].astype(str).str.contains("AtBoundary", regex=False) &
        direct_wald["Term"].astype(str).str.contains("DistType") &
        direct_wald["Term"].astype(str).str.contains(":")
    ].iloc[0]

    style = {
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
        "axes.labelsize": 11, "legend.fontsize": 9.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none",
    }

    with plt.rc_context(style):
        fig = plt.figure(figsize=(13.2, 12.8), facecolor="white", constrained_layout=True)
        outer = fig.add_gridspec(3, 2, height_ratios=[1.02, 1.10, 1.02], width_ratios=[1, 1], hspace=0.12, wspace=0.10)
        ax_cd = fig.add_subplot(outer[0, 0])
        ax_rd = fig.add_subplot(outer[0, 1], sharey=ax_cd)
        ax_aicc_cd = fig.add_subplot(outer[1, 0])
        ax_aicc_rd = fig.add_subplot(outer[1, 1])
        ax_boundary = fig.add_subplot(outer[2, 0])
        ax_persistence = fig.add_subplot(outer[2, 1])

        top_axes = {"CD": ax_cd, "RD": ax_rd}
        top_bounds = []
        for dist in DIST_TYPES:
            axis = top_axes[dist]
            observed = distance_effects[distance_effects["DistType"] == dist].sort_values("Distance").copy()
            best_row = best_models.loc[best_models["DistType"] == dist].iloc[0]
            best_name = best_row["Model"]
            best_params = curve_fits[dist][best_name]["params"]
            x_grid = np.linspace(observed["Distance"].min(), observed["Distance"].max(), 500)
            y_grid = CURVE_SPECS[best_name]["function"](x_grid, *best_params)

            axis.axhline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=0)
            axis.errorbar(
                observed["Distance"], observed["Estimate"],
                yerr=[observed["Estimate"] - observed["CILow"], observed["CIHigh"] - observed["Estimate"]],
                fmt="o", color=COLORS[dist], ecolor=COLORS[dist], markersize=6, capsize=3,
                elinewidth=1.4, label="Clustered-OLS estimate", zorder=3
            )
            axis.plot(x_grid, y_grid, color=COLORS[dist], linewidth=2.8, label=f"Best fit: {MODEL_LABELS.get(best_name, best_name)}", zorder=2)
            axis.set_title(f"{dist} {trajectory_label} trajectory")
            axis.set_xlabel("Word distance from final K token")
            ticks = list(range(MIN_DISTANCE, MAX_DISTANCE + 1))
            axis.set_xticks(ticks)
            axis.set_xticklabels([distance_label(d) for d in ticks])
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
            axis.legend(frameon=False, loc="upper right")
            axis.text(0.98, 0.03, f"Best-model AICc = {best_row['AICc']:.2f}", transform=axis.transAxes, ha="right", va="bottom", color="#555555")
            top_bounds.extend(observed["CILow"].tolist() + observed["CIHigh"].tolist() + y_grid.tolist())

        ax_cd.set_ylabel(top_ylabel)
        plt.setp(ax_rd.get_yticklabels(), visible=False)
        finite_bounds = np.asarray(top_bounds, dtype=float)
        finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
        y_low = min(0.0, finite_bounds.min())
        y_high = max(0.0, finite_bounds.max())
        y_pad = max((y_high - y_low) * 0.12, 0.005)
        ax_cd.set_ylim(y_low - y_pad, y_high + y_pad)
        panel_label(ax_cd, "A")
        panel_label(ax_rd, "B")

        fig.text(
            0.5, 1.005,
            f"Direct distance × distortion-type interaction: F({int(distance_interaction['df_constraint'])}, {int(distance_interaction['df_denom'])}) = {float(distance_interaction['statistic']):.2f}, {p_text(float(distance_interaction['pvalue']))}",
            ha="center", va="bottom", fontsize=11.5
        )

        comparison_plot = curve_comparison[np.isfinite(curve_comparison["AICc"])].copy()
        available_models = [model for model in MODEL_ORDER if model in comparison_plot["Model"].unique()]
        model_positions = np.arange(len(available_models))
        model_tick_labels = [MODEL_LABELS.get(model, model) for model in available_models]

        for dist, axis, panel in [("CD", ax_aicc_cd, "C"), ("RD", ax_aicc_rd, "D")]:
            subset = comparison_plot[comparison_plot["DistType"] == dist].set_index("Model")
            aicc_values = np.array([subset.loc[model, "AICc"] if model in subset.index else np.nan for model in available_models], dtype=float)
            finite_aicc = aicc_values[np.isfinite(aicc_values)]
            if finite_aicc.size == 0:
                axis.text(0.5, 0.5, "No valid AICc values", transform=axis.transAxes, ha="center", va="center")
                continue
            selected_model = best_models.loc[best_models["DistType"] == dist, "Model"].iloc[0]
            axis.scatter(model_positions, aicc_values, s=72, color=BAR_COLORS[dist], edgecolor="#222222", linewidth=0.8, zorder=3)
            if selected_model in available_models:
                selected_index = available_models.index(selected_model)
                selected_aicc = aicc_values[selected_index]
                axis.scatter(selected_index, selected_aicc, s=190, marker="*", color=COLORS[dist], edgecolor="#222222", linewidth=0.8, zorder=4)
                axis.text(0.98, 0.96, f"★ Best AICc = {selected_aicc:.2f}", transform=axis.transAxes, ha="right", va="top", fontsize=9)
            aicc_pad = max(float(np.ptp(finite_aicc)) * 0.12, 1.0)
            axis.set_ylim(float(finite_aicc.min()) - aicc_pad, float(finite_aicc.max()) + aicc_pad)
            axis.set_xticks(model_positions, model_tick_labels, rotation=32, ha="right")
            axis.set_ylabel("AICc (lower is better)")
            axis.set_title(f"{dist} model comparison")
            axis.set_xlabel("Candidate functional form")
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
            panel_label(axis, panel, x=-0.05, y=1.08)

        boundary_plot = direct_boundary.set_index("DistType").loc[DIST_TYPES].reset_index()
        x_boundary = np.arange(len(boundary_plot))
        for idx, row in boundary_plot.iterrows():
            color = COLORS[row["DistType"]]
            ax_boundary.errorbar(
                idx, row["Estimate"],
                yerr=[[row["Estimate"] - row["CILow"]], [row["CIHigh"] - row["Estimate"]]],
                fmt="o", color=color, ecolor=color, markersize=8, capsize=4, elinewidth=1.7
            )
        ax_boundary.axhline(0, color="#777777", linestyle="--", linewidth=1.2)
        ax_boundary.set_xticks(x_boundary, boundary_plot["DistType"])
        ax_boundary.set_ylabel(boundary_ylabel)
        ax_boundary.set_title(f"Boundary modulation: interaction {p_text(float(boundary_interaction['pvalue']))}")
        ax_boundary.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        boundary_range = max(boundary_plot["CIHigh"].max() - boundary_plot["CILow"].min(), 0.01)
        for idx, row in boundary_plot.iterrows():
            ax_boundary.text(idx, row["CIHigh"] + boundary_range * 0.06, p_text(float(row["P"])), ha="center", va="bottom", fontsize=9)
        ax_boundary.set_ylim(min(0, boundary_plot["CILow"].min()) - boundary_range * 0.12, boundary_plot["CIHigh"].max() + boundary_range * 0.22)
        panel_label(ax_boundary, "E", x=-0.05, y=1.08)

        persistence_p = float(direct_persistence["P"].iloc[0])
        window_designs = {"CD": [cd_early, cd_late], "RD": [rd_early, rd_late]}
        x_windows = np.array([0, 1], dtype=float)
        for dist, marker in [("CD", "o"), ("RD", "s")]:
            values = [contrast_stats(direct_fit, design) for design in window_designs[dist]]
            estimates = np.array([value[0] for value in values], dtype=float)
            lows = np.array([value[4] for value in values], dtype=float)
            highs = np.array([value[5] for value in values], dtype=float)
            ax_persistence.errorbar(
                x_windows, estimates, yerr=[estimates - lows, highs - estimates],
                fmt=f"{marker}-", color=COLORS[dist], markersize=7, linewidth=2.2,
                elinewidth=1.5, capsize=3, label=dist, zorder=3
            )
        ax_persistence.set_xticks(x_windows, [f"Early\n{distance_label(min(EARLY_DISTANCES))}–{distance_label(max(EARLY_DISTANCES))}", f"Late\n{distance_label(min(LATE_DISTANCES))}–{distance_label(max(LATE_DISTANCES))}"])
        ax_persistence.set_xlim(-0.28, 1.28)
        ax_persistence.set_ylabel(persistence_ylabel)
        ax_persistence.set_title(f"Relative persistence: distance-window interaction {p_text(persistence_p)}")
        ax_persistence.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        ax_persistence.legend(frameon=False, loc="best")
        panel_label(ax_persistence, "F", x=-0.05, y=1.08)

        fig.savefig(os.path.join(OUT, f"{output_stem}.svg"), format="svg", bbox_inches="tight", facecolor="white")
        plt.show()

# 4. commands
# read saved Qwen outcomes
distance_data = pd.read_csv(FILE)

required = [
    "DistType", STORY, "PerturbationID", "Distance", "AtBoundary",
    "DistWordIndex", "OriginalWordIndex", OUTCOME,
]

missing = [column for column in required if column not in distance_data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

distance_data = distance_data.dropna(subset=required).copy()
distance_data["Distance"] = pd.to_numeric(distance_data["Distance"], errors="raise").astype(int)
distance_data["AtBoundary"] = pd.to_numeric(distance_data["AtBoundary"], errors="raise").astype(int)
distance_data = distance_data[
    distance_data["DistType"].isin(DIST_TYPES) &
    distance_data["Distance"].between(MIN_DISTANCE, MAX_DISTANCE)
].copy()

distance_data["TrajectoryID"] = (
    distance_data["DistType"].astype(str) + "|" +
    distance_data[STORY].astype(str) + "|" +
    distance_data["PerturbationID"].astype(str)
)

distance_data = distance_data.drop_duplicates(["DistType", "TrajectoryID", "Distance", "DistWordIndex", "OriginalWordIndex"])

distance_counts = distance_data.groupby(["DistType", "Distance"], as_index=False).agg(
    N=(OUTCOME, "size"),
    NTrajectories=("TrajectoryID", "nunique"),
    NStories=(STORY, "nunique"),
    NBoundary=("AtBoundary", "sum"),
)

distance_counts.to_csv(os.path.join(OUT, "cosine_distance_counts.csv"), index=False,)
print(distance_counts)

# 3. functions and commands for clustered OLS
term_test_tables, model_fits, model_data = [], {}, {}

for dist in DIST_TYPES:
    dat = distance_data[distance_data["DistType"] == dist].dropna(subset=[OUTCOME, "Distance", "AtBoundary", "TrajectoryID"]).copy()
    support = dat.groupby("Distance").agg(NObservations=(OUTCOME, "size"), NTrajectories=("TrajectoryID", "nunique")).reset_index()
    valid_distances = support.loc[support["NTrajectories"] >= MIN_TRAJECTORIES_PER_DISTANCE, "Distance"].astype(int).sort_values().tolist()
    dat = dat[dat["Distance"].isin(valid_distances)].copy()
    
    if MIN_DISTANCE not in valid_distances or len(valid_distances) < 4:
        print(f"{dist}: insufficient supported distances")
        continue
    
    dat["Distance"] = pd.Categorical(dat["Distance"], categories=valid_distances, ordered=True)
    full_formula = f"{OUTCOME} ~ C(Distance) + AtBoundary"
    full_fit = smf.ols(full_formula, data=dat).fit(cov_type="cluster", cov_kwds={"groups": dat["TrajectoryID"], "use_correction": True}, use_t=True)
    model_fits[dist], model_data[dist] = full_fit, dat
    with open(os.path.join(OUT, f"{dist}_cosine_distance_ols.txt"), "w", encoding="utf-8") as file:
        file.write(full_formula + "\n\n" + full_fit.summary().as_text())
    
    wald_table = full_fit.wald_test_terms(skip_single=False, scalar=True).table.copy()
    wald_table.index.name = "Term"
    wald_table = wald_table.reset_index()
    wald_table.insert(0, "DistType", dist)
    wald_table["N"] = len(dat)
    wald_table["NTrajectories"] = dat["TrajectoryID"].nunique()
    wald_table["NDistances"] = len(valid_distances)
    term_test_tables.append(wald_table)

all_term_tests = pd.concat(term_test_tables, ignore_index=True)
all_term_tests.to_csv(os.path.join(OUT, "cosine_distance_ols_tests.csv"), index=False)
print(all_term_tests)

# Extract K–K+10 effects and their full covariance matrices
distance_effect_rows, boundary_by_distance_rows, boundary_effect_rows = [], [], []
effect_covariances = {}

for dist in DIST_TYPES:
    if dist not in model_fits: continue
    full_fit, dat = model_fits[dist], model_data[dist]
    valid_distances = [int(distance) for distance in dat["Distance"].cat.categories]
    
    nonboundary_contrasts = []
    for distance in valid_distances:
        nonboundary_row = {"Distance": distance, "AtBoundary": 0}
        nonboundary_contrast = fixed_design_row(full_fit, nonboundary_row)
        estimate, se, t_value, p_value, ci_low, ci_high = contrast_stats(full_fit, nonboundary_contrast)
        nonboundary_contrasts.append(nonboundary_contrast)
        distance_effect_rows.append({"DistType": dist, "Distance": distance, "Position": distance_label(distance), "Estimate": estimate, "SE": se, "T": t_value, "P": p_value, "CILow": ci_low, "CIHigh": ci_high})
        boundary_row = {"Distance": distance, "AtBoundary": 1}
        boundary_contrast = fixed_design_row(full_fit, boundary_row)
        boundary_estimate, boundary_se, boundary_t, boundary_p, boundary_low, boundary_high = contrast_stats(full_fit, boundary_contrast)
        boundary_by_distance_rows.append({
            "DistType": dist, "Distance": distance, 
            "Position": distance_label(distance), 
            "Estimate": boundary_estimate, "SE": boundary_se, "T": boundary_t,
            "P": boundary_p, "CILow": boundary_low, "CIHigh": boundary_high
            })
    
    contrast_matrix = np.vstack(nonboundary_contrasts)
    parameter_covariance = full_fit.cov_params().loc[full_fit.params.index, full_fit.params.index].to_numpy(float)
    effect_covariances[dist] = contrast_matrix @ parameter_covariance @ contrast_matrix.T
    reference_distance = valid_distances[0]
    boundary_0 = fixed_design_row(full_fit, {"Distance": reference_distance, "AtBoundary": 0})
    boundary_1 = fixed_design_row(full_fit, {"Distance": reference_distance, "AtBoundary": 1})
    estimate, se, t_value, p_value, ci_low, ci_high = contrast_stats(full_fit, boundary_1 - boundary_0)
    boundary_effect_rows.append({
        "DistType": dist, "BoundaryModulation": estimate, "SE": se, 
        "T": t_value, "P": p_value, "CILow": ci_low, "CIHigh": ci_high
        })

distance_effects = pd.DataFrame(distance_effect_rows)
boundary_by_distance = pd.DataFrame(boundary_by_distance_rows)
boundary_modulation = pd.DataFrame(boundary_effect_rows)
distance_effects.to_csv(os.path.join(OUT, "cosine_distance_effects_ols.csv"), index=False)
boundary_by_distance.to_csv(os.path.join(OUT, "cosine_boundary_effects_by_distance.csv"), index=False)
boundary_modulation.to_csv(os.path.join(OUT, "cosine_boundary_modulation.csv"), index=False)
print(distance_effects.round(4))
print(boundary_modulation.round(4))

# Candidate distance curves
CURVE_SPECS = {
    "constant": {"function": curve_constant, "parameters": ["baseline"]},
    "linear": {"function": curve_linear, "parameters": ["baseline", "slope"]},
    "quadratic": {"function": curve_quadratic, "parameters": ["baseline", "slope", "quadratic"]},
    "exponential": {"function": curve_exponential, "parameters": ["baseline", "amplitude", "rate"]},
    "power_law": {"function": curve_power_law, "parameters": ["baseline", "amplitude", "alpha"]},
    "log_normal": {"function": curve_log_normal, "parameters": ["baseline", "amplitude", "mu", "sigma"]},
    "log_cauchy": {"function": curve_log_cauchy, "parameters": ["baseline", "amplitude", "mu", "sigma"]},
    "zipf_alekseev": {"function": curve_zipf_alekseev, "parameters": ["baseline", "amplitude", "alpha", "beta"]},
    "broken_stick": {"function": curve_broken_stick, "parameters": ["baseline", "slope_1", "slope_2", "knot"]},
    "exp_plus_exp": {"function": curve_exp_plus_exp, "parameters": ["baseline", "amplitude", "rate_1", "rate_2", "exp_weight"]},
    "exp_plus_power": {"function": curve_exp_plus_power, "parameters": ["baseline", "amplitude", "rate", "alpha", "power_weight"]},
}

curve_comparison_rows, curve_parameter_rows, curve_fits = [], [], {}

for dist in DIST_TYPES:
    effects = distance_effects[distance_effects["DistType"] == dist].sort_values("Distance").copy()
    if dist not in effect_covariances or len(effects) < 4: 
        continue
    
    x = effects["Distance"].to_numpy(float)
    y = effects["Estimate"].to_numpy(float)
    curve_cov = nearest_psd(effect_covariances[dist])
    if curve_cov.shape != (len(x), len(x)): 
        raise ValueError(f"{dist}: covariance dimensions do not match")
    
    curve_fits[dist] = {}
    for model_name in tqdm(CURVE_SPECS, desc=f"{dist} curve fitting"):
        fitted = fit_curve_model(model_name, x, y, curve_cov)
        curve_fits[dist][model_name] = fitted
        curve_comparison_rows.append({
            "DistType": dist, "Model": model_name, "NDistances": len(x),
            "NParameters": len(CURVE_SPECS[model_name]["parameters"]),
            "Minus2LogLik": fitted["Minus2LogLik"], "AIC": fitted["AIC"],
            "AICc": fitted["AICc"], "BIC": fitted["BIC"],
            "RawRMSE": fitted["RawRMSE"],
            "ParamCovCondition": fitted["ParamCovCondition"], "Error": "",
        })
        for parameter, estimate in zip(CURVE_SPECS[model_name]["parameters"], fitted["params"]):
            curve_parameter_rows.append({
                "DistType": dist, "Model": model_name,
                "Parameter": parameter, "Estimate": estimate,
            })

curve_comparison = pd.DataFrame(curve_comparison_rows)
curve_parameters = pd.DataFrame(curve_parameter_rows)
curve_comparison["DeltaAICc"] = np.nan
curve_comparison["AkaikeWeight"] = np.nan

for dist in DIST_TYPES:
    mask = (curve_comparison["DistType"] == dist) & curve_comparison["AICc"].notna()
    if not mask.any(): continue
    delta = curve_comparison.loc[mask, "AICc"] - curve_comparison.loc[mask, "AICc"].min()
    weights = np.exp(-0.5 * delta)
    curve_comparison.loc[mask, "DeltaAICc"] = delta
    curve_comparison.loc[mask, "AkaikeWeight"] = weights / weights.sum()

curve_comparison = curve_comparison.sort_values(["DistType", "AICc"], na_position="last").reset_index(drop=True)
valid_models = curve_comparison.dropna(subset=["AICc"]).copy()
best_models = valid_models.loc[valid_models.groupby("DistType")["AICc"].idxmin()].sort_values("DistType")
selected_parameters = curve_parameters.merge(best_models[["DistType", "Model"]], on=["DistType", "Model"], how="inner")

curve_comparison.to_csv(os.path.join(OUT, "cosine_curve_model_comparison.csv"), index=False)
curve_parameters.to_csv(os.path.join(OUT, "cosine_curve_parameters.csv"), index=False)
best_models.to_csv(os.path.join(OUT, "cosine_best_curve_models.csv"), index=False)

print()
print(curve_comparison[["DistType", "Model", "AICc", "DeltaAICc", "AkaikeWeight", "RawRMSE", "ParamCovCondition", "Error"]].round(4))
print(best_models[["DistType", "Model", "AICc", "AkaikeWeight", "RawRMSE"]].round(4))
print(selected_parameters.round(4))

# Direct CD-versus-RD trajectory comparison
direct_df = distance_data.dropna(subset=[OUTCOME, "DistType", "Distance", "AtBoundary", "TrajectoryID"]).copy()
support = direct_df.groupby(["DistType", "Distance"]).agg(NObservations=(OUTCOME, "size"), NTrajectories=("TrajectoryID", "nunique")).reset_index()
supported = support[support["NTrajectories"] >= MIN_TRAJECTORIES_PER_DISTANCE].copy()

cd_distances = set(supported.loc[supported["DistType"] == "CD", "Distance"].astype(int))
rd_distances = set(supported.loc[supported["DistType"] == "RD", "Distance"].astype(int))
common_distances = sorted(cd_distances & rd_distances)

direct_df = direct_df[direct_df["Distance"].isin(common_distances)].copy()
direct_df["Distance"] = pd.Categorical(direct_df["Distance"], categories=common_distances, ordered=True)
direct_df["DistType"] = pd.Categorical(direct_df["DistType"], categories=["CD", "RD"])
distortion_type_term = "C(DistType, Treatment(reference='CD'))"
direct_formula = f"{OUTCOME} ~ C(Distance) * {distortion_type_term} + AtBoundary * {distortion_type_term}"
direct_fit = smf.ols(direct_formula, data=direct_df).fit(cov_type="cluster", cov_kwds={"groups": direct_df["TrajectoryID"], "use_correction": True}, use_t=True)

with open(os.path.join(OUT, "direct_CD_RD_cosine_ols.txt"), "w", encoding="utf-8") as file:
    file.write(direct_formula + "\n\n" + direct_fit.summary().as_text())

direct_wald = direct_fit.wald_test_terms(skip_single=False, scalar=True).table.copy()
direct_wald.index.name = "Term"
direct_wald = direct_wald.reset_index()
direct_wald.to_csv(os.path.join(OUT, "direct_CD_RD_cosine_wald.csv"), index=False)
print("\nCommon distances:", common_distances)
print(direct_wald)

# Planned contrast: does RD decline less than CD?
if not set(EARLY_DISTANCES + LATE_DISTANCES).issubset(common_distances):
    raise ValueError("Early or late distances are not supported in both distortion types")

def average_direct_design(dist_type, distances, at_boundary=0):
    rows = [{"DistType": dist_type, "Distance": distance, "AtBoundary": at_boundary} for distance in distances]
    return np.mean([fixed_design_row(direct_fit, row) for row in rows], axis=0)

cd_early = average_direct_design("CD", EARLY_DISTANCES)
cd_late = average_direct_design("CD", LATE_DISTANCES)
rd_early = average_direct_design("RD", EARLY_DISTANCES)
rd_late = average_direct_design("RD", LATE_DISTANCES)
persistence_contrast = (rd_late - rd_early) - (cd_late - cd_early)
estimate, se, t_value, p_value, ci_low, ci_high = contrast_stats(direct_fit, persistence_contrast)

direct_persistence = pd.DataFrame([{
    "Test": "RD_vs_CD_difference_in_early_to_late_change",
    "EarlyDistances": f"{distance_label(min(EARLY_DISTANCES))} to {distance_label(max(EARLY_DISTANCES))}",
    "LateDistances": f"{distance_label(min(LATE_DISTANCES))} to {distance_label(max(LATE_DISTANCES))}",
    "Estimate": estimate, "SE": se, "T": t_value, "P": p_value,
    "CILow": ci_low, "CIHigh": ci_high,
}])

direct_persistence.to_csv(os.path.join(OUT, "direct_CD_RD_cosine_persistence.csv"), index=False)

# Boundary effects for CD and RD from the direct model
boundary_rows = []
for dist in DIST_TYPES:
    boundary_0 = average_direct_design(dist, common_distances, at_boundary=0)
    boundary_1 = average_direct_design(dist, common_distances, at_boundary=1)
    estimate, se, t_value, p_value, ci_low, ci_high = contrast_stats(direct_fit, boundary_1 - boundary_0)
    boundary_rows.append({"DistType": dist, "Estimate": estimate, "SE": se, "T": t_value, "P": p_value, "CILow": ci_low, "CIHigh": ci_high})

direct_boundary = pd.DataFrame(boundary_rows)
direct_boundary.to_csv(os.path.join(OUT, "direct_CD_RD_cosine_boundary.csv"), index=False)
print(direct_persistence.round(4))
print(direct_boundary.round(4))

# Main visualization
make_qwen_main_figure(
    trajectory_label="representation-distance",
    top_ylabel="Output-layer cosine distance",
    boundary_ylabel="Additional cosine-distance effect\nat sentence boundary",
    persistence_ylabel="Model-estimated cosine distance",
    output_stem="cosine_distance_results",
)















