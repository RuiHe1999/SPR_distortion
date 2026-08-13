# 1. packages
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from difflib import SequenceMatcher
from patsy import build_design_matrices

import statsmodels.formula.api as smf
from wordfreq import word_frequency, tokenize
from scipy.optimize import curve_fit, OptimizeWarning

import matplotlib.pyplot as plt

# 2. constants
FILE = "data/participant_word_RT.xlsx"
OUT = "results/residual_analysis/decay"
os.makedirs(OUT, exist_ok=True)

SUBJ, STORY, VERSION, WORD = "Participant", "Story", "Story_Version", "Word"
INDEX, SENT, LABEL, RT = "WordIndex", "SentenceIndex", "Word_Condition", "RT"

DIST_TYPES = ["CD", "RD"]
OUTCOME = "resid_logRT"

TRAJECTORY_MIN_DISTANCE = 0
MIN_DISTANCE = 1
MAX_DISTANCE = 10

MIN_ITEMS_PER_DISTANCE = 2
MIN_OBSERVATIONS_PER_DISTANCE = 20

CURVE_EFFECT_MODE = "signed"
N_BOOT = 1000
RANDOM_SEED = 42

USE_SPARSE_RANDOM_EFFECTS = True
INCLUDE_WORDPAIR_RANDOM_INTERCEPT = False
MIXEDLM_OPTIMIZERS = (None,)
MIXEDLM_MAXITER = 300

# 3. functions
def lexical_controls(text, language="en", epsilon=1e-9):
    
    tokens = tokenize(str(text).lower(), language)
    
    if not tokens: 
        return np.nan, 0
    
    frequencies = np.array([word_frequency(token, language) for token in tokens], dtype=float)
    frequencies = np.where(frequencies > 0, frequencies, epsilon)
    lexical_surprisal = -np.mean(np.log2(frequencies))
    word_length = sum(len(token) for token in tokens)
    
    return lexical_surprisal, word_length

def normalize_word(word):
    word = str(word).lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"^[^\w']+|[^\w']+$", "", word)

def make_local_map(original_sentence, distorted_sentence):
    """Align distorted sentence with original sentence"""
    
    if len(original_sentence) == len(distorted_sentence):
        return {j: j for j in range(len(distorted_sentence))}, "position"

    matcher = SequenceMatcher(None, original_sentence["NormWord"].tolist(), distorted_sentence["NormWord"].tolist(), autojunk=False)
    local_map = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                local_map[j1 + offset] = i1 + offset

        elif tag == "replace" and i2 > i1 and j2 > j1:
            for j in range(j1, j2):
                original_j = i1 if j2 - j1 == 1 else i1 + round((j - j1) * (i2 - i1 - 1) / (j2 - j1 - 1))
                local_map[j] = min(max(original_j, i1), i2 - 1)

        elif tag == "insert" and j2 > j1:
            anchor = i1 - 1 if i1 > 0 else (i1 if i1 < len(original_sentence) else None)
            if anchor is not None:
                for j in range(j1, j2):
                    local_map[j] = anchor

    return local_map, "sequence"

def count_consecutive_spans(indices):
    if not indices: 
        return 0
    return 1 + sum(current != previous + 1 for previous, current in zip(indices[:-1], indices[1:]))

def fit_mixedlm(formula, data, reml=False):
    data = data.copy()
    data["_AllGroups"] = "All"
    vc_formula = {"Participant": "0 + C(Participant)"}
    if INCLUDE_WORDPAIR_RANDOM_INTERCEPT: 
        vc_formula["WordPair"] = "0 + C(ItemID)"

    last_fit = None
    last_error = None

    for method in MIXEDLM_OPTIMIZERS:
        start_time = time.perf_counter()

        try:
            model = smf.mixedlm(formula, data=data, groups=data["_AllGroups"], re_formula="0", vc_formula=vc_formula, use_sparse=USE_SPARSE_RANDOM_EFFECTS)
            fit = model.fit(reml=reml, method=method, maxiter=MIXEDLM_MAXITER, disp=False)
            last_fit = fit
            print(f"Finished in {time.perf_counter() - start_time:.1f} seconds; optimizer={method}; converged={fit.converged}")
            if fit.converged: 
                return fit, method, ""
        except Exception as error:
            last_error = str(error)
            print(f"Model failed: {last_error}")

    if last_fit is not None: 
        return last_fit, "nonconverged", last_error or "Model did not converge"
    
    raise RuntimeError(last_error or "MixedLM fitting failed")

def fixed_design_row(fit, row):
    design_info = fit.model.data.design_info
    matrix = build_design_matrices([design_info], pd.DataFrame([row]), return_type="dataframe")[0]
    return matrix.reindex(columns=fit.fe_params.index, fill_value=0.0).to_numpy(float).ravel()

def contrast_stats(fit, contrast):
    contrast = np.asarray(contrast, dtype=float).reshape(1, -1)
    result = fit.t_test(contrast, use_t=False)
    estimate = float(np.squeeze(result.effect))
    standard_error = float(np.squeeze(result.sd))
    z_value = float(np.squeeze(result.tvalue))
    p_value = float(np.squeeze(result.pvalue))
    return estimate, standard_error, z_value, p_value

def effect_contrast(fit, base_row, dist, at_boundary):
    original_row = {**base_row, "Condition": "Original", "AtBoundary": at_boundary}
    distorted_row = {**base_row, "Condition": dist, "AtBoundary": at_boundary}
    return fixed_design_row(fit, distorted_row) - fixed_design_row(fit, original_row)

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
    x = np.maximum(np.asarray(x, float), 1e-12)
    return baseline + amplitude * np.exp(-0.5 * ((np.log(x) - mu) / sigma)**2)

def curve_log_cauchy(x, baseline, amplitude, mu, sigma):
    x = np.maximum(np.asarray(x, float), 1e-12)
    z = np.log(x) - mu
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
        return [-np.inf, -np.inf, np.log(x.min()) - 3.0, 0.05], [np.inf, np.inf, np.log(x.max()) + 3.0, 5.0]
    
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
    
    log_x = np.log(np.asarray(x, float))
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
            "PeakDistance": np.exp(mu),
            "PeakEffect": baseline + amplitude,
            "HalfMaximumLow": np.exp(mu - half_width),
            "HalfMaximumHigh": np.exp(mu + half_width),
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

def distance_label(distance):
    return "K" if int(distance) == 0 else f"K+{int(distance)}"

def scalar_value(value):
    return float(np.asarray(value).squeeze())

def p_label(p_value):
    if not np.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")

def panel_label(axis, label, x=-0.12, y=1.08):
    axis.text(x, y, label, transform=axis.transAxes, fontsize=14,
              fontweight="bold", va="top", ha="left")

def direct_interaction_result(wald_table):
    terms = wald_table["Term"].astype(str)
    mask = terms.str.contains("C\\(Distance\\)", regex=True) & terms.str.contains("DistType") & terms.str.contains(":")
    row = wald_table.loc[mask].iloc[0]
    statistic_column = "statistic" if "statistic" in wald_table.columns else "Statistic"
    p_column = "pvalue" if "pvalue" in wald_table.columns else "P"
    df_column = "df_constraint" if "df_constraint" in wald_table.columns else "DF"
    return scalar_value(row[statistic_column]), int(round(scalar_value(row[df_column]))), scalar_value(row[p_column])

def direct_mean_and_ci(fit, design_row):
    test = fit.t_test(np.asarray(design_row, dtype=float)[np.newaxis, :], use_t=False)
    estimate = scalar_value(test.effect)
    ci_low, ci_high = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1, 2)[0]
    return estimate, float(ci_low), float(ci_high)

# 4. commands
# Read and clean data
df = pd.read_excel(FILE)

required = [SUBJ, STORY, VERSION, WORD, INDEX, SENT, LABEL, RT]
missing = [column for column in required if column not in df.columns]

df[SUBJ] = df[SUBJ].astype(str)
df[STORY] = df[STORY].astype(str).str.strip()
df[VERSION] = df[VERSION].astype(str).str.strip()
df[WORD] = df[WORD].astype(str).str.strip()
df[INDEX] = pd.to_numeric(df[INDEX], errors="coerce")
df[SENT] = pd.to_numeric(df[SENT], errors="coerce")
df[RT] = pd.to_numeric(df[RT], errors="coerce")

df = df.dropna(subset=required).copy()
df = df[df[RT] > 0].copy()
df = df.drop_duplicates([SUBJ, STORY, VERSION, INDEX], keep="first",)

df[INDEX] = df[INDEX].astype(int)
df[SENT] = df[SENT].astype(int)

df["StoryNumber"] = pd.to_numeric(df[STORY].str.extract(r"(\d+)")[0], errors="coerce")
df["VersionOrder"] = df[VERSION].map({"Original": 0, "CD": 1, "RD": 2})

# Calculate lexical surprisal and word length
lexicon = pd.DataFrame({WORD: df[WORD].drop_duplicates()})
lexicon[["FreqH", "Length"]] = pd.DataFrame(lexicon[WORD].apply(lexical_controls).tolist(), index=lexicon.index,)
lexicon.to_csv(os.path.join(OUT, "lexical_controls.csv"),  index=False,)
df = df.merge(lexicon, on=WORD, how="left", validate="many_to_one",)

print("Rows:", len(df))
print("Participants:", df[SUBJ].nunique())
print(df[[WORD, "FreqH", "Length"]].head())

# Lag-1 residualization
df = df.sort_values([SUBJ, "StoryNumber", "VersionOrder", INDEX]).reset_index(drop=True)
df["log_RT"] = np.log(df[RT])

lag_groups = [SUBJ, STORY, VERSION]
df["prev_index"] = df.groupby(lag_groups)[INDEX].shift(1)
df["lag1_logRT"] = df.groupby(lag_groups)["log_RT"].shift(1)
contiguous = (df[INDEX] - df["prev_index"]).eq(1)

gap_lag = (df["prev_index"].notna() & ~contiguous)
df.loc[~contiguous, "lag1_logRT",] = np.nan
df.loc[ gap_lag, [SUBJ, STORY, VERSION, INDEX, "prev_index"],].to_csv(os.path.join(OUT, "gap_lag_qc.csv"), index=False,)

df["item_uid"] = df[STORY] + "_"  + df[VERSION] + "_w" + df[INDEX].astype(str)

resid_df = df.dropna(subset=["log_RT", "lag1_logRT", SUBJ, "item_uid"]).copy()

resid_formula = "log_RT ~ lag1_logRT"
resid_model = smf.mixedlm(resid_formula, data=resid_df, groups=resid_df[SUBJ], re_formula="1")
resid_fit = resid_model.fit()
df[OUTCOME] = np.nan
df.loc[resid_df.index, OUTCOME] = resid_fit.resid

print()
print("Rows:", len(df))
print("Participants:", df[SUBJ].nunique())
print("Rows with residualized log RT:", df[OUTCOME].notna().sum(),)

# Build stimulus template
template = df.groupby([STORY, VERSION, INDEX], as_index=False)\
    .agg(Word=(WORD, "first"), SentenceIndex=(SENT, "first"), Word_Condition=(LABEL, "first"))
template["StoryNumber"] = pd.to_numeric(template[STORY].str.extract(r"(\d+)")[0], errors="coerce")
template["VersionOrder"] = template[VERSION].map({"Original": 0, "CD": 1, "RD": 2})
template = template.sort_values(["StoryNumber", "VersionOrder", INDEX]).reset_index(drop=True)

template["NormWord"] = template["Word"].apply(normalize_word)
template["Word_Condition"] = template["Word_Condition"].astype(str).str.replace("CD_", "", regex=False).str.replace("RD_", "", regex=False).str.replace("K+END", "K_END", regex=False)
template["Labels"] = template["Word_Condition"].apply(lambda value: [label.strip() for label in re.split(r"[|,]", str(value)) if label.strip()])

print()
print("Template rows:", len(template))
print(template.groupby(VERSION).size())
print(template[[STORY, VERSION, INDEX, "Word", "NormWord", "Labels"]].head(10))

# Build trajectories
trajectory_rows = []
trajectory_qc_rows = []
stories = sorted(template[STORY].unique(), key=lambda value: int(re.search(r"\d+", value).group()))

for dist in DIST_TYPES:
    for story in tqdm(stories, desc=f"Building {dist} trajectories"):
        original_story = template[(template[STORY] == story) & (template[VERSION] == "Original")]
        distorted_story = template[(template[STORY] == story) & (template[VERSION] == dist)]
        if original_story.empty or distorted_story.empty: 
            continue

        shared_sentences = sorted(set(original_story[SENT]) & set(distorted_story[SENT]))

        for sentence in shared_sentences:
            original_sentence = original_story[original_story[SENT] == sentence].sort_values(INDEX).reset_index(drop=True)
            distorted_sentence = distorted_story[distorted_story[SENT] == sentence].sort_values(INDEX).reset_index(drop=True)

            k_local_indices = [j for j, labels in enumerate(distorted_sentence["Labels"]) if "K" in labels]
            if not k_local_indices: 
                continue

            local_map, alignment_method = make_local_map(original_sentence, distorted_sentence)

            anchor_j = max(k_local_indices)
            anchor_word_index = int(distorted_sentence.iloc[anchor_j][INDEX])
            perturbation_id = f"{dist}_{story}_s{int(sentence)}_k{anchor_word_index}"
            n_spans = count_consecutive_spans(k_local_indices)
            sentence_end_j = len(distorted_sentence) - 1
            end_distance = sentence_end_j - anchor_j
            mapped = 0

            for j in range(anchor_j, len(distorted_sentence)):
                distance = j - anchor_j
                if distance < TRAJECTORY_MIN_DISTANCE: 
                    continue
                if MAX_DISTANCE is not None and distance > MAX_DISTANCE: 
                    continue

                original_j = local_map.get(j)
                if original_j is None or not 0 <= original_j < len(original_sentence): 
                    continue

                distorted_row = distorted_sentence.iloc[j]
                original_row = original_sentence.iloc[original_j]

                trajectory_rows.append({
                    "DistType": dist,
                    STORY: story,
                    "MapSentenceIndex": int(sentence),
                    "PerturbationID": perturbation_id,
                    "AnchorDistWordIndex": anchor_word_index,
                    "Distance": int(distance),
                    "AtBoundary": int(j == sentence_end_j),
                    "EndDistance": int(end_distance),
                    "N_K_Spans": int(n_spans),
                    "DistWordIndex": int(distorted_row[INDEX]),
                    "OriginalWordIndex": int(original_row[INDEX]),
                    "DistWord": distorted_row["Word"],
                    "OriginalWord": original_row["Word"],
                })

                mapped += 1

            trajectory_qc_rows.append({
                "DistType": dist,
                STORY: story,
                "SentenceIndex": int(sentence),
                "PerturbationID": perturbation_id,
                "N_K_Tokens": len(k_local_indices),
                "N_K_Spans": int(n_spans),
                "AnchorDistWordIndex": anchor_word_index,
                "WordsAfterFinalK": int(end_distance),
                "MappedResponses": int(mapped),
                "OriginalSentenceLength": len(original_sentence),
                "DistortedSentenceLength": len(distorted_sentence),
                "AlignmentMethod": alignment_method,
            })

trajectory_map = pd.DataFrame(trajectory_rows).drop_duplicates()
trajectory_qc = pd.DataFrame(trajectory_qc_rows)

if trajectory_map.empty:
    raise ValueError("No post-perturbation trajectories were created. Check exact K labels.")

trajectory_map.to_csv(os.path.join(OUT, "perturbation_trajectory_map.csv"), index=False)
trajectory_qc.to_csv(os.path.join(OUT, "perturbation_trajectory_qc.csv"), index=False)

# exclude distance at 0 
trajectory_map = trajectory_map[trajectory_map["Distance"].between(MIN_DISTANCE, MAX_DISTANCE)]

print()
print(trajectory_qc.groupby("DistType", as_index=False).agg(
    NTrajectories=("PerturbationID", "nunique"),
    MedianEndDistance=("WordsAfterFinalK", "median"),
    MaxEndDistance=("WordsAfterFinalK", "max"),
    MultiSpanSentences=("N_K_Spans", lambda values: int((values > 1).sum())),
))

# Match participant-level distorted and Original observations
matched_tables = []

for dist in DIST_TYPES:
    current_map = trajectory_map[trajectory_map["DistType"] == dist].copy()
    current_map["ItemID"] = current_map["PerturbationID"] + "_d" + current_map["Distance"].astype(str) + "_dw" + current_map["DistWordIndex"].astype(str) + "_ow" + current_map["OriginalWordIndex"].astype(str)

    distorted = df[df[VERSION] == dist].merge(current_map, left_on=[STORY, INDEX], right_on=[STORY, "DistWordIndex"], how="inner")
    original = df[df[VERSION] == "Original"].merge(current_map, left_on=[STORY, INDEX], right_on=[STORY, "OriginalWordIndex"], how="inner")

    distorted["Condition"] = dist
    original["Condition"] = "Original"
    matched_tables.extend([distorted, original])

local_df = pd.concat(matched_tables, ignore_index=True)
local_df["ItemID"] = local_df["ItemID"].astype(str)
local_df["PerturbationID"] = local_df["PerturbationID"].astype(str)
local_df["Distance"] = pd.to_numeric(local_df["Distance"], errors="coerce").astype(int)
local_df["AtBoundary"] = pd.to_numeric(local_df["AtBoundary"], errors="coerce").astype(int)

item_check = local_df.groupby(["DistType", "ItemID"])["Condition"].nunique().reset_index(name="NConditions")
valid_items = item_check.loc[item_check["NConditions"] == 2, "ItemID"]
local_df = local_df[local_df["ItemID"].isin(valid_items)].copy()

local_df.to_csv(os.path.join(OUT, "matched_distance_data.csv"), index=False)

distance_counts = local_df.groupby(["DistType", "Condition", "Distance", "AtBoundary"], as_index=False).agg(
    NObservations=(OUTCOME, "count"),
    NParticipants=(SUBJ, "nunique"),
    NItems=("ItemID", "nunique"),
    NTrajectories=("PerturbationID", "nunique"),
)

distance_counts.to_csv(os.path.join(OUT, "distance_counts.csv"), index=False)

print()
print("Matched rows:", len(local_df))
print(distance_counts.to_string(index=False))

# Fit CD and RD joint models
term_test_tables = []
model_fits = {}
model_data = {}

for dist in DIST_TYPES:
    dat = local_df[local_df["DistType"] == dist].dropna(subset=[OUTCOME, "FreqH", "Length", INDEX, SUBJ, "ItemID", "PerturbationID", "Condition", "Distance", "AtBoundary"]).copy()

    support = dat.groupby("Distance").agg(NConditions=("Condition", "nunique"), NObservations=(OUTCOME, "size"), NItems=("ItemID", "nunique")).reset_index()
    valid_distances = support.loc[(support["NConditions"] == 2) & (support["NObservations"] >= MIN_OBSERVATIONS_PER_DISTANCE) & (support["NItems"] >= MIN_ITEMS_PER_DISTANCE), "Distance"].astype(int).sort_values().tolist()
    dat = dat[dat["Distance"].isin(valid_distances)].copy()

    if MIN_DISTANCE not in valid_distances or len(valid_distances) < 3:
        print(f"{dist}: insufficient supported distances")
        continue

    dat["Distance"] = pd.Categorical(dat["Distance"], categories=valid_distances, ordered=True)
    dat["Condition"] = pd.Categorical(dat["Condition"], categories=["Original", dist])

    condition_term = "C(Condition, Treatment(reference='Original'))"
    full_formula = f"{OUTCOME} ~ C(Distance) * {condition_term} + AtBoundary * {condition_term} + WordIndex + FreqH + Length"

    print(f"\n{dist}: N={len(dat)}, participants={dat[SUBJ].nunique()}, items={dat['ItemID'].nunique()}, trajectories={dat['PerturbationID'].nunique()}")
    full_fit, optimizer, fit_error = fit_mixedlm(full_formula, dat, reml=False)

    model_fits[dist] = full_fit
    model_data[dist] = dat

    with open(os.path.join(OUT, f"{dist}_joint_mixedlm.txt"), "w", encoding="utf-8") as file:
        file.write(full_formula + "\n\n" + full_fit.summary().as_text())

    wald_result = full_fit.wald_test_terms(skip_single=False, scalar=True)
    wald_table = wald_result.table.copy()
    wald_table.index.name = "Term"
    wald_table = wald_table.reset_index()
    wald_table = wald_table.rename(columns={"statistic": "Statistic", "pvalue": "P", "df_constraint": "DF"})

    wald_table.insert(0, "DistType", dist)
    wald_table["N"] = len(dat)
    wald_table["NParticipants"] = dat[SUBJ].nunique()
    wald_table["NItems"] = dat["ItemID"].nunique()
    wald_table["NTrajectories"] = dat["PerturbationID"].nunique()
    wald_table["NDistances"] = len(valid_distances)
    wald_table["Converged"] = bool(full_fit.converged)
    wald_table["Optimizer"] = optimizer
    wald_table["Error"] = fit_error

    term_test_tables.append(wald_table)

all_term_tests = pd.concat(term_test_tables, ignore_index=True)
all_term_tests.to_csv(os.path.join(OUT, "all_wald_term_tests.csv"), index=False)

distance_interaction = all_term_tests["Term"].str.contains("C(Distance)", regex=False) & all_term_tests["Term"].str.contains("C(Condition", regex=False) & all_term_tests["Term"].str.contains(":", regex=False)
boundary_interaction = all_term_tests["Term"].str.contains("AtBoundary", regex=False) & all_term_tests["Term"].str.contains("C(Condition", regex=False) & all_term_tests["Term"].str.contains(":", regex=False)

joint_tests = all_term_tests[distance_interaction | boundary_interaction].copy()
joint_tests["Test"] = np.where(distance_interaction[distance_interaction | boundary_interaction], "Condition_x_Distance", "Condition_x_Boundary")
joint_tests.to_csv(os.path.join(OUT, "joint_model_tests.csv"), index=False)

print("\nDistance and boundary interaction tests:")
print(joint_tests[["DistType", "Test", "Statistic", "DF", "P", "Converged"]].to_string(index=False))

# Extract distance and boundary effects from the fitted MixedLMs
distance_effect_rows = []
boundary_by_distance_rows = []
boundary_effect_rows = []
effect_covariances = {}

for dist in DIST_TYPES:
    if dist not in model_fits: continue

    full_fit = model_fits[dist]
    dat = model_data[dist]
    valid_distances = [int(distance) for distance in dat["Distance"].cat.categories]
    base_values = {"WordIndex": float(dat[INDEX].mean()), "FreqH": float(dat["FreqH"].mean()), "Length": float(dat["Length"].mean())}
    nonboundary_contrasts = []

    for distance in valid_distances:
        base_row = {**base_values, "Distance": distance}

        nonboundary_contrast = effect_contrast(full_fit, base_row, dist, at_boundary=0)
        beta, standard_error, z_value, p_value = contrast_stats(full_fit, nonboundary_contrast)
        nonboundary_contrasts.append(nonboundary_contrast)

        distance_effect_rows.append({
            "DistType": dist, "Distance": distance, "Position": f"K+{distance}",
            "Beta": beta, "SE": standard_error, "Z": z_value, "P": p_value,
            "CILow": beta - 1.96 * standard_error, "CIHigh": beta + 1.96 * standard_error,
            "PercentChange": (np.exp(beta) - 1) * 100, "EffectContext": "Non-boundary"
        })

        boundary_contrast = effect_contrast(full_fit, base_row, dist, at_boundary=1)
        boundary_beta, boundary_se, boundary_z, boundary_p = contrast_stats(full_fit, boundary_contrast)

        boundary_by_distance_rows.append({
            "DistType": dist, "Distance": distance, "Position": distance_label(distance),
            "Beta": boundary_beta, "SE": boundary_se, "Z": boundary_z, "P": boundary_p,
            "CILow": boundary_beta - 1.96 * boundary_se, "CIHigh": boundary_beta + 1.96 * boundary_se,
            "PercentChange": (np.exp(boundary_beta) - 1) * 100
        })

    contrast_matrix = np.vstack(nonboundary_contrasts)
    fixed_names = list(full_fit.fe_params.index)
    fixed_covariance = full_fit.cov_params().loc[fixed_names, fixed_names].to_numpy(float)
    effect_covariances[dist] = contrast_matrix @ fixed_covariance @ contrast_matrix.T

    reference_row = {**base_values, "Distance": valid_distances[0]}
    boundary_modulation_contrast = effect_contrast(full_fit, reference_row, dist, at_boundary=1) - effect_contrast(full_fit, reference_row, dist, at_boundary=0)
    modulation_beta, modulation_se, modulation_z, modulation_p = contrast_stats(full_fit, boundary_modulation_contrast)

    boundary_effect_rows.append({
        "DistType": dist, 
        "BoundaryModulationBeta": modulation_beta,
        "SE": modulation_se, 
        "Z": modulation_z, 
        "P_Wald": modulation_p,
        "CILow": modulation_beta - 1.96 * modulation_se,
        "CIHigh": modulation_beta + 1.96 * modulation_se,
        "AdditionalPercentChangeAtBoundary": (np.exp(modulation_beta) - 1) * 100
    })

distance_effects = pd.DataFrame(distance_effect_rows)
boundary_by_distance = pd.DataFrame(boundary_by_distance_rows)
boundary_modulation = pd.DataFrame(boundary_effect_rows)

distance_effects.to_csv(os.path.join(OUT, "distance_effects_mixedlm.csv"), index=False)
boundary_by_distance.to_csv(os.path.join(OUT, "boundary_effects_by_distance.csv"), index=False)
boundary_modulation.to_csv(os.path.join(OUT, "boundary_modulation.csv"), index=False)

print("\nNon-boundary distance effects:")
print(distance_effects[["DistType", "Position", "Beta", "SE", "P", "PercentChange"]].round(4).to_string(index=False))

print("\nBoundary modulation:")
print(boundary_modulation.round(4).to_string(index=False))

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

# Fit and compare all 11 curve families
curve_comparison_rows = []
curve_parameter_rows = []
curve_fits = {}

for dist in DIST_TYPES:
    effects = distance_effects[distance_effects["DistType"] == dist].sort_values("Distance").copy()
    if dist not in effect_covariances or len(effects) < 4:
        continue
    x = effects["Distance"].to_numpy(float)
    signed_y = effects["Beta"].to_numpy(float)
    signed_cov = effect_covariances[dist]
    if signed_cov.shape != (len(x), len(x)):
        raise ValueError(f"{dist}: effect covariance does not match the number of distances")
    if CURVE_EFFECT_MODE == "absolute":
        signs = np.where(signed_y >= 0, 1.0, -1.0)
        sign_matrix = np.diag(signs)
        y = np.abs(signed_y)
        curve_cov = sign_matrix @ signed_cov @ sign_matrix
    else:
        y = signed_y
        curve_cov = signed_cov
    curve_fits[dist] = {}
    for model_name in tqdm(CURVE_SPECS, desc=dist):
        try:
            fitted = fit_curve_model(model_name, x, y, curve_cov)
            curve_fits[dist][model_name] = fitted
            curve_comparison_rows.append({"DistType": dist, "EffectMode": CURVE_EFFECT_MODE, "Model": model_name, "NDistances": len(x), "NParameters": len(CURVE_SPECS[model_name]["parameters"]), "Minus2LogLik": fitted["Minus2LogLik"], "AIC": fitted["AIC"], "AICc": fitted["AICc"], "BIC": fitted["BIC"], "RawRMSE": fitted["RawRMSE"], "ParamCovCondition": fitted["ParamCovCondition"], "Error": ""})
            for parameter, estimate in zip(CURVE_SPECS[model_name]["parameters"], fitted["params"]):
                curve_parameter_rows.append({"DistType": dist, "Model": model_name, "Parameter": parameter, "Estimate": estimate})
        except Exception as exc:
            curve_comparison_rows.append({"DistType": dist, "EffectMode": CURVE_EFFECT_MODE, "Model": model_name, "NDistances": len(x), "NParameters": len(CURVE_SPECS[model_name]["parameters"]), "Minus2LogLik": np.nan, "AIC": np.nan, "AICc": np.nan, "BIC": np.nan, "RawRMSE": np.nan, "ParamCovCondition": np.nan, "Error": str(exc)})

curve_comparison = pd.DataFrame(curve_comparison_rows)
curve_parameters = pd.DataFrame(curve_parameter_rows)
curve_comparison["DeltaAICc"] = np.nan
curve_comparison["AkaikeWeight"] = np.nan

for dist in DIST_TYPES:
    mask = (curve_comparison["DistType"] == dist) & curve_comparison["AICc"].notna()
    if not mask.any():
        continue
    delta = curve_comparison.loc[mask, "AICc"] - curve_comparison.loc[mask, "AICc"].min()
    weights = np.exp(-0.5 * delta)
    curve_comparison.loc[mask, "DeltaAICc"] = delta
    curve_comparison.loc[mask, "AkaikeWeight"] = weights / weights.sum()

curve_comparison = curve_comparison.sort_values(["DistType", "AICc"], na_position="last").reset_index(drop=True)
curve_comparison.to_csv(os.path.join(OUT, "curve_model_comparison.csv"), index=False)
curve_parameters.to_csv(os.path.join(OUT, "curve_parameters.csv"), index=False)

print()
print(curve_comparison[["DistType", "Model", "AICc", "DeltaAICc", "AkaikeWeight", "RawRMSE", "ParamCovCondition", "Error"]].round(4).to_string(index=False))

# lowest AICc
valid_models = curve_comparison.dropna(subset=["AICc"]).copy()
best_models = valid_models.loc[valid_models.groupby("DistType")["AICc"].idxmin()].sort_values("DistType")

print()
print(best_models[["DistType", "Model", "AICc", "AkaikeWeight", "RawRMSE"]].round(4).to_string(index=False))

# print parameters and peak distance 
selected_parameters = curve_parameters.merge(best_models[["DistType", "Model"]], on=["DistType", "Model"], how="inner")
print()
print(selected_parameters.round(4).to_string(index=False))

cd_params = dict(zip(
    selected_parameters[selected_parameters["DistType"] == "CD"]["Parameter"], 
    selected_parameters[selected_parameters["DistType"] == "CD"]["Estimate"])
    )
rd_params = dict(zip(
    selected_parameters[selected_parameters["DistType"] == "RD"]["Parameter"], 
    selected_parameters[selected_parameters["DistType"] == "RD"]["Estimate"])
    )

cd_peak_distance = np.exp(cd_params["mu"])
cd_peak_effect = cd_params["baseline"] + cd_params["amplitude"]
half_width = cd_params["sigma"] * np.sqrt(2 * np.log(2))
cd_half_low = np.exp(cd_params["mu"] - half_width)
cd_half_high = np.exp(cd_params["mu"] + half_width)

cd_effect_k1 = float(curve_log_normal(np.array([1.0]), cd_params["baseline"], cd_params["amplitude"], cd_params["mu"], cd_params["sigma"])[0])
rd_effect_k1 = float(curve_linear(np.array([1.0]), rd_params["baseline"], rd_params["slope"])[0])
cd_effect_k10 = float(curve_log_normal(np.array([10.0]), cd_params["baseline"], cd_params["amplitude"], cd_params["mu"], cd_params["sigma"])[0])
rd_effect_k10 = float(curve_linear(np.array([10.0]), rd_params["baseline"], rd_params["slope"])[0])

print()
print(f"CD peak distance: K+{cd_peak_distance:.2f}")
print(f"CD peak effect: {cd_peak_effect:.4f}")
print(f"CD half-maximum interval: K+{cd_half_low:.2f} to K+{cd_half_high:.2f}")
print(f"CD predicted effect at K+1: {cd_effect_k1:.4f}")
print(f"CD predicted effect at K+10: {cd_effect_k10:.4f}")
print(f"RD change per additional word: {rd_params['slope']:.4f}")
print(f"RD predicted effect at K+1: {rd_effect_k1:.4f}")
print(f"RD predicted effect at K+10: {rd_effect_k10:.4f}")
print(f"RD total change from K+1 to K+10: {rd_effect_k10 - rd_effect_k1:.4f}")

# Bootstrap confidence intervals for selected curve models
best_model_names = best_models.set_index("DistType")["Model"].to_dict()
bootstrap_rows = []
point_rows = []
rng = np.random.default_rng(RANDOM_SEED)

for dist in DIST_TYPES:
    effects = distance_effects[distance_effects["DistType"] == dist].sort_values("Distance").copy()
    model_name = best_model_names[dist]
    x = effects["Distance"].to_numpy(float)
    signed_mean = effects["Beta"].to_numpy(float)
    signed_cov = nearest_psd(effect_covariances[dist])
    if CURVE_EFFECT_MODE == "absolute":
        signs = np.where(signed_mean >= 0, 1.0, -1.0)
        sign_matrix = np.diag(signs)
        y = np.abs(signed_mean)
        curve_cov = nearest_psd(sign_matrix @ signed_cov @ sign_matrix)
    else:
        y = signed_mean
        curve_cov = signed_cov
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        point_fit = fit_curve_model(model_name, x, y, curve_cov)
    point_metrics = calculate_curve_metrics(model_name, point_fit["params"], x.min(), x.max())
    for metric, estimate in point_metrics.items():
        point_rows.append({"DistType": dist, "Model": model_name, "Metric": metric, "PointEstimate": estimate})
    samples = rng.multivariate_normal(signed_mean, signed_cov, size=N_BOOT)
    successful_fits = 0
    for bootstrap_number, sample in enumerate(tqdm(samples, desc=f"Bootstrap {dist}"), start=1):
        sample_y = np.abs(sample) if CURVE_EFFECT_MODE == "absolute" else sample
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                bootstrap_fit = fit_curve_model(model_name, x, sample_y, curve_cov)
            metrics = calculate_curve_metrics(model_name, bootstrap_fit["params"], x.min(), x.max())
            if not np.all(np.isfinite(list(metrics.values()))):
                continue
            successful_fits += 1
            for metric, estimate in metrics.items():
                bootstrap_rows.append({"DistType": dist, "Model": model_name, "Bootstrap": bootstrap_number, "Metric": metric, "Estimate": estimate})
        except Exception:
            continue
    print(f"{dist}: {successful_fits}/{N_BOOT} successful bootstrap fits")

bootstrap_draws = pd.DataFrame(bootstrap_rows)
point_estimates = pd.DataFrame(point_rows)

bootstrap_quantiles = bootstrap_draws.groupby(["DistType", "Model", "Metric"])["Estimate"].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
bootstrap_quantiles.columns = ["DistType", "Model", "Metric", "CILow", "BootstrapMedian", "CIHigh"]

bootstrap_counts = bootstrap_draws.groupby(["DistType", "Model", "Metric"]).size().reset_index(name="NBootstrap")
bootstrap_summary = point_estimates.merge(bootstrap_quantiles, on=["DistType", "Model", "Metric"], how="left")
bootstrap_summary = bootstrap_summary.merge(bootstrap_counts, on=["DistType", "Model", "Metric"], how="left")

bootstrap_draws.to_csv(os.path.join(OUT, "selected_curve_bootstrap_draws.csv"), index=False)
bootstrap_summary.to_csv(os.path.join(OUT, "selected_curve_bootstrap_summary.csv"), index=False)

print()
print(bootstrap_summary.round(4).to_string(index=False))

# Direct CD-versus-RD trajectory comparison
direct_df = local_df[local_df["Condition"].isin(["CD", "RD"])].dropna(
    subset=[OUTCOME, "FreqH", "Length", INDEX, SUBJ, "ItemID", "PerturbationID", "DistType", "Distance", "AtBoundary"]
).copy()

support = direct_df.groupby(["DistType", "Distance"]).agg(
    NObservations=(OUTCOME, "size"),
    NItems=("ItemID", "nunique"),
).reset_index()

supported = support.loc[
    (support["NObservations"] >= MIN_OBSERVATIONS_PER_DISTANCE)
    & (support["NItems"] >= MIN_ITEMS_PER_DISTANCE)
].copy()

cd_distances = set(supported.loc[supported["DistType"] == "CD", "Distance"].astype(int))
rd_distances = set(supported.loc[supported["DistType"] == "RD", "Distance"].astype(int))
common_distances = sorted(cd_distances & rd_distances)
direct_df = direct_df[direct_df["Distance"].isin(common_distances)].copy()

direct_df["Distance"] = pd.Categorical(direct_df["Distance"], categories=common_distances, ordered=True)
direct_df["DistType"] = pd.Categorical(direct_df["DistType"], categories=["CD", "RD"])

distortion_type_term = "C(DistType, Treatment(reference='CD'))"
direct_formula = f"{OUTCOME} ~ C(Distance) * {distortion_type_term} + AtBoundary * {distortion_type_term} + WordIndex + FreqH + Length"

print()
print(f"Direct CD-RD model: N={len(direct_df)}, participants={direct_df[SUBJ].nunique()}, trajectories={direct_df['PerturbationID'].nunique()}")
print("Common distances:", common_distances)

direct_fit, direct_optimizer, direct_error = fit_mixedlm(direct_formula, direct_df, reml=False)

with open(os.path.join(OUT, "direct_CD_RD_trajectory_model.txt"), "w", encoding="utf-8") as file:
    file.write(direct_formula + "\n\n" + direct_fit.summary().as_text())

direct_wald = direct_fit.wald_test_terms().table.copy()
direct_wald.index.name = "Term"
direct_wald = direct_wald.reset_index()
direct_wald.to_csv(os.path.join(OUT, "direct_CD_RD_wald_terms.csv"), index=False)

print()
print("Direct-model Wald tests:")
print(direct_wald)

# Planned contrast: does RD decline less from early to late than CD?
EARLY_DISTANCES = [1, 2, 3]
LATE_DISTANCES = [8, 9, 10]

if not set(EARLY_DISTANCES + LATE_DISTANCES).issubset(common_distances):
    raise ValueError("The requested early or late distances are not supported in both distortion types")

base_values = {
    "WordIndex": float(direct_df[INDEX].mean()),
    "FreqH": float(direct_df["FreqH"].mean()),
    "Length": float(direct_df["Length"].mean()),
    "AtBoundary": 0,
}

cd_early = np.mean([
    fixed_design_row(
        direct_fit, {**base_values, "DistType": "CD", "Distance": distance}
        ) for distance in EARLY_DISTANCES
    ], axis=0)

cd_late = np.mean([
    fixed_design_row(
        direct_fit, {**base_values, "DistType": "CD", "Distance": distance}
        ) for distance in LATE_DISTANCES
    ], axis=0)

rd_early = np.mean([
    fixed_design_row(
        direct_fit, {**base_values, "DistType": "RD", "Distance": distance}
        ) for distance in EARLY_DISTANCES
    ], axis=0)

rd_late = np.mean([
    fixed_design_row(
        direct_fit, {**base_values, "DistType": "RD", "Distance": distance}
        ) for distance in LATE_DISTANCES
    ], axis=0)

persistence_contrast = (rd_late - rd_early) - (cd_late - cd_early)
persistence_test = direct_fit.t_test(persistence_contrast[np.newaxis, :], use_t=False)

estimate = float(np.asarray(persistence_test.effect).squeeze())
se = float(np.asarray(persistence_test.sd).squeeze())
z = float(np.asarray(persistence_test.tvalue).squeeze())
p_value = float(np.asarray(persistence_test.pvalue).squeeze())
ci_low, ci_high = np.asarray(persistence_test.conf_int(alpha=0.05)).reshape(-1, 2)[0]

direct_persistence = pd.DataFrame([{
    "Test": "RD_vs_CD_difference_in_early_to_late_change",
    "EarlyDistances": f"{distance_label(min(EARLY_DISTANCES))} to {distance_label(max(EARLY_DISTANCES))}",
    "LateDistances": f"{distance_label(min(LATE_DISTANCES))} to {distance_label(max(LATE_DISTANCES))}",
    "Estimate": estimate,
    "SE": se,
    "Z": z,
    "P": p_value,
    "CILow": ci_low,
    "CIHigh": ci_high,
    "Converged": bool(direct_fit.converged),
    "Optimizer": direct_optimizer,
    "Error": direct_error,
}])

direct_persistence.to_csv(os.path.join(OUT, "direct_CD_RD_persistence_test.csv"), index=False)

print()
print("Direct persistence contrast:")
print(direct_persistence.round(4))

# Main-text visualization: 
# selected curves, model comparison, boundary effect, and the planned RD-vs-CD persistence contrast
required_plot_objects = [
    "distance_effects", "curve_comparison", "curve_fits", "best_models",
    "boundary_modulation", "direct_wald", "direct_persistence", "direct_fit",
    "cd_early", "cd_late", "rd_early", "rd_late",
]

missing_plot_objects = [name for name in required_plot_objects if name not in globals()]
if missing_plot_objects:
    raise RuntimeError(
        "Run the full analysis through the direct persistence contrast first. "
        f"Missing objects: {missing_plot_objects}"
    )

COLORS = {"CD": "#69B3A2", "RD": "#AD5B70"}

MODEL_LABELS = {
    "constant": "Constant", "linear": "Linear", "quadratic": "Quadratic",
    "exponential": "Exponential", "log_normal": "Log-normal",
    "log_cauchy": "Log-Cauchy", "zipf_alekseev": "Zipf-Alekseev",
    "broken_stick": "Broken stick", "power_law": "Power law",
    "exp_plus_exp": "Exp. + exp.", "exp_plus_power": "Exp. + power",
}

MODEL_ORDER = [
    "log_normal", "log_cauchy", "zipf_alekseev", "broken_stick",
    "exponential", "power_law", "exp_plus_exp", "exp_plus_power",
    "linear", "quadratic", "constant",
]

BAR_COLORS = {"CD": "#B8DCD4", "RD": "#DDB7C1"}

style = {
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "svg.fonttype": "none",
}

with plt.rc_context(style):
    fig = plt.figure(figsize=(13.2, 12.8), facecolor="white", constrained_layout=True)
    outer = fig.add_gridspec(
        3, 2, height_ratios=[1.02, 1.10, 1.02], width_ratios=[1, 1],
        hspace=0.12, wspace=0.10,
    )
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
        best_parameters = curve_fits[dist][best_name]["params"]
        x_grid = np.linspace(observed["Distance"].min(), observed["Distance"].max(), 500)
        y_grid = CURVE_SPECS[best_name]["function"](x_grid, *best_parameters)
        color = COLORS[dist]

        axis.axhline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=0)
        axis.errorbar(
            observed["Distance"], observed["Beta"],
            yerr=[observed["Beta"] - observed["CILow"], observed["CIHigh"] - observed["Beta"]],
            fmt="o", color=color, ecolor=color, markersize=6, capsize=3,
            elinewidth=1.4, label="Mixed-model estimate", zorder=3,
        )
        axis.plot(
            x_grid, y_grid, color=color, linewidth=2.8,
            label=f"Best fit: {MODEL_LABELS.get(best_name, best_name)}", zorder=2,
        )
        
        axis.set_title(f"{dist} trajectory")
        axis.set_xlabel("Word distance from final K token")
        
        ticks = list(range(MIN_DISTANCE, MAX_DISTANCE + 1))
        axis.set_xticks(ticks)
        axis.set_xticklabels([distance_label(d) for d in ticks])
        
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        axis.legend(frameon=False, loc="upper right")
        axis.text(
            0.98, 0.03,
            f"Best-model AICc = {best_row['AICc']:.2f}",
            transform=axis.transAxes, ha="right", va="bottom", color="#555555",
        )
        top_bounds.extend(observed["CILow"].tolist() + observed["CIHigh"].tolist() + y_grid.tolist())

    ax_cd.set_ylabel("Distortion − Original\n(residualized log RT)")
    plt.setp(ax_rd.get_yticklabels(), visible=False)
    finite_bounds = np.asarray(top_bounds, dtype=float)
    finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
    y_low = min(0.0, finite_bounds.min())
    y_high = max(0.0, finite_bounds.max())
    y_pad = max((y_high - y_low) * 0.12, 0.005)
    ax_cd.set_ylim(y_low - y_pad, y_high + y_pad)
    panel_label(ax_cd, "A")
    panel_label(ax_rd, "B")

    interaction_chi2, interaction_df, interaction_p = direct_interaction_result(direct_wald)
    fig.text(
        0.5, 1.005,
        f"Direct distance × distortion-type interaction: χ²({interaction_df}) = {interaction_chi2:.2f}, {p_label(interaction_p)}",
        ha="center", va="bottom", fontsize=11.5,
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

    boundary_plot = boundary_modulation.set_index("DistType").loc[DIST_TYPES].reset_index()
    x_boundary = np.arange(len(boundary_plot))
    for index, row in boundary_plot.iterrows():
        color = COLORS[row["DistType"]]
        ax_boundary.errorbar(
            index, row["BoundaryModulationBeta"],
            yerr=[[row["BoundaryModulationBeta"] - row["CILow"]], [row["CIHigh"] - row["BoundaryModulationBeta"]]],
            fmt="o", color=color, ecolor=color, markersize=8, capsize=4, elinewidth=1.7,
        )
    ax_boundary.axhline(0, color="#777777", linestyle="--", linewidth=1.2)
    ax_boundary.set_xticks(x_boundary, boundary_plot["DistType"])
    ax_boundary.set_ylabel("Additional distortion effect\nat sentence boundary")
    
    terms = direct_wald["Term"].astype(str)
    mask = terms.str.contains("AtBoundary", regex=False) & terms.str.contains("DistType", regex=False) & terms.str.contains(":", regex=False)
    boundary_row = direct_wald.loc[mask].iloc[0]
    boundary_p = scalar_value(boundary_row["pvalue"])

    ax_boundary.set_title(f"Boundary modulation: interaction {p_label(float(boundary_p))}")
    ax_boundary.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    boundary_range = max(boundary_plot["CIHigh"].max() - boundary_plot["CILow"].min(), 0.01)
    for index, row in boundary_plot.iterrows():
        ax_boundary.text(
            index, row["CIHigh"] + boundary_range * 0.06,
            p_label(float(row["P_Wald"])), ha="center", va="bottom", fontsize=9,
        )
    ax_boundary.set_ylim(
        min(0, boundary_plot["CILow"].min()) - boundary_range * 0.12,
        boundary_plot["CIHigh"].max() + boundary_range * 0.22,
    )
    panel_label(ax_boundary, "E", x=-0.05, y=1.08)

    persistence = direct_persistence.iloc[0]
    persistence_p = float(persistence["P"])
    direct_windows = {
        "CD": [direct_mean_and_ci(direct_fit, cd_early), direct_mean_and_ci(direct_fit, cd_late)],
        "RD": [direct_mean_and_ci(direct_fit, rd_early), direct_mean_and_ci(direct_fit, rd_late)],
    }
    x_windows = np.array([0, 1], dtype=float)
    for dist, marker in [("CD", "o"), ("RD", "s")]:
        estimates = np.array([value[0] for value in direct_windows[dist]])
        lows = np.array([value[1] for value in direct_windows[dist]])
        highs = np.array([value[2] for value in direct_windows[dist]])
        ax_persistence.errorbar(
            x_windows, estimates, yerr=[estimates - lows, highs - estimates],
            fmt=f"{marker}-", color=COLORS[dist], markersize=7, linewidth=2.2,
            elinewidth=1.5, capsize=3, label=dist, zorder=3,
        )
    ax_persistence.set_xticks(x_windows, [
        f"Early\n{distance_label(min(EARLY_DISTANCES))}–{distance_label(max(EARLY_DISTANCES))}",
        f"Late\n{distance_label(min(LATE_DISTANCES))}–{distance_label(max(LATE_DISTANCES))}",
    ])
    
    ax_persistence.set_xlim(-0.28, 1.28)
    ax_persistence.set_ylabel("Model-estimated residualized log RT")
    ax_persistence.set_title(f"Relative persistence: distance-window interaction {p_label(persistence_p)}")
    ax_persistence.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax_persistence.legend(frameon=False, loc="best")
    panel_label(ax_persistence, "F", x=-0.05, y=1.08)

    figure_path = os.path.join(OUT, "main_distance_results.svg")
    fig.savefig(figure_path, format="svg", bbox_inches="tight", facecolor="white")
    print(f"Saved main-text figure: {figure_path}")
    plt.show()



