# 1. packages
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt

# 2. constants
FILE = "data/story_word_template.xlsx"
OUT = "results/llama32"
os.makedirs(OUT, exist_ok=True)

SUBJ = "Participant"
STORY = "Story"
VERSION = "Story_Version"
WORD = "Word"
INDEX = "WordIndex"
SENT = "SentenceIndex"
LABEL = "Word_Condition"

DIST_TYPES = ["CD", "RD"]
MIN_DISTANCE = 0
MAX_DISTANCE = 10

MODEL_NAME = "meta-llama/Llama-3.2-3B"
MODEL_LABEL = "Llama-3.2-3B"
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

POSITION_ORDER = ["K-3", "K-2", "K-1", "K", "K+1", "K+2", "K+3", "K_END"]

EFFECT_COLORS = {"CD": "#69B3A2", "RD": "#B05A6E"}
POSITION_X = {position: index for index, position in enumerate(POSITION_ORDER)}
DODGE = {"CD": -0.06, "RD": 0.06}

# load Llama 3.2 1B model 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
elif DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

print("Model:", MODEL_LABEL)
print("Device:", DEVICE)
print("Data type:", DTYPE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE,).to(DEVICE).eval()

# 3. functions
def clean_model_word(word):
    return re.sub(r'[.,!?;:"“”()\[\]{}…]', "", str(word).lower().replace("’", "'").replace("‘", "'")).strip()

def extract_story_outputs(story_data):
    
    story_data = story_data.sort_values(INDEX).reset_index(drop=True)

    # clean words 
    story_data["ModelWord"] = story_data[WORD].apply(clean_model_word)

    if story_data["ModelWord"].eq("").any(): 
        raise ValueError("Some words became empty after punctuation removal")

    # process the text 
    encoded = tokenizer(story_data[WORD].tolist(), is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
    word_ids = encoded.word_ids(batch_index=0)
    inputs = {name: value.to(DEVICE) for name, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # emebdding and surprisal 
    input_ids = inputs["input_ids"][0]
    log_probs = F.log_softmax(outputs.logits[0, :-1].float(), dim=-1)
    token_surprisal = torch.full((len(input_ids),), torch.nan, device=DEVICE)
    token_surprisal[1:] = -log_probs.gather(1, input_ids[1:].unsqueeze(1)).squeeze(1)
    final_hidden = outputs.hidden_states[-1][0].float()

    # alignment
    result = {}
    for local_index, word_index in enumerate(story_data[INDEX]):
        positions = [position for position, word_id in enumerate(word_ids) if word_id == local_index]
        if not positions: continue
        surprisal = token_surprisal[positions]
        result[int(word_index)] = {
            "Surprisal": float(surprisal.sum().cpu()) if torch.isfinite(surprisal).all() else np.nan,
            "Hidden": final_hidden[positions[-1]].cpu().numpy(),
        }
        
    return result

def make_local_map(original_sentence, distorted_sentence):
    original_words = original_sentence[WORD].apply(clean_model_word).tolist()
    distorted_words = distorted_sentence[WORD].apply(clean_model_word).tolist()
    if len(original_words) == len(distorted_words):
        return {j: j for j in range(len(distorted_words))}
    matcher = SequenceMatcher(None, original_words, distorted_words, autojunk=False)
    local_map = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1): local_map[j1 + offset] = i1 + offset
        elif tag == "replace" and i2 > i1 and j2 > j1:
            for j in range(j1, j2):
                original_j = i1 if j2 - j1 == 1 else i1 + round((j - j1) * (i2 - i1 - 1) / (j2 - j1 - 1))
                local_map[j] = min(max(original_j, i1), i2 - 1)
        elif tag == "insert" and j2 > j1:
            anchor = i1 - 1 if i1 > 0 else (i1 if i1 < len(original_words) else None)
            if anchor is not None:
                for j in range(j1, j2): local_map[j] = anchor
    return local_map


#  4. commands
# read stimulus template
template = pd.read_excel(FILE)
required = [STORY, VERSION, INDEX, WORD, SENT, LABEL]
missing = [column for column in required if column not in template.columns]
if missing: 
    raise ValueError(f"Missing columns: {missing}")

template = template[required].dropna().copy()
template[STORY] = template[STORY].astype(str).str.strip()
template[VERSION] = template[VERSION].astype(str).str.strip()
template[WORD] = template[WORD].astype(str).str.strip()
template[INDEX] = pd.to_numeric(template[INDEX], errors="raise").astype(int)
template[SENT] = pd.to_numeric(template[SENT], errors="raise").astype(int)
template = template.sort_values([STORY, VERSION, INDEX]).reset_index(drop=True)

# load perturbation trajectories
TRAJECTORY_FILE = "results/residual_analysis/decay/perturbation_trajectory_map.csv"
trajectory_map = pd.read_csv(TRAJECTORY_FILE)
required_map = ["DistType", STORY, "PerturbationID", "Distance", "AtBoundary", "DistWordIndex", "OriginalWordIndex"]
missing = [column for column in required_map if column not in trajectory_map.columns]
if missing:
    raise ValueError(f"Missing trajectory columns: {missing}")

trajectory_map = trajectory_map[trajectory_map["DistType"].isin(DIST_TYPES) & trajectory_map["Distance"].between(MIN_DISTANCE, MAX_DISTANCE)].copy()
trajectory_map["Distance"] = trajectory_map["Distance"].astype(int)
trajectory_map["AtBoundary"] = trajectory_map["AtBoundary"].astype(int)

# extract LLM outcome 
story_outputs = {}
story_groups = template.groupby([STORY, VERSION], sort=False)
for (story, version), story_data in tqdm(story_groups, total=story_groups.ngroups, desc="Llama inference"):
    story_outputs[(story, version)] = extract_story_outputs(story_data)

# pair Original and distorted outputs
effect_rows = []
for _, row in trajectory_map.iterrows():
    story, dist = row[STORY], row["DistType"]
    distorted = story_outputs.get((story, dist), {}).get(int(row["DistWordIndex"]))
    original = story_outputs.get((story, "Original"), {}).get(int(row["OriginalWordIndex"]))
    if distorted is None or original is None: 
        continue
    
    dist_hidden, orig_hidden = distorted["Hidden"], original["Hidden"]
    denominator = np.linalg.norm(dist_hidden) * np.linalg.norm(orig_hidden)
    cosine = 1 - np.dot(dist_hidden, orig_hidden) / denominator if denominator > 0 else np.nan
    effect_rows.append({
        "DistType": dist, STORY: story, 
        "PerturbationID": row["PerturbationID"],
        "Distance": int(row["Distance"]), 
        "AtBoundary": int(row["AtBoundary"]),
        "DistWordIndex": int(row["DistWordIndex"]), 
        "OriginalWordIndex": int(row["OriginalWordIndex"]),
        "DistSurprisal": distorted["Surprisal"], 
        "OriginalSurprisal": original["Surprisal"],
        "DeltaSurprisal": distorted["Surprisal"] - original["Surprisal"],
        "RepresentationDistance": float(cosine),
    })

llm_effects = pd.DataFrame(effect_rows)
llm_effects.to_csv(os.path.join(OUT, "llama_word_effects.csv"), index=False)

# position match 
original_sentences = {
    (story, int(sentence)): data.sort_values(INDEX).reset_index(drop=True)
    for (story, sentence), data in template[template[VERSION] == "Original"].groupby([STORY, SENT])
}

position_rows = []
distorted_template = template[template[VERSION].isin(DIST_TYPES)]

for (story, dist_type, sentence), distorted_sentence in distorted_template.groupby([STORY, VERSION, SENT], sort=False):
    distorted_sentence = distorted_sentence.sort_values(INDEX).reset_index(drop=True)
    original_sentence = original_sentences.get((story, int(sentence)))
    if original_sentence is None: 
        continue
    
    local_map = make_local_map(original_sentence, distorted_sentence)
    for distorted_local_index, distorted_row in distorted_sentence.iterrows():
        labels = [label.strip() for label in str(distorted_row[LABEL]).split("|")]
        positions = [label for label in labels if label in POSITION_ORDER]
        
        if not positions or distorted_local_index not in local_map: 
            continue
        
        original_row = original_sentence.iloc[local_map[distorted_local_index]]
        distorted_output = story_outputs.get((story, dist_type), {}).get(int(distorted_row[INDEX]))
        original_output = story_outputs.get((story, "Original"), {}).get(int(original_row[INDEX]))
        
        if distorted_output is None or original_output is None: 
            continue
        
        distorted_hidden = distorted_output["Hidden"]
        original_hidden = original_output["Hidden"]
        denominator = np.linalg.norm(distorted_hidden) * np.linalg.norm(original_hidden)
        cosine = 1 - np.dot(distorted_hidden, original_hidden) / denominator if denominator > 0 else np.nan
        for position in positions:
            position_rows.append({
                "DistType": dist_type, STORY: story, SENT: int(sentence),
                "PositionUnitID": f"{dist_type}_{story}_s{int(sentence)}",
                "Position": position,
                "DistWordIndex": int(distorted_row[INDEX]),
                "OriginalWordIndex": int(original_row[INDEX]),
                "DistWord": distorted_row[WORD],
                "OriginalWord": original_row[WORD],
                "DistSurprisal": distorted_output["Surprisal"],
                "OriginalSurprisal": original_output["Surprisal"],
                "DeltaSurprisal": distorted_output["Surprisal"] - original_output["Surprisal"],
                "RepresentationDistance": float(cosine),
            })

position_effects = pd.DataFrame(position_rows)
position_effects["Position"] = pd.Categorical(position_effects["Position"], categories=POSITION_ORDER, ordered=True)
position_effects.to_csv(os.path.join(OUT, "llama_position_effects.csv"), index=False)

position_counts = position_effects.groupby(["DistType", "Position"], observed=False).agg(
    N=("DeltaSurprisal", "count"),
    NUnits=("PositionUnitID", "nunique"),
    NStories=(STORY, "nunique"),
).reset_index()

print(position_counts.to_string(index=False))

# Regression modelat every position for CD and RD
effect_rows = []
position_model_fits = {}

for outcome in ["DeltaSurprisal", "RepresentationDistance"]:
    for dist_type in DIST_TYPES:
        for position in POSITION_ORDER:
            dat = position_effects[
                (position_effects["DistType"] == dist_type) &
                (position_effects["Position"].astype(str) == position)
            ].dropna(subset=[outcome, "PositionUnitID"]).copy()

            if len(dat) < 3 or dat["PositionUnitID"].nunique() < 2:
                continue

            fit = smf.ols(f"{outcome} ~ 1", data=dat).fit(
                cov_type="cluster",
                cov_kwds={"groups": dat["PositionUnitID"], "use_correction": True,},
                use_t=True,
            )

            estimate = float(fit.params["Intercept"])
            standard_error = float(fit.bse["Intercept"])
            t_value = float(fit.tvalues["Intercept"])
            p_value = float(fit.pvalues["Intercept"])
            ci_low, ci_high = fit.conf_int().loc["Intercept"]

            position_model_fits[(outcome, dist_type, position)] = fit
            effect_rows.append({
                "Outcome": outcome,
                "DistType": dist_type,
                "Position": position,
                "Contrast": f"{dist_type} - Original" if outcome == "DeltaSurprisal" else f"{dist_type} vs Original",
                "N": len(dat),
                "NUnits": dat["PositionUnitID"].nunique(),
                "Estimate": estimate,
                "SE": standard_error,
                "T": t_value,
                "P": p_value,
                "CILow": float(ci_low),
                "CIHigh": float(ci_high),
            })

position_results = pd.DataFrame(effect_rows)
position_results["Position"] = pd.Categorical(position_results["Position"], ordered=True,)
position_results["P_FDR_Within"] = np.nan
position_results["Significant_FDR_Within"] = False

for outcome in ["DeltaSurprisal", "RepresentationDistance"]:
    for dist_type in DIST_TYPES:
        mask = (position_results["Outcome"] == outcome) & (position_results["DistType"] == dist_type)
        reject, corrected_p, _, _ = multipletests(position_results.loc[mask, "P"], method="fdr_bh",)
        position_results.loc[mask, "P_FDR_Within"] = corrected_p
        position_results.loc[mask, "Significant_FDR_Within"] = reject

position_results = position_results.sort_values(["Outcome", "DistType", "Position"])
position_results.to_csv(os.path.join(OUT, "llama_position_regression_results.csv"), index=False,)

print(position_results.round(4))

# visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
panels = [
    ("DeltaSurprisal", "Contextual surprisal", 
     "Estimated difference in contextual surprisal", "A"),
    ("RepresentationDistance", "Output-layer representation distance", 
     "Cosine distance", "B"),
]

for ax, (outcome, title, ylabel, panel) in zip(axes, panels):
    plot_data = position_results[position_results["Outcome"] == outcome].copy()
    plot_data["PositionText"] = plot_data["Position"].astype(str)
    y_range = plot_data["CIHigh"].max() - plot_data["CILow"].min()
    star_offset = max(y_range * 0.025, 0.0002)
    for dist_type in DIST_TYPES:
        dat = plot_data[plot_data["DistType"] == dist_type].copy()
        dat["PositionOrder"] = dat["PositionText"].map(POSITION_X)
        dat = dat.sort_values("PositionOrder")
        x = dat["PositionOrder"].to_numpy(float) + DODGE[dist_type]
        y = dat["Estimate"].to_numpy(float)
        yerr = np.vstack([y - dat["CILow"].to_numpy(float), dat["CIHigh"].to_numpy(float) - y])
        legend_label = f"{dist_type} − Original" if outcome == "DeltaSurprisal" else f"{dist_type} vs Original"
        ax.errorbar(x, y, yerr=yerr, color=EFFECT_COLORS[dist_type], marker="o", markersize=6.5, linewidth=2.2, capsize=3.5, label=legend_label)
        for x_value, ci_high, significant in zip(x, dat["CIHigh"], dat["Significant_FDR_Within"].fillna(False)):
            if significant: ax.text(x_value, ci_high + star_offset, "*", ha="center", fontsize=15, color=EFFECT_COLORS[dist_type])
    lower, upper = min(plot_data["CILow"].min(), 0), max(plot_data["CIHigh"].max(), 0)
    padding = max((upper - lower) * 0.08, star_offset * 2)
    ax.set_ylim(lower - padding, upper + padding * 1.8)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.1)
    ax.set_xticks(range(len(POSITION_ORDER)), POSITION_ORDER)
    ax.set_xlabel("Analysis position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.10, 1.06, panel, transform=ax.transAxes, fontsize=15, fontweight="bold")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3)
fig.savefig(os.path.join(OUT, "llama_position_effects_combined.svg"), format="svg", bbox_inches="tight")
plt.show()




