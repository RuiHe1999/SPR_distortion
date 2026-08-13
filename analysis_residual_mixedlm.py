# 1. Packages
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from tqdm import tqdm
from difflib import SequenceMatcher
from wordfreq import word_frequency, tokenize
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# 2. Settings
FILE = "data/participant_word_RT.xlsx"
OUT = "results/residual_analysis"
os.makedirs(OUT, exist_ok=True)

SUBJ, STORY, VERSION, WORD = "Participant", "Story", "Story_Version", "Word"
INDEX, SENT, LABEL, RT = "WordIndex", "SentenceIndex", "Word_Condition", "RT"
VERSION_ORDER = ["Original", "CD", "RD"]
MAX_K_OFFSET = 3
POSITION_ORDER = [
    f"K{offset:+d}" if offset else "K"
    for offset in range(-MAX_K_OFFSET, MAX_K_OFFSET + 1)
] + ["K_END"]
OFFSET_MAP = dict(zip(POSITION_ORDER[:-1], range(-MAX_K_OFFSET, MAX_K_OFFSET + 1)))
OFFSET_MAP["K_END"] = MAX_K_OFFSET + 1
COLORS = {"Original": "#9A8678", "CD": "#72BAA9", "RD": "#AD5C71"}
EFFECT_COLORS = {"CD - Original": COLORS["CD"], "RD - Original": COLORS["RD"]}
OUTCOME = "resid_logRT"
OUTCOME_LABEL = "Residualized log RT"

def normalize_word(x):
    x = str(x).lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"^[^\w']+|[^\w']+$", "", x)

def lexical_controls(text, lang="en", eps=1e-9):
    toks = tokenize(str(text).lower(), lang)
    if not toks:
        return np.nan, 0
    freqs = np.array([word_frequency(tok, lang) for tok in toks], dtype=float)
    freqs = np.where(freqs > 0, freqs, eps)
    return -np.mean(np.log2(freqs)), sum(len(tok) for tok in toks)

# 3. Data and outcome
df = pd.read_excel(FILE)
required = [SUBJ, STORY, VERSION, WORD, INDEX, SENT, LABEL, RT]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df[SUBJ] = df[SUBJ].astype(str)
df[STORY] = df[STORY].astype(str).str.strip()
df[VERSION] = df[VERSION].astype(str).str.strip()
df[WORD] = df[WORD].astype(str).str.strip()
df[INDEX] = pd.to_numeric(df[INDEX], errors="coerce")
df[SENT] = pd.to_numeric(df[SENT], errors="coerce")
df[RT] = pd.to_numeric(df[RT], errors="coerce")
df = df.dropna(subset=required).copy()
df = df[df[RT] > 0].drop_duplicates([SUBJ, STORY, VERSION, INDEX], keep="first")
df[INDEX] = df[INDEX].astype(int)
df[SENT] = df[SENT].astype(int)
df["StoryNumber"] = pd.to_numeric(df[STORY].str.extract(r"(\d+)")[0], errors="coerce")
df["VersionOrder"] = df[VERSION].map({"Original": 0, "CD": 1, "RD": 2})

lexicon = pd.DataFrame({WORD: df[WORD].drop_duplicates().to_numpy()})
lexicon[["FreqH", "Length"]] = pd.DataFrame(lexicon[WORD].map(lexical_controls).tolist(), index=lexicon.index)
lexicon.to_csv(os.path.join(OUT, "lexical_controls.csv"), index=False)
df = df.merge(lexicon, on=WORD, how="left", validate="many_to_one")
print("Lexical controls calculated:", len(lexicon), "unique words | Missing FreqH:", df["FreqH"].isna().sum())

df = df.sort_values([SUBJ, "StoryNumber", "VersionOrder", INDEX]).reset_index(drop=True)
df["log_RT"] = np.log(df[RT])
grp = [SUBJ, STORY, VERSION]
df["prev_index"] = df.groupby(grp)[INDEX].shift(1)
df["lag1_logRT"] = df.groupby(grp)["log_RT"].shift(1)

# Do not calculate lag across missing words
contiguous = (df[INDEX] - df["prev_index"]).eq(1)
gap_lag = df["prev_index"].notna() & ~contiguous
df.loc[~contiguous, "lag1_logRT"] = np.nan
df.loc[gap_lag, [SUBJ, STORY, VERSION, INDEX, "prev_index"]].to_csv(os.path.join(OUT, "gap_lag_qc.csv"), index=False)
print("Lag removed across gaps:", gap_lag.sum())

df["item_uid"] = df[STORY] + "_" + df[VERSION] + "_w" + df[INDEX].astype(str)
resid_df = df.dropna(subset=["log_RT", "lag1_logRT", SUBJ, "item_uid"]).copy()

# Residualize autocorrelation
resid_formula = "log_RT ~ lag1_logRT"
resid_model = smf.mixedlm(resid_formula, data=resid_df, groups=resid_df[SUBJ], re_formula="1")
resid_fit = resid_model.fit(reml=False, method=None, maxiter=300, disp=False)
df[OUTCOME] = np.nan
df.loc[resid_df.index, OUTCOME] = resid_fit.resid

with open(os.path.join(OUT, "residual_model.txt"), "w", encoding="utf-8") as f:
    f.write(resid_formula + "\n\n" + resid_fit.summary().as_text())

print("Rows:", len(df), "| Participants:", df[SUBJ].nunique(), "| Non-missing outcome:", df[OUTCOME].notna().sum())

# 4. Complete stimulus template and global trajectory
template = df.groupby([STORY, VERSION, INDEX], as_index=False).agg(Word=(WORD, "first"), SentenceIndex=(SENT, "first"), Word_Condition=(LABEL, "first"))
template["StoryNumber"] = pd.to_numeric(template[STORY].str.extract(r"(\d+)")[0], errors="coerce")
template["VersionOrder"] = template[VERSION].map({"Original": 0, "CD": 1, "RD": 2})
template = template.sort_values(["StoryNumber", "VersionOrder", INDEX]).reset_index(drop=True)
template["PositionInSentence"] = template.groupby([STORY, VERSION, "SentenceIndex"]).cumcount()
template["SentenceLength"] = template.groupby([STORY, VERSION, "SentenceIndex"])["Word"].transform("size")
template["RelativePosition"] = np.where(template["SentenceLength"] > 1, template["PositionInSentence"] / (template["SentenceLength"] - 1), 1.0)

df = df.merge(template[[STORY, VERSION, INDEX, "PositionInSentence", "SentenceLength", "RelativePosition"]], on=[STORY, VERSION, INDEX], how="left", validate="many_to_one")
edges = np.linspace(0, 1, 11)
centres = (edges[:-1] + edges[1:]) / 2
df["RelativeBin"] = pd.cut(df["RelativePosition"], bins=edges, labels=np.round(centres, 3), include_lowest=True).astype(float)

global_summary = df.dropna(subset=["RelativeBin", OUTCOME]).groupby([VERSION, "RelativeBin"], as_index=False).agg(Mean=(OUTCOME, "mean"), SD=(OUTCOME, "std"), N=(OUTCOME, "size"))
global_summary["SE"] = global_summary["SD"] / np.sqrt(global_summary["N"])
global_summary["CI95"] = 1.96 * global_summary["SE"]
global_summary.to_csv(os.path.join(OUT, "global_observation_summary.csv"), index=False)

plt.figure(figsize=(7.5, 5))
for version in VERSION_ORDER:
    tmp = global_summary[global_summary[VERSION] == version].sort_values("RelativeBin")
    if tmp.empty:
        continue
    x, y, ci = tmp["RelativeBin"].to_numpy(float), tmp["Mean"].to_numpy(float), tmp["CI95"].fillna(0).to_numpy(float)
    plt.plot(x, y, marker="o", linewidth=2.3, label=version, color=COLORS[version])
    plt.fill_between(x, y - ci, y + ci, alpha=0.18, color=COLORS[version])
plt.xlabel("Relative position within sentence")
plt.ylabel(OUTCOME_LABEL)
plt.legend(title="Story version", frameon=False)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "global_trajectory.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(OUT, "global_trajectory.png"), dpi=300, bbox_inches="tight")
plt.show()

# 5. Position mapping
template["NormWord"] = template["Word"].map(normalize_word)
template["Word_Condition"] = template["Word_Condition"].astype(str).str.replace("CD_", "", regex=False).str.replace("RD_", "", regex=False).str.replace("K+END", "K_END", regex=False)
template["Labels"] = template["Word_Condition"].map(lambda x: [v.strip() for v in re.split(r"[|,]", str(x)) if v.strip()])
position_rows = []
stories = sorted(template[STORY].unique(), key=lambda x: int(re.search(r"\d+", x).group()))

for dist in ["CD", "RD"]:
    for story in tqdm(stories, desc=f"Matching {dist}"):
        original_story = template[(template[STORY] == story) & (template[VERSION] == "Original")]
        distorted_story = template[(template[STORY] == story) & (template[VERSION] == dist)]
        if original_story.empty or distorted_story.empty:
            continue

        for sentence in sorted(set(original_story["SentenceIndex"]) & set(distorted_story["SentenceIndex"])):
            original_sentence = original_story[original_story["SentenceIndex"] == sentence].sort_values(INDEX).reset_index(drop=True)
            distorted_sentence = distorted_story[distorted_story["SentenceIndex"] == sentence].sort_values(INDEX).reset_index(drop=True)

            # Position mapping unless stimulus lengths differ
            if len(original_sentence) == len(distorted_sentence):
                local_map = {j: j for j in range(len(distorted_sentence))}
            else:
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

            for j, row in distorted_sentence.iterrows():
                for position in [x for x in row["Labels"] if x in POSITION_ORDER]:
                    original_j = len(original_sentence) - 1 if position == "K_END" else local_map.get(j)
                    if original_j is None or not 0 <= original_j < len(original_sentence):
                        continue
                    position_rows.append({"DistType": dist, STORY: story, "MapSentenceIndex": sentence, "Position": position, "DistWordIndex": int(row[INDEX]), "OriginalWordIndex": int(original_sentence.iloc[original_j][INDEX]), "DistWord": row["Word"], "OriginalWord": original_sentence.iloc[original_j]["Word"]})

position_map_raw = pd.DataFrame(position_rows).drop_duplicates()
if position_map_raw.empty:
    raise ValueError("No K positions were mapped. Check Word_Condition labels.")

# Avoid reusing one Original word for multiple distorted words at the same position
dup_keys = ["DistType", STORY, "MapSentenceIndex", "Position", "OriginalWordIndex"]
mapping_duplicates = position_map_raw[position_map_raw.duplicated(dup_keys, keep=False)].copy()
mapping_duplicates.to_csv(os.path.join(OUT, "distortion_mapping_duplicates_qc.csv"), index=False)
print("Duplicate Original mappings collapsed:", len(mapping_duplicates))

position_map = position_map_raw.sort_values(["DistType", STORY, "MapSentenceIndex", "Position", "DistWordIndex"]).drop_duplicates(dup_keys, keep="last").copy()
position_map["Offset"] = position_map["Position"].map(OFFSET_MAP)
position_map["Position"] = pd.Categorical(position_map["Position"], categories=POSITION_ORDER, ordered=True)
position_map = position_map.sort_values(["DistType", STORY, "MapSentenceIndex", "Offset", "DistWordIndex"]).reset_index(drop=True)
position_map.to_csv(os.path.join(OUT, "distortion_position_map.csv"), index=False)
print(position_map.groupby(["DistType", "Position"], observed=True).size())

# 6. Matched data
matched_tables = []

for dist in ["CD", "RD"]:
    current_map = position_map[position_map["DistType"] == dist].copy()
    current_map["ItemID"] = dist + "_" + current_map[STORY] + "_s" + current_map["MapSentenceIndex"].astype(str) + "_d" + current_map["DistWordIndex"].astype(str) + "_o" + current_map["OriginalWordIndex"].astype(str) + "_" + current_map["Position"].astype(str)

    distorted = df[df[VERSION] == dist].merge(current_map, left_on=[STORY, INDEX], right_on=[STORY, "DistWordIndex"], how="inner")
    original = df[df[VERSION] == "Original"].merge(current_map, left_on=[STORY, INDEX], right_on=[STORY, "OriginalWordIndex"], how="inner")
    distorted["Condition"], original["Condition"] = dist, "Original"
    matched_tables.extend([distorted, original])

local_df = pd.concat(matched_tables, ignore_index=True)
local_df["Position"] = pd.Categorical(local_df["Position"], categories=POSITION_ORDER, ordered=True)
local_df["Offset"] = local_df["Position"].map(OFFSET_MAP)
local_df["ItemID"] = local_df["ItemID"].astype(str)

item_check = local_df.groupby(["DistType", "ItemID"])["Condition"].nunique().reset_index(name="NConditions")
local_df = local_df[local_df["ItemID"].isin(item_check.loc[item_check["NConditions"] == 2, "ItemID"])].copy()
local_df.to_csv(os.path.join(OUT, "matched_local_data.csv"), index=False)

count_table = local_df.groupby(["DistType", "Condition", "Position"], observed=True, as_index=False).agg(NObservations=(OUTCOME, "count"), NParticipants=(SUBJ, "nunique"), NItems=("ItemID", "nunique"))
count_table.to_csv(os.path.join(OUT, "matched_position_counts.csv"), index=False)
print(count_table.to_string(index=False))

# 7. Observed profiles
local_summary = local_df.dropna(subset=[OUTCOME]).groupby(["DistType", "Condition", "Position", "Offset"], observed=True, as_index=False).agg(Mean=(OUTCOME, "mean"), SD=(OUTCOME, "std"), N=(OUTCOME, "size"))
local_summary["SE"] = local_summary["SD"] / np.sqrt(local_summary["N"])
local_summary["CI95"] = 1.96 * local_summary["SE"]
local_summary.to_csv(os.path.join(OUT, "local_observation_summary.csv"), index=False)

curve_specs = [("CD", "Original", "Original (CD-matched)", "#CAAA98", "--"), ("CD", "CD", "CD", COLORS["CD"], "-"), ("RD", "Original", "Original (RD-matched)", "#9A8678", "--"), ("RD", "RD", "RD", COLORS["RD"], "-")]

plt.figure(figsize=(8, 6))
for dist, condition, label, color, linestyle in curve_specs:
    tmp = local_summary[(local_summary["DistType"] == dist) & (local_summary["Condition"] == condition)].sort_values("Offset")
    if tmp.empty:
        continue
    plt.plot(tmp["Offset"], tmp["Mean"], marker="o", linewidth=2.3, linestyle=linestyle, label=label, color=color)
plt.xticks(list(OFFSET_MAP.values()), POSITION_ORDER)
plt.xlabel("Analysis position")
plt.ylabel(OUTCOME_LABEL)
plt.legend(frameon=False)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "observed_profiles.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(OUT, "observed_profiles.png"), dpi=300, bbox_inches="tight")
plt.show()

# 8. MixedLM
result_rows = []

for dist in ["CD", "RD"]:
    for position in POSITION_ORDER:
        dat = local_df[(local_df["DistType"] == dist) & (local_df["Position"] == position) & local_df["Condition"].isin(["Original", dist])].dropna(subset=[OUTCOME, "FreqH", "Length", INDEX, SUBJ, "ItemID", "Condition"]).copy()
        n_obs, n_subj, n_items = len(dat), dat[SUBJ].nunique(), dat["ItemID"].nunique()
        print(f"Fitting {dist}, {position}: N={n_obs}, participants={n_subj}, items={n_items}")

        base = {"DistType": dist, "Position": position, "Offset": OFFSET_MAP[position], "Contrast": f"{dist} - Original", "N": n_obs, "NParticipants": n_subj, "NItems": n_items}
        if dat["Condition"].nunique() < 2 or n_obs < 20 or n_subj < 2 or n_items < 2:
            result_rows.append({**base, "Beta": np.nan, "SE": np.nan, "Z": np.nan, "P": np.nan, "CILow": np.nan, "CIHigh": np.nan, "Converged": False, "Error": "Insufficient data"})
            continue

        dat["Condition"] = pd.Categorical(dat["Condition"], categories=["Original", dist])
        
        formula = f"{OUTCOME} ~ WordIndex + C(Condition, Treatment(reference='Original')) + FreqH + Length"
        model = smf.mixedlm(formula, data=dat, groups=dat[SUBJ], re_formula="1", vc_formula={"Item": "0 + C(ItemID)"})
        fit = model.fit()
        term = next(x for x in fit.params.index if f"[T.{dist}]" in x)
        ci_low, ci_high = fit.conf_int().loc[term]
        result_rows.append({**base, "Beta": fit.params[term], "SE": fit.bse[term], "Z": fit.tvalues[term], "P": fit.pvalues[term], "CILow": ci_low, "CIHigh": ci_high, "Converged": bool(fit.converged), "Error": ""})
    
results = pd.DataFrame(result_rows)
results["P_FDR_All"], results["Significant_FDR_All"] = np.nan, False
valid = results["P"].notna()

if valid.any():
    reject, corrected, _, _ = multipletests(results.loc[valid, "P"], method="fdr_bh")
    results.loc[valid, "P_FDR_All"], results.loc[valid, "Significant_FDR_All"] = corrected, reject

results["P_FDR_Within"], results["Significant_FDR_Within"] = np.nan, False

for dist in ["CD", "RD"]:
    mask = (results["DistType"] == dist) & results["P"].notna()
    if mask.any():
        reject, corrected, _, _ = multipletests(results.loc[mask, "P"], method="fdr_bh")
        results.loc[mask, "P_FDR_Within"], results.loc[mask, "Significant_FDR_Within"] = corrected, reject

results["PercentChange"] = (np.exp(results["Beta"]) - 1) * 100
results["PercentCILow"] = (np.exp(results["CILow"]) - 1) * 100
results["PercentCIHigh"] = (np.exp(results["CIHigh"]) - 1) * 100
results = results.sort_values(["DistType", "Offset"]).reset_index(drop=True)
results.to_csv(os.path.join(OUT, "mixedlm_results.csv"), index=False)

print(results[["DistType", "Position", "Beta", "PercentChange", "P", "P_FDR_Within", "P_FDR_All", "Converged"]].round(4).to_string(index=False))

# 9. Model effects
plt.figure(figsize=(8, 6))
dodge = {"CD - Original": -0.07, "RD - Original": 0.07}

for contrast in ["CD - Original", "RD - Original"]:
    tmp = results[results["Contrast"] == contrast].dropna(subset=["Beta", "CILow", "CIHigh"]).sort_values("Offset")
    if tmp.empty:
        continue
    x = tmp["Offset"].to_numpy(float) + dodge[contrast]
    y = tmp["Beta"].to_numpy(float)
    yerr = np.vstack([y - tmp["CILow"].to_numpy(float), tmp["CIHigh"].to_numpy(float) - y])
    plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.3, capsize=4, label=contrast, color=EFFECT_COLORS[contrast])

    for _, row in tmp.iterrows():
        if row["Significant_FDR_All"]:
            plt.text(row["Offset"] + dodge[contrast], row["CIHigh"] + 0.002, "*", ha="center", va="bottom", fontsize=15, color=EFFECT_COLORS[contrast])

plt.xticks(list(OFFSET_MAP.values()), POSITION_ORDER)
plt.xlabel("Analysis position")
plt.ylabel(f"Estimated difference in {OUTCOME_LABEL.lower()}")
plt.legend(frameon=False)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "mixedlm_effects.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(OUT, "mixedlm_effects.png"), dpi=300, bbox_inches="tight")
plt.show()

df.drop(columns=["StoryNumber", "VersionOrder", "prev_index"], errors="ignore").to_csv(os.path.join(OUT, "participant_word_data_with_outcome.csv"), index=False)
print("Saved results to:", OUT)
