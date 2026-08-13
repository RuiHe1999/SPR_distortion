# 1. Packages
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

# 2. Settings
DATA_FILE = "data/data_exp_221200-v10_task-eyrj_deidentified.csv"
STIMULUS_FILE = "data/StoryStimulus.csv"
ID_COL = "Participant Private ID"
SESSION_COL = "Schedule ID"
CONDITION_ORDER = ["Original", "CD", "RD"]
CONDITION_MAP = {"original": "Original", "conceptual": "CD", "cd": "CD", "referential": "RD", "rd": "RD"}
PRE_K_WINDOW = 3
POST_K_WINDOW = 3
K_LABELS = ([f"K-{offset}" for offset in range(PRE_K_WINDOW, 0, -1)]
            + ["K"]
            + [f"K+{offset}" for offset in range(1, POST_K_WINDOW + 1)]
            + ["K_END"])
LABEL_ORDER = {label: order for order, label in enumerate(K_LABELS)}

def get_body(text):
    """Remove story title."""
    return re.split(r"\r?\n\s*\r?\n", str(text).strip(), maxsplit=1)[-1].strip()

def normalize_word(word):
    """Normalize words for QC."""
    word = str(word).lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"^[^\w]+|[^\w]+$", "", word)

# 3. Create stimulus word template and K labels
stimulus = pd.read_csv(STIMULUS_FILE)
stimulus["Disruption type"] = stimulus["Disruption type"].astype(str).str.strip().str.lower()
stimulus_rows = []
stimulus_qc = []
k_distance_rows = []

for story_id in sorted(stimulus["Story ID"].dropna().unique()):
    story_id = int(story_id)
    story_data = stimulus[stimulus["Story ID"] == story_id]
    original_text = story_data.loc[story_data["Disruption type"] == "original", "Stories"].iloc[0]
    original_sentences = re.split(r"(?<=[.!?])\s+", get_body(original_text))

    for raw_version, version in {"original": "Original", "conceptual": "CD", "referential": "RD"}.items():
        text = story_data.loc[story_data["Disruption type"] == raw_version, "Stories"].iloc[0]
        sentences = re.split(r"(?<=[.!?])\s+", get_body(text))
        assert len(sentences) == len(original_sentences), f"Sentence mismatch: Story{story_id}, {version}"
        word_index = 0

        for sentence_index, (original_sentence, sentence) in enumerate(zip(original_sentences, sentences), start=1):
            original_tokens = original_sentence.split()
            tokens = sentence.split()
            labels = [set() for _ in tokens]
            spans = []

            if version != "Original":
                original_norm = [normalize_word(x) for x in original_tokens]
                token_norm = [normalize_word(x) for x in tokens]

                # Prefer positional comparison when word counts are identical
                if len(original_tokens) == len(tokens):
                    changed = [i for i, (a, b) in enumerate(zip(original_norm, token_norm)) if a != b]
                    if changed:
                        start = previous = changed[0]
                        for position in changed[1:]:
                            if position == previous + 1:
                                previous = position
                            else:
                                spans.append((start, previous))
                                start = previous = position
                        spans.append((start, previous))
                    method = "position"
                else:
                    # Fallback only when stimulus versions differ in length
                    matcher = SequenceMatcher(None, original_norm, token_norm, autojunk=False)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == "equal" or len(tokens) == 0:
                            continue
                        if j1 < j2:
                            spans.append((j1, j2 - 1))
                        else:
                            position = min(j1, len(tokens) - 1)
                            spans.append((position, position))
                    method = "sequence_fallback"

                for k_start, k_end in spans:
                    for position in range(k_start, k_end + 1):
                        labels[position].add("K")
                    for offset in range(PRE_K_WINDOW, 0, -1):
                        position = k_start - offset
                        if position >= 0:
                            labels[position].add(f"K-{offset}")
                    for offset in range(1, POST_K_WINDOW + 1):
                        position = k_end + offset
                        if position < len(tokens):
                            labels[position].add(f"K+{offset}")

                    k_distance_rows.append({
                        "Story": f"Story{story_id}",
                        "Story_Version": version,
                        "SentenceIndex": sentence_index,
                        "K_Start_SentencePos": k_start + 1,
                        "K_End_SentencePos": k_end + 1,
                        "Sentence_Length": len(tokens),
                        "Words_Before_K": k_start,
                        "Words_After_K": len(tokens) - (k_end + 1),
                    })

                if spans and len(tokens) > 0:
                    labels[-1].add("K_END")

                stimulus_qc.append({"Story": f"Story{story_id}", "Story_Version": version, "SentenceIndex": sentence_index, "Original_N": len(original_tokens), "Version_N": len(tokens), "Alignment_Method": method, "Changed_Spans": ";".join(f"{a + 1}-{b + 1}" for a, b in spans)})

            for token, token_labels in zip(tokens, labels):
                word_index += 1
                word_condition = version if len(token_labels) == 0 else "|".join(sorted(token_labels, key=lambda x: LABEL_ORDER[x]))
                stimulus_rows.append({"Story": f"Story{story_id}", "Story_Version": version, "WordIndex": word_index, "Word": token, "SentenceIndex": sentence_index, "Word_Condition": word_condition})

stimulus_words = pd.DataFrame(stimulus_rows)
stimulus_words["StoryNumber"] = pd.to_numeric(stimulus_words["Story"].str.extract(r"(\d+)")[0], errors="coerce")
stimulus_words["Story_Version"] = pd.Categorical(stimulus_words["Story_Version"], categories=CONDITION_ORDER, ordered=True)
stimulus_words = stimulus_words.sort_values(["StoryNumber", "Story_Version", "WordIndex"]).drop(columns="StoryNumber").reset_index(drop=True)
stimulus_words.to_excel("data/story_word_template.xlsx", index=False)

stimulus_qc = pd.DataFrame(stimulus_qc)
stimulus_qc.to_csv("data/stimulus_alignment_qc.csv", index=False, encoding="utf-8-sig")

# Words before K are counted before the first token of each K span; words after
# K are counted after its final token. The K span itself is excluded both ways.
k_sentence_end_distance = pd.DataFrame(k_distance_rows)
k_sentence_end_distance.to_csv("data/k_sentence_end_distance_detail.csv", index=False, encoding="utf-8-sig")
k_sentence_end_summary = k_sentence_end_distance.groupby("Story_Version", as_index=False).agg(
    N_Spans=("Words_After_K", "size"),
    Mean_Words_Before_K=("Words_Before_K", "mean"),
    Median_Words_Before_K=("Words_Before_K", "median"),
    SD_Words_Before_K=("Words_Before_K", "std"),
    Min_Words_Before_K=("Words_Before_K", "min"),
    Max_Words_Before_K=("Words_Before_K", "max"),
    Mean_Words_After_K=("Words_After_K", "mean"),
    Median_Words_After_K=("Words_After_K", "median"),
    SD_Words_After_K=("Words_After_K", "std"),
    Min_Words_After_K=("Words_After_K", "min"),
    Max_Words_After_K=("Words_After_K", "max"),
)
k_sentence_end_summary["Condition_Order"] = k_sentence_end_summary["Story_Version"].map({"CD": 0, "RD": 1})
k_sentence_end_summary = k_sentence_end_summary.sort_values("Condition_Order").drop(columns="Condition_Order").reset_index(drop=True)
distance_decimal_columns = [
    "Mean_Words_Before_K", "Median_Words_Before_K", "SD_Words_Before_K",
    "Mean_Words_After_K", "Median_Words_After_K", "SD_Words_After_K",
]
k_sentence_end_summary[distance_decimal_columns] = k_sentence_end_summary[distance_decimal_columns].round(2)
k_sentence_end_summary.to_csv("data/k_sentence_end_distance_summary.csv", index=False, encoding="utf-8-sig")

print("Stimulus words:", stimulus_words.shape)
print("\nK-label counts:")
print(stimulus_words["Word_Condition"].value_counts().to_string())

print("\nK position distances by condition (before first K / after final K):")
print(k_sentence_end_summary.to_string(index=False))

if len(stimulus_qc):
    fallback = stimulus_qc[stimulus_qc["Alignment_Method"] == "sequence_fallback"]
    print("\nStimulus sentences requiring SequenceMatcher fallback:", len(fallback))
    if len(fallback):
        print(fallback.to_string(index=False))

# 4. Read experimental data
df = pd.read_csv(DATA_FILE, low_memory=False)
df = df[df[ID_COL].notna()].copy()
df["Event Index"] = pd.to_numeric(df["Event Index"], errors="coerce")

print("\nRaw shape:", df.shape)
print("Participants:", df[ID_COL].nunique())
print("Sessions:", df[SESSION_COL].nunique())

# 5. Participant summary
participant_summary = df.groupby(ID_COL, as_index=False).agg(
    n_rows=("Event Index", "size"), 
    n_trials=("Trial Number", "nunique"), 
    final_score=("Store: Score", "max"))
participant_summary.to_csv("data/participant_summary.csv", index=False, encoding="utf-8-sig")
print((participant_summary["final_score"] / participant_summary["n_trials"]).describe())

# 6. Get condition for each story
story_columns = [f"Store: Story{i}" for i in range(1, 22)]
conditions = df.groupby([ID_COL, SESSION_COL], as_index=False)[story_columns].first().melt(id_vars=[ID_COL, SESSION_COL], var_name="Story", value_name="Story_Version").dropna(subset=["Story_Version"])
conditions["Story"] = conditions["Story"].str.replace("Store: ", "", regex=False)
conditions["Story_Version"] = conditions["Story_Version"].astype(str).str.strip()
condition_normalized = conditions["Story_Version"].str.lower().map(CONDITION_MAP)
conditions["Story_Version"] = condition_normalized.fillna(conditions["Story_Version"])

# 7. Extract presented words and following responses
events = df[df["Component Name"].isin(["Reading", "Keyboard Response"])].copy()
events = events.sort_values([ID_COL, SESSION_COL, "Trial Number", "Event Index"]).reset_index(drop=True)
event_groups = events.groupby([ID_COL, SESSION_COL, "Trial Number"], sort=False)
events["Story"] = event_groups["Spreadsheet: text"].transform(lambda x: x.ffill().bfill())
events["next_type"] = event_groups["Response Type"].shift(-1)
events["next_component"] = event_groups["Component Name"].shift(-1)
events["next_rt"] = pd.to_numeric(event_groups["Reaction Time"].shift(-1), errors="coerce")
events["Reaction Time"] = pd.to_numeric(events["Reaction Time"], errors="coerce")

word_mask = events["Response Type"].eq("info") & events["Component Name"].eq("Reading") & events["next_type"].eq("response") & events["next_component"].eq("Keyboard Response")
observed = events.loc[word_mask, [ID_COL, SESSION_COL, "Trial Number", "Event Index", "Story", "Response", "Reaction Time", "next_rt"]].copy()
observed = observed.rename(columns={ID_COL: "Participant", "Response": "Observed_Word"})
observed["Observed_Word"] = observed["Observed_Word"].astype(str).str.strip()
observed["RT"] = observed["next_rt"] - observed["Reaction Time"]
observed = observed.merge(conditions, left_on=["Participant", SESSION_COL, "Story"], right_on=[ID_COL, SESSION_COL, "Story"], how="left").drop(columns=ID_COL)

if observed["Story_Version"].isna().any():
    missing = observed.loc[observed["Story_Version"].isna(), ["Participant", "Story"]].drop_duplicates()
    raise ValueError(f"Missing conditions:\n{missing.head(20)}")

print("\nObserved word events:", len(observed))

# 8. Hybrid position/gap alignment
template_lookup = {}
for keys, template in stimulus_words.groupby(["Story", "Story_Version"], observed=True, sort=False):
    template_lookup[(str(keys[0]), str(keys[1]))] = template.sort_values("WordIndex").reset_index(drop=True)

aligned_trials = []
alignment_report = []
missing_rows = []
mismatch_rows = []
excluded_trials = []
group_columns = ["Participant", SESSION_COL, "Trial Number", "Story", "Story_Version"]
observed_groups = observed.groupby(group_columns, observed=True, sort=False)

for keys, trial in tqdm(observed_groups, total=observed_groups.ngroups):
    participant, session, trial_number, story, version = keys
    trial = trial.sort_values("Event Index").reset_index(drop=True)
    template = template_lookup.get((str(story), str(version)))

    if template is None:
        excluded_trials.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Observed": len(trial), "Template": 0, "Reason": "missing_template"})
        continue

    observed_tokens = [normalize_word(x) for x in trial["Observed_Word"]]
    template_tokens = [normalize_word(x) for x in template["Word"]]
    mapping = {}

    # Equal length: position must match exactly
    if len(trial) == len(template):
        if observed_tokens != template_tokens:
            for i, (a, b) in enumerate(zip(observed_tokens, template_tokens)):
                if a != b:
                    mismatch_rows.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Position": i + 1, "Observed_Word": trial.loc[i, "Observed_Word"], "Template_Word": template.loc[i, "Word"], "Word_Condition": template.loc[i, "Word_Condition"]})
            excluded_trials.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Observed": len(trial), "Template": len(template), "Reason": "equal_length_token_mismatch"})
            continue
        mapping = {i: i for i in range(len(trial))}
        status = "exact"

    # Shorter trials: allow only small exact gaps
    # Shorter trials: allow only small unique gaps
    elif len(trial) < len(template):
        n_missing = len(template) - len(trial)

        if n_missing > 3:
            excluded_trials.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Observed": len(trial), "Template": len(template), "Reason": "too_many_missing_words"})
            continue

        # Match from left
        left_map = {}
        j = 0
        for i, word in enumerate(observed_tokens):
            while j < len(template_tokens) and template_tokens[j] != word:
                j += 1
            if j == len(template_tokens):
                break
            left_map[i] = j
            j += 1

        # Match from right
        right_map = {}
        j = len(template_tokens) - 1
        for i in range(len(observed_tokens) - 1, -1, -1):
            while j >= 0 and template_tokens[j] != observed_tokens[i]:
                j -= 1
            if j < 0:
                break
            right_map[i] = j
            j -= 1

        if len(left_map) != len(trial) or len(right_map) != len(trial):
            excluded_trials.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Observed": len(trial), "Template": len(template), "Reason": "not_exact_subsequence"})
            continue

        if any(left_map[i] != right_map[i] for i in range(len(trial))):
            excluded_trials.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Observed": len(trial), "Template": len(template), "Reason": "ambiguous_gap_alignment"})
            continue

        mapping = left_map
        missing_template = sorted(set(range(len(template))) - set(mapping.values()))

        for i in missing_template:
            missing_rows.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "WordIndex": template.loc[i, "WordIndex"], "Word": template.loc[i, "Word"], "Word_Condition": template.loc[i, "Word_Condition"], "SentenceIndex": template.loc[i, "SentenceIndex"]})

        status = "gap_aligned"
        
    # Longer trials are likely duplicated/restarted
    else:
        excluded_trials.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Observed": len(trial), "Template": len(template), "Reason": "extra_or_duplicate_events"})
        continue

    obs_idx = sorted(mapping)
    template_idx = [mapping[i] for i in obs_idx]
    aligned = trial.iloc[obs_idx].copy().reset_index(drop=True)
    matched_template = template.iloc[template_idx].reset_index(drop=True)
    aligned["WordIndex"] = matched_template["WordIndex"].to_numpy()
    aligned["Word"] = matched_template["Word"].to_numpy()
    aligned["SentenceIndex"] = matched_template["SentenceIndex"].to_numpy()
    aligned["Word_Condition"] = matched_template["Word_Condition"].to_numpy()
    aligned["Token_Match"] = [normalize_word(a) == normalize_word(b) for a, b in zip(aligned["Observed_Word"], aligned["Word"])]
    aligned["Trial_Status"] = status
    aligned["Trial_Coverage"] = len(aligned) / len(template)

    if not aligned["Token_Match"].all():
        raise ValueError(f"Alignment error: {participant}, {story}, {version}, trial {trial_number}")

    missing_conditions = matched_template.iloc[0:0]
    missing_idx = sorted(set(range(len(template))) - set(template_idx))
    if missing_idx:
        missing_conditions = template.iloc[missing_idx]
    n_critical_missing = missing_conditions["Word_Condition"].astype(str).str.contains(r"(?:^|\|)K(?:$|\|)|K[-+_]", regex=True, na=False).sum() if len(missing_conditions) else 0

    aligned_trials.append(aligned)
    alignment_report.append({"Participant": participant, "Session": session, "Story": story, "Story_Version": version, "Trial": trial_number, "Observed": len(trial), "Template": len(template), "Aligned": len(aligned), "Coverage": len(aligned) / len(template), "Missing": len(template) - len(aligned), "Critical_Missing": n_critical_missing, "Status": status})

words = pd.concat(aligned_trials, ignore_index=True)
alignment_report = pd.DataFrame(alignment_report)
excluded_trials = pd.DataFrame(excluded_trials)
missing_words = pd.DataFrame(missing_rows)
alignment_mismatches = pd.DataFrame(mismatch_rows)

alignment_report.to_csv("data/alignment_report.csv", index=False, encoding="utf-8-sig")
excluded_trials.to_csv("data/alignment_excluded_trials.csv", index=False, encoding="utf-8-sig")
alignment_mismatches.to_csv("data/alignment_mismatches.csv", index=False, encoding="utf-8-sig")

print("\nAlignment status:")
print(alignment_report["Status"].value_counts())

print("\nExcluded trials:")
print(excluded_trials["Reason"].value_counts() if len(excluded_trials) else "None")

print("\nMissing template words:", len(missing_words))

if len(missing_words):
    critical_missing = missing_words[missing_words["Word_Condition"].astype(str).str.contains(r"(?:^|\|)K(?:$|\|)|K[-+_]", regex=True, na=False)].copy()
    print("Missing critical-region words:", len(critical_missing))
    print("\nMissing words per trial:")
    print(missing_words.groupby(["Participant", "Story", "Story_Version", "Trial"]).size().value_counts().sort_index().to_string())
    critical_missing.to_csv("data/critical_missing_words.csv", index=False, encoding="utf-8-sig")
    
# 9. Sort and check coverage
words["Story_Version"] = pd.Categorical(words["Story_Version"], categories=CONDITION_ORDER, ordered=True)
words["StoryNumber"] = pd.to_numeric(words["Story"].str.extract(r"(\d+)")[0], errors="coerce")
words = words.sort_values(["Participant", "StoryNumber", "Story_Version", "WordIndex", "Event Index"]).drop(columns="StoryNumber").reset_index(drop=True)

word_counts = words.groupby(["Participant", "Story", "Story_Version"], observed=True, as_index=False).agg(n_words=("WordIndex", "nunique"))
word_count_summary = word_counts.groupby(["Story", "Story_Version"], observed=True, as_index=False).agg(min_words=("n_words", "min"), median_words=("n_words", "median"), max_words=("n_words", "max"), n_participants=("Participant", "nunique"))
word_count_summary["StoryNumber"] = pd.to_numeric(word_count_summary["Story"].str.extract(r"(\d+)")[0], errors="coerce")
word_count_summary = word_count_summary.sort_values(["StoryNumber", "Story_Version"]).drop(columns="StoryNumber").reset_index(drop=True)

print("\nWord-count summary:")
print(word_count_summary.to_string(index=False))

# 10. Replace RTs outside 100–3000 ms
words["RT_original"] = pd.to_numeric(words["RT"], errors="coerce")
words["RT_replaced"] = words["RT_original"].notna() & ~words["RT_original"].between(100, 3000)
valid_rt = words["RT_original"].where(~words["RT_replaced"])
story_median = valid_rt.groupby([words["Participant"], words["Story"], words["Story_Version"]], observed=True).transform("median")
participant_median = valid_rt.groupby(words["Participant"]).transform("median")
replacement_rt = story_median.fillna(participant_median).fillna(valid_rt.median())
words["RT"] = words["RT_original"]
words.loc[words["RT_replaced"], "RT"] = replacement_rt[words["RT_replaced"]]
words["log_RT"] = np.log(words["RT"])

n_total = words["RT_original"].notna().sum()
n_replaced = words["RT_replaced"].sum()
print("\nTotal RTs:", n_total)
print("Replaced RTs:", n_replaced)
print(f"Replacement proportion: {n_replaced / n_total * 100:.2f}%")

replacement_report = words.groupby("Story_Version", observed=True, as_index=False).agg(Total_RTs=("RT_original", "count"), Replaced_RTs=("RT_replaced", "sum"))
replacement_report["Replacement_Percentage"] = replacement_report["Replaced_RTs"] / replacement_report["Total_RTs"] * 100
overall_report = pd.DataFrame({"Story_Version": ["Overall"], "Total_RTs": [n_total], "Replaced_RTs": [n_replaced], "Replacement_Percentage": [n_replaced / n_total * 100]})
replacement_report = pd.concat([replacement_report, overall_report], ignore_index=True)
replacement_report.to_csv("data/RT_replacement_report.csv", index=False, encoding="utf-8-sig")
print(replacement_report.to_string(index=False))

# 11. Average duplicate observations within participant and word
participant_word_rt = words.groupby(["Participant", "Story", "Story_Version", "WordIndex", "Word", "SentenceIndex", "Word_Condition"], observed=True, as_index=False).agg(RT=("RT", "mean"), log_RT=("log_RT", "mean"), N_Observations=("RT", "size"), RT_Replaced=("RT_replaced", "max"))

# 12. Average each word across participants
word_rt = participant_word_rt.groupby(["Story", "Story_Version", "WordIndex", "Word", "SentenceIndex", "Word_Condition"], observed=True, as_index=False).agg(Mean_RT=("RT", "mean"), SD_RT=("RT", "std"), Mean_log_RT=("log_RT", "mean"), N_Participants=("Participant", "nunique"))

# 13. Sort outputs
participant_word_rt["StoryNumber"] = pd.to_numeric(participant_word_rt["Story"].str.extract(r"(\d+)")[0], errors="coerce")
participant_word_rt = participant_word_rt.sort_values(["Participant", "StoryNumber", "Story_Version", "WordIndex"]).drop(columns="StoryNumber").reset_index(drop=True)

word_rt["StoryNumber"] = pd.to_numeric(word_rt["Story"].str.extract(r"(\d+)")[0], errors="coerce")
word_rt = word_rt.sort_values(["StoryNumber", "Story_Version", "WordIndex"]).drop(columns="StoryNumber").reset_index(drop=True)

# 14. Save outputs
words = words[["Participant", "Story", "Story_Version", "Word", "RT", "Word_Condition", "WordIndex", "SentenceIndex", "Observed_Word", "Token_Match", "RT_original", "RT_replaced", "log_RT"]]
participant_word_rt = participant_word_rt[["Participant", "Story", "Story_Version", "Word", "RT", "Word_Condition", "WordIndex", "SentenceIndex", "log_RT", "N_Observations", "RT_Replaced"]]
word_rt = word_rt[["Story", "Story_Version", "Word", "Mean_RT", "Word_Condition", "WordIndex", "SentenceIndex", "SD_RT", "Mean_log_RT", "N_Participants"]]

words.to_excel("data/reading_data_preprocessed.xlsx", index=False)
participant_word_rt.to_excel("data/participant_word_RT.xlsx", index=False)
word_rt.to_excel("data/word_RT_preprocessed.xlsx", index=False)

print("\nParticipant-word data:", participant_word_rt.shape)
print("Averaged word data:", word_rt.shape)

# # Inspect stimulus sentences requiring fallback
# for story_id, version, sentence_index in [(4, "conceptual", 2), (5, "conceptual", 3)]:
#     story_data = stimulus[stimulus["Story ID"] == story_id]
#     original_text = story_data.loc[story_data["Disruption type"] == "original", "Stories"].iloc[0]
#     version_text = story_data.loc[story_data["Disruption type"] == version, "Stories"].iloc[0]

#     original_sentences = re.split(r"(?<=[.!?])\s+", get_body(original_text))
#     version_sentences = re.split(r"(?<=[.!?])\s+", get_body(version_text))

#     original_tokens = original_sentences[sentence_index - 1].split()
#     version_tokens = version_sentences[sentence_index - 1].split()

#     print(f"\nStory{story_id}, sentence {sentence_index}")
#     print("Original:", original_sentences[sentence_index - 1])
#     print("CD:      ", version_sentences[sentence_index - 1])
#     print("\nOriginal tokens:")
#     print(" ".join(f"{i + 1}:{w}" for i, w in enumerate(original_tokens)))
#     print("\nCD tokens:")
#     print(" ".join(f"{i + 1}:{w}" for i, w in enumerate(version_tokens)))
    
# # Check excluded trials
# print("\nExcluded trial summary:")
# print(excluded_trials.groupby(["Story", "Story_Version", "Observed", "Template"]).size().reset_index(name="N").sort_values("N", ascending=False).head(50).to_string(index=False))

# print("\nDifference:")
# excluded_trials["Difference"] = excluded_trials["Observed"] - excluded_trials["Template"]
# print(excluded_trials["Difference"].value_counts().sort_index().to_string())