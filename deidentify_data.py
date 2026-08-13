import pandas as pd

DATA_FILE = "data/data_exp_221200-v10_task-eyrj.csv"
df = pd.read_csv(DATA_FILE, sep=";", low_memory=False)
df = df[df["Participant Private ID"].notna()].copy()	

# replace Participant Private ID
df["Participant Private ID"] = df["Participant Private ID"].apply(
    lambda x: str(int(x))
    )

raw_id = df["Participant Private ID"].astype("string")
id_map = {
    old_id: f"P{i:03d}"
    for i, old_id in enumerate(sorted(raw_id.unique()), start=1)
}

df["Participant Private ID"] = raw_id.map(id_map)

# remove the test data
test_id = "13647079"
df = df[df["Participant Private ID"] != id_map[test_id]]

# replace Schedule ID
if df["Schedule ID"].isna().any():
    raise ValueError("Schedule ID contains missing values")

session_number = df.groupby("Participant Private ID")["Schedule ID"].transform(
    lambda x: pd.factorize(x, sort=True)[0] + 1
)

df["Schedule ID"] = (
    df["Participant Private ID"]
    + "-S"
    + session_number.astype(int).astype(str).str.zfill(2)
)

# Remove columns that must not appear in the public dataset
must_drop = [
    "UTC Timestamp",
    "UTC Date and Time",
    "Local Timezone",
    "Local Timestamp",
    "Local Date and Time",
    "Participant Completion Code",
    "Participant External Session ID",
]

df = df.drop(columns=must_drop, errors="ignore")

# Remove empty columns
empty_columns = []
for column in df.columns:
    if df[column].empty:
        empty_columns.append(column)
    elif (df[column].unique() != df[column].unique()).all():
        empty_columns.append(column)
    else:
        pass

# ['Repeat Key', 'Participant Starting Group', 'Checkpoint', 'Room ID', 'Context', 
# 'Spreadsheet: store', 'Spreadsheet: QuestionText', 'Spreadsheet: option1', 
# 'Store: Question_Accuracy', 'Store: Correct', 'Store: spreadsheet', 
# 'Store: active_button']
df = df.drop(columns=empty_columns, errors="ignore")

# remove participant's metadata
participant_metadata_columns = [
    "Participant Status",
    "Participant Device Type",
    "Participant Device",
    "Participant OS",
    "Participant Browser",
    "Participant Monitor Size",
    "Participant Viewport Size",
]

df = df.drop(columns=participant_metadata_columns, errors="ignore")

# Check columns with only 1 valid values  
unqiue_columns = []
for column in df.columns:
    if len(df[column].unique()) == 1:
        unqiue_columns.append(column)
        print(f"{column}: {df[column].unique().item()}")
    elif len(df[column][df[column].notna()].unique()) == 1:
        unqiue_columns.append(column)
        print(f"{column}: {df[column][df[column].notna()].unique().item()}")
    else:
        pass
"""
Remove these columns:
Experiment ID: 221200.0
Experiment Version: 10.0
Tree Node Key: task-eyrj
Participant Public ID: BLINDED
Room Order: 0.0
Task Name: STORIES
Task Version: 7.0
branch-ty4v: Consent
branch-um27: Passed Comprehension Questions
Manipulation: Spreadsheet: Speadsheet
Current Spreadsheet: Speadsheet
Response Duration: 0.0
Proportion: 0.0
Tag: button

Not remove these columns - will do later, to avoid bugs
Spreadsheet: O_Q_R4: d) None of the previous.
Spreadsheet: CD_Q_R4: d) None of the previous.
Spreadsheet: RD_Q_R4: d) None of the previous.
"""

df = df.drop(columns=[c for c in unqiue_columns if "R4" not in c], errors="ignore")

# we don't need the full story text
df = df.drop(
    columns=[
        "Spreadsheet: Original",
        "Spreadsheet: CD",
        "Spreadsheet: RD",
    ],
    errors="ignore",
)

# we also don't need the full question text
question_text_columns = [
    "Spreadsheet: Original_question",
    "Spreadsheet: CD_question",
    "Spreadsheet: RD_question",

    "Spreadsheet: O_Q_R1",
    "Spreadsheet: O_Q_R2",
    "Spreadsheet: O_Q_R3",
    "Spreadsheet: O_Q_R4",

    "Spreadsheet: CD_Q_R1",
    "Spreadsheet: CD_Q_R2",
    "Spreadsheet: CD_Q_R3",
    "Spreadsheet: CD_Q_R4",

    "Spreadsheet: RD_Q_R1",
    "Spreadsheet: RD_Q_R2",
    "Spreadsheet: RD_Q_R3",
    "Spreadsheet: RD_Q_R4",
]

df = df.drop(columns=question_text_columns, errors="ignore")

# save 
OUT= "data/data_exp_221200-v10_task-eyrj_deidentified.csv"
df.to_csv(OUT, index=False, encoding="utf-8-sig")


