# SPR Distortion

Code and analysis pipeline for studying how **conceptual distortion (CD)** and **referential distortion (RD)** propagate through human self-paced reading and large language models.

The project compares three story versions:

* **Original**: unaltered baseline
* **CD**: conceptual distortions created by replacing a contextually appropriate lexical item with a semantically incongruent alternative
* **RD**: referential distortions created by altering pronouns or determiners so that the intended discourse referent is disrupted

Rather than treating each manipulation as a single violation cost, the analyses trace its consequences around the distortion position (**K**) and across subsequent linguistic context.

## Overview

The repository contains three linked analysis streams:

1. **Human self-paced reading**

   * reconstruction and alignment of word-level reading times
   * lag-1 residualization of log RT
   * mixed-effects analyses around K
   * downstream propagation and decay analyses
   * sentence-boundary modulation

2. **Qwen3-4B**

   * word-level surprisal
   * output-layer hidden-state representations
   * cosine distance from the Original condition
   * downstream decay and boundary analyses

3. **Llama-3.2-3B replication**

   * the same surprisal and representational analyses applied to Llama

## Workflow

The intended analysis order is:

```bash
sbatch 1_data_preprocess.sh
sbatch 2_reading_time.sh
sbatch 3_qwen.sh
sbatch 4_llama.sh
sbatch 5_plot_figures.sh
```

The supplied SLURM scripts assume a Conda environment named `graph` and write job output to `logs/`.

## 1. Data preprocessing

```bash
sbatch 1_data_preprocess.sh
```

The standard reproducibility workflow starts from the **already de-identified Gorilla dataset**.

Therefore, `deidentify_data.py` should **not** be run as part of the normal reproduction pipeline. It is retained only to document how the original restricted Gorilla export was anonymized before release.

For the shared dataset, Step 1 should run:
(Note: please delete ```python -u organize_data.py```)

```bash
python -u organize_data.py
```

`organize_data.py`:

* constructs the word-level stimulus template
* identifies conceptual and referential distortion positions
* assigns positions around K
* aligns observed participant word sequences with the stimulus
* reconstructs word-level reading times
* generates quality-control outputs

Important outputs include:

```text
data/story_word_template.xlsx
data/participant_word_RT.xlsx
data/stimulus_alignment_qc.csv
data/alignment_report.csv
data/alignment_excluded_trials.csv
data/k_sentence_end_distance_detail.csv
data/k_sentence_end_distance_summary.csv
```

If `story_word_template.xlsx` and `participant_word_RT.xlsx` are already available, the main statistical analyses can begin directly from Step 2.

## 2. Human reading-time analysis

```bash
sbatch 2_reading_time.sh
```

This runs:

```bash
python -u analysis_residual_mixedlm.py
python -u analysis_residual_decay.py
```

The RT analysis uses log-transformed reading times and removes lag-1 autocorrelation before evaluating distortion-related effects. Lexical controls include word frequency and word length.

The analyses examine:

* positions around K
* downstream effects following K
* differences between CD and RD trajectories
* decay shape across subsequent words
* sentence-boundary modulation

Outputs are written primarily to:

```text
results/residual_analysis/
results/distance_analysis/
```

## 3. Qwen3-4B analysis

```bash
sbatch 3_qwen.sh
```

This runs:

```bash
python -u analysis_qwen_ols.py
python -u analysis_qwen_surprisal_decay.py
python -u analysis_qwen_cosine_decay.py
```

Model:

```text
Qwen/Qwen3-4B-Base
```

For each story version, the scripts extract:

* word-level surprisal
* output-layer hidden states
* cosine distance between distorted and Original representations

Surprisal and representational displacement are then analyzed as a function of distance from K and sentence boundaries.

Outputs are written to:

```text
results/qwen3/
```

## 4. Llama-3.2-3B replication

```bash
sbatch 4_llama.sh
```

This runs:

```bash
python -u analysis_llama3_ols.py
python -u analysis_llama3_surprisal_decay.py
python -u analysis_llama3_cosine_decay.py
```

Model:

```text
meta-llama/Llama-3.2-3B
```

The Llama analyses reproduce the Qwen pipeline using the same stimulus template, distortion mapping, surprisal measures, output-layer representations, decay analyses, and sentence-boundary tests.

Outputs are written to:

```text
results/llama32/
```

## 5. Figures

```bash
sbatch 5_plot_figures.sh
```

This runs:

```bash
python -u plot_figures.py
```

The plotting script combines the human RT and LLM analyses into the manuscript figures.

## Distortion positions

`K` denotes the manipulated word or manipulated span.

The local analyses also use:

```text
K-3, K-2, K-1, K, K+1, K+2, K+3, K_END
```

`K_END` denotes the sentence-final position of a sentence containing a distortion.

For downstream distance analyses, trajectories extend from K to the end of the current sentence and do not continue into the following sentence. Sentence-boundary analyses therefore concern effects **at sentence boundaries**, rather than distortions crossing sentence boundaries.

## Measures

### Human RT

The primary behavioral outcome is residualized log RT. Lag-1 log RT is modeled first to reduce serial autocorrelation, after which distortion-related effects are evaluated on the residualized outcome.

### LLM surprisal

Word-level surprisal measures the predictive penalty assigned to the current word given its preceding context.

### Representational distance

Hidden-state effects are quantified using cosine distance between distorted and matched Original output-layer representations.

This measure captures geometric displacement of the model state and should not be interpreted directly as processing difficulty.

## Repository structure

```text
SPR_distortion/
├── 1_data_preprocess.sh
├── 2_reading_time.sh
├── 3_qwen.sh
├── 4_llama.sh
├── 5_plot_figures.sh
│
├── deidentify_data.py
├── organize_data.py
│
├── analysis_residual_mixedlm.py
├── analysis_residual_decay.py
│
├── analysis_qwen_ols.py
├── analysis_qwen_surprisal_decay.py
├── analysis_qwen_cosine_decay.py
│
├── analysis_llama3_ols.py
├── analysis_llama3_surprisal_decay.py
├── analysis_llama3_cosine_decay.py
│
├── plot_figures.py
├── data/
├── results/
└── logs/
```

## Requirements

Core Python dependencies include:

```text
numpy
pandas
matplotlib
statsmodels
tqdm
wordfreq
torch
transformers
openpyxl
```

GPU access is recommended for the Qwen and Llama analyses. The supplied SLURM scripts request one CUDA GPU for both model pipelines.

Llama weights may require an authenticated Hugging Face account with access to:

```text
meta-llama/Llama-3.2-3B
```

## Reproducibility notes

* Random seed for the LLM analyses is set to `42`.
* Qwen and Llama use the same stimulus template and distortion-position mapping.
* Distorted positions are explicitly aligned with their corresponding Original positions before condition contrasts are calculated.
* QC outputs generated during preprocessing should be inspected before rerunning analyses on modified data.
* `deidentify_data.py` is **not part of the standard reproduction workflow**. It documents preprocessing of the restricted original Gorilla export only.

