# Economically Optimized Pure Payout Pricing via Stratified Expected Deviance Reduction (SEDR) in Active Learning

**Dual Degree Project | Indian Institute of Technology Madras**

---

## Overview

This project investigates the application of **Active Learning (AL)** to Tweedie-distributed pure insurance premium prediction on the French Motor Third-Party Liability (MTPL) dataset. The central contribution is the **Stratified Expected Deviance Reduction (SEDR)** framework — an acquisition strategy engineered to decouple epistemic model disagreement from aleatoric claim severity, while enforcing structural Poisson frequency anchors across risk strata.

The project documents the full architectural evolution from a single-model heuristic uncertainty proxy, through the discovery and resolution of a zero-mass starvation bug induced by naive bootstrapping, to the empirical validation of a $K=10$ True Ensemble as the global optimum. A Markov Decision Process (MDP) formulation of the active learning budget allocation problem, solved via Proximal Policy Optimization (PPO), is also developed and empirically tested.

> **TL;DR:** Naive uncertainty-based active learning catastrophically degrades Tweedie insurance models by conflating claim severity with model ignorance. SEDR with a $K=10$ True Ensemble resolves this, achieving maximum Gini discrimination, minimum Tweedie Deviance, and near-zero Zero-Mass Calibration Error simultaneously — outperforming random sampling, naive variance, and a trained PPO policy.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Key Results](#key-results)
- [Environment Setup](#environment-setup)
- [Data Setup](#data-setup)
- [Notebook Execution Guide](#notebook-execution-guide)
- [Results Artifacts](#results-artifacts)
- [Submissions](#submissions)
- [Academic Deliverables](#academic-deliverables)
- [Citation](#citation)

---

## Project Structure

```
Dual-Degree-Project/
│
├── data/                                           # All datasets
│   ├── freMTPL2freq.csv                            # French MTPL frequency table (677,991 policies)
│   ├── freMTPL2sev.csv                             # French MTPL severity table (claim amounts)
│   ├── synthetic_insurance_claims_with_fraud_3%_label.csv  # Synthetic fraud detection dataset
│   ├── train_flagged.csv                           # Processed training set with AL flags
│   ├── train_flagged_plus_queried.csv              # Training set after AL query augmentation
│   ├── test_set.csv                                # Held-out evaluation set
│   ├── raw_data.zip                                # Raw data archive
│   ├── data_5L_new.zip                             # 500k-sample processed dataset (v1)
│   └── data_5L_diff.zip                            # 500k-sample processed dataset (v2, diff split)
│
├── notebooks/                                      # Jupyter notebooks (see execution guide below)
│   ├── 0_Insurance_Fraud_Algo_Flow_With_Real_Numbers.ipynb
│   ├── 1_Cost_Sensitive_RL_paper_implementation.ipynb
│   ├── 2_data_cleaning_scaling.ipynb
│   ├── 3_AL_implementation.ipynb
│   ├── 4_AL_monthly_implementation.ipynb
│   ├── 5_Clean_compilation.ipynb
│   ├── 6_AL_template.ipynb
│   ├── 7_Last_data_generation_with_results_final.ipynb
│   ├── 8_Modelling_after_last_data_generation_with_accuracy_AL.ipynb
│   ├── 9_Tweedie_and_dual_modeling.ipynb
│   └── 10_Tweedie_final.ipynb                            # Final SEDR + Ensemble + MDP pipeline
│
├── results/                                        # Serialized experiment outputs
│   ├── al_experiment_results.pkl                   # SEDR vs baseline AL trajectories
│   ├── ensemble_ablation_results.pkl               # K=5/10/15 ablation study results
│   ├── rl_phase2_results.pkl                       # PPO policy deployment results
│   └── ppo_surrogate_agent.zip                     # Trained PPO agent (stable-baselines3)
│
├── submission_01/                                  # First formal submission (Lloyds)
│   ├── data_creation.ipynb
│   ├── AL_submission.ipynb
│   ├── data_5L_new.zip
│   └── LLoyds ppt.pdf
│
├── submission_02/                                  # Second formal submission
│   └── AL_implementation_improved.ipynb
│
├── submission_mid_review/                          # DDP Mid-Review Poster (LaTeX source)
│   ├── main.tex
│   ├── refs.bib
│   ├── custom-defs.tex
│   ├── beamerthemegemini.sty
│   ├── beamercolorthemeseagull.sty
│   ├── DDP_Mid_review_poster_nikshay_jain_mm21b044.pdf
│   └── logos/
│
├── docs/                                           # Literature, meeting notes, reference PDFs
│   ├── ActiveLearning_TweedieRisk.pdf
│   ├── Tweedie Modeling in Insurance.pdf
│   ├── Roadmap ahead of tweedie.pdf
│   ├── insurance_case_ppt.pdf
│   ├── tweedie_meet_transcription.txt
│   ├── Insurance AL/
│   │   ├── Insurance Fraud Detection using Active Learning.pdf
│   │   └── ...
│   └── Literature review/
│       ├── Literature Review - Articles, Blogs, Videos.pdf
│       ├── Literature review - Papers.pdf
│       ├── Literature review - RL-Bandits.pdf
│       └── ...
│
├── DDP_Thesis_MM21B044.pdf                         # Final thesis document
├── DDP_PPT_MM21B044.pdf                            # Final presentation slides
└── DDP_Video_MM21B044.mp4                          # Project walkthrough video
```

---

## Key Results

| Strategy | Tweedie Deviance ↓ | Gini Index ↑ | ZMCE ↓ |
|---|---|---|---|
| One-Shot Baseline | ~220 | ~0.14 | — |
| Random (NSL) | ~195 | ~0.13 | ~0.05 |
| Naive Variance AL | 945.6 | 0.121 | 0.0017 *(illusory)* |
| Epistemic-Only AL | 292.0 | 0.099 | ~0.01 |
| PPO MDP Policy | ~171 | ~0.13 | ~0.10 |
| **SEDR + K=10 Ensemble** | **148.68** | **0.1850** | **0.0020** |

The Naive Variance ZMCE of 0.0017 is not a calibration victory — it is the **ZMCE Illusion**, where the model collapses to predicting $P(Y=0) \approx 1$ for the entire policyholder population. SEDR is the only strategy achieving global dominance across all three metrics simultaneously.

---

## Environment Setup

### Prerequisites

- Python 3.9 or higher
- `pip` or `conda` package manager
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/Dual-Degree-Project.git
cd Dual-Degree-Project
```

### Step 2 — Create a Virtual Environment

**Using `venv` (recommended):**

```bash
python -m venv ddp_env
# On Windows:
ddp_env\Scripts\activate
# On Linux/macOS:
source ddp_env/bin/activate
```

**Using `conda`:**

```bash
conda create -n ddp_env python=3.10
conda activate ddp_env
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the core stack manually:

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn joblib
pip install stable-baselines3[extra] gymnasium
pip install jupyter ipykernel tqdm
```

Key version constraints to be aware of:

| Package | Minimum Version | Notes |
|---|---|---|
| `lightgbm` | ≥ 3.3 | Required for `tweedie_variance_power` objective |
| `stable-baselines3` | ≥ 2.0 | PPO agent serialization format |
| `scikit-learn` | ≥ 1.1 | OOF cross-validation utilities |
| `pandas` | ≥ 1.4 | `merge` behavior on large frames |
| `gymnasium` | ≥ 0.26 | Replaces legacy `gym` for MDP environment |

### Step 4 — Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name=ddp_env --display-name "DDP (Python 3.10)"
```

### Step 5 — Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

Open notebooks from the `notebooks/` directory and select the **DDP (Python 3.10)** kernel.

---

## Data Setup

All raw data files are provided in `data/`. No external downloads are required.

### Primary Dataset — freMTPL2 (French MTPL Insurance)

The two CSV files must be joined before modeling:

```python
import pandas as pd

freq = pd.read_csv("data/freMTPL2freq.csv")
sev  = pd.read_csv("data/freMTPL2sev.csv")

# Aggregate multiple claims per policy
sev_agg = sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()

# Left join: policies with no claims get ClaimAmount = 0
df = freq.merge(sev_agg, on="IDpol", how="left")
df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

# Engineer the pure premium target
df["Exposure"] = df["Exposure"].clip(lower=1e-4)
df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
```

### Supplementary Dataset — Synthetic Fraud Detection

Used in early-phase notebooks (0–3) for insurance fraud detection experiments with Active Learning:

```python
fraud_df = pd.read_csv("data/synthetic_insurance_claims_with_fraud_3%_label.csv")
```

This dataset carries a 3% fraud prevalence label, consistent with real-world class imbalance in P&C fraud detection.

### Pre-processed Splits

For notebooks 7 onwards, pre-processed splits are available directly:

```python
train = pd.read_csv("data/train_flagged_plus_queried.csv")
test  = pd.read_csv("data/test_set.csv")
```

### Loading Results Artifacts

Serialized experiment results can be loaded without re-running the full AL loop:

```python
import pickle

with open("results/al_experiment_results.pkl", "rb") as f:
    al_results = pickle.load(f)

with open("results/ensemble_ablation_results.pkl", "rb") as f:
    ablation_results = pickle.load(f)

with open("results/rl_phase2_results.pkl", "rb") as f:
    rl_results = pickle.load(f)
```

### Loading the Trained PPO Agent

```python
from stable_baselines3 import PPO

ppo_agent = PPO.load("results/ppo_surrogate_agent")
obs, _ = env.reset()
action, _ = ppo_agent.predict(obs, deterministic=True)
```

---

## Notebook Execution Guide

The notebooks are numbered to reflect their development chronology. They fall into four logical phases:

### Phase 0 — Fraud Detection Precursor *(Background Context)*

These notebooks represent the initial insurance fraud detection work that preceded the Tweedie pricing problem. They are not required to reproduce the core SEDR results but provide important methodological context.

| Notebook | Description |
|---|---|
| `0_Insurance_Fraud_Algo_Flow_With_Real_Numbers.ipynb` | Algorithm flowchart with real fraud dataset statistics |
| `1_Cost_Sensitive_RL_paper_implementation.ipynb` | Implementation of cost-sensitive RL for fraud detection (paper replication) |

### Phase 1 — Data Engineering and Early AL Prototypes

| Notebook | Description | Key Output |
|---|---|---|
| `2_data_cleaning_scaling.ipynb` | freMTPL2 join, feature engineering, exposure normalization | `train_flagged.csv`, `test_set.csv` |
| `3_AL_implementation.ipynb` | First AL loop implementation on fraud dataset | Baseline AL metrics |
| `4_AL_monthly_implementation.ipynb` | AL with temporal monthly batching | Temporal drift analysis |

### Phase 2 — Compilation and Framework Stabilization

| Notebook | Description |
|---|---|
| `5_Clean_compilation.ipynb` | Clean consolidated pipeline — first end-to-end run |
| `6_AL_template.ipynb` | Reusable AL loop template with pluggable acquisition functions |

### Phase 3 — Tweedie Modeling and SEDR Development

| Notebook | Description | Key Output |
|---|---|---|
| `7_Last_data_generation_with_results_final.ipynb` | Final data splits with stratified AL tracking flags | `train_flagged_plus_queried.csv` |
| `8_Modelling_after_last_data_generation_with_accuracy_AL.ipynb` | Full AL loop with all four strategies; stratified WMAE analysis | `al_experiment_results.pkl` |
| `9_Tweedie_and_dual_modeling.ipynb` | Tweedie power $p$ estimation; single-model SEDR vs baselines | Deviance/Gini curves |

### Phase 4 — True Ensemble, MDP, and Final Validation *(Core Contribution)*

| Notebook | Description | Key Output |
|---|---|---|
| `10_Tweedie_final.ipynb` | **Full SEDR pipeline:** True Ensemble ($K \in \{5, 10, 15\}$) ablation; MDP formulation; PPO training + deployment; dual failure validation | `ensemble_ablation_results.pkl`, `rl_phase2_results.pkl`, `ppo_surrogate_agent.zip` |

> **To reproduce the core results**, run only `2_data_cleaning_scaling.ipynb` followed by `10_tweedie.ipynb`. All other notebooks document the research evolution and are self-contained.

---

## Results Artifacts

All serialized results in `results/` allow full metric reproduction and plot generation without re-running the computationally intensive AL loops.

| File | Contents | Estimated Size |
|---|---|---|
| `al_experiment_results.pkl` | Per-iteration Deviance, Gini, ZMCE, WMAE for all 4 strategies × 10 iterations | ~2 MB |
| `ensemble_ablation_results.pkl` | Same metrics for K=5, K=10, K=15 ensemble ablation | ~4 MB |
| `rl_phase2_results.pkl` | PPO policy deployment metrics + surrogate training curves | ~1 MB |
| `ppo_surrogate_agent.zip` | Trained PPO agent weights (stable-baselines3 format) | ~500 KB |

---

## Submissions

### `submission_01/` — Lloyds Banking Group Presentation

Contains the initial project submission including data generation pipeline and early AL results presented to Lloyds Banking Group:

- `data_creation.ipynb` — Dataset construction pipeline used for the external presentation
- `AL_submission.ipynb` — AL results notebook submitted for review
- `LLoyds ppt.pdf` — Presentation slides

### `submission_02/` — Improved AL Implementation

Contains the improved AL implementation submitted for the second review cycle:

- `AL_implementation_improved.ipynb` — Refined strategy with corrected ZMCE computation and exposure-weighted Gini

### `submission_mid_review/` — DDP Mid-Review Poster

LaTeX source for the academic mid-review poster presented at IIT Madras:

```bash
# To compile the poster locally (requires a LaTeX distribution, e.g. TeX Live)
cd submission_mid_review
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled output is also provided as `DDP_Mid_review_poster_nikshay_jain_mm21b044.pdf`.

---

## Academic Deliverables

| File | Description |
|---|---|
| `DDP_Thesis_MM21B044.pdf` | Full thesis document |
| `DDP_PPT_MM21B044.pdf` | Final viva presentation slides |
| `DDP_Video_MM21B044.mp4` | Project walkthrough video |
| `docs/ActiveLearning_TweedieRisk.pdf` | Core reference paper on AL for Tweedie risk models |
| `docs/Tweedie Modeling in Insurance.pdf` | Actuarial background reference |
| `docs/Literature review/` | Comprehensive literature survey (papers, blogs, RL/bandits) |

---

## Methodology Summary

The SEDR framework operates as follows at each of the 10 active learning iterations:

1. **Stratify** the unlabeled pool into 5 risk quintiles $Q_1, \dots, Q_5$ by predicted mean $\hat{\mu}$.
2. **Anchor** 30% of the query budget $B$ to $Q_1$ (safe drivers) to prevent Poisson frequency collapse.
3. **Score** instances in $Q_{2:5}$ by epistemic uncertainty — the inter-model variance across a $K=10$ True Ensemble:
$$U_{epistemic}(x_i) = \frac{1}{K-1} \sum_{k=1}^{K} \left(\hat{\mu}_k(x_i) - \bar{\mu}(x_i)\right)^2$$
4. **Query** the top-$b_k$ highest-uncertainty instances per stratum and label them.
5. **Retrain** the full ensemble on the expanded labeled set with dynamic leaf sizing:
$$\text{min\_data\_in\_leaf} = \max\!\left(5,\ \left\lfloor \frac{|\mathcal{X}_{train}|}{50} \right\rfloor\right)$$
6. **Evaluate** on the held-out test set via Tweedie Deviance, Gini Index, and ZMCE.

---

## Citation

If this work is referenced in academic work, please cite:

```bibtex
@thesis{jain2026sedr,
  author       = {Nikshay Jain},
  title        = {Economically Optimized Pure Payout Pricing via Stratified Expected
                  Deviance Reduction (SEDR) in Active Learning},
  school       = {Indian Institute of Technology Madras},
  year         = {2026},
  type         = {Dual Degree Project Thesis},
  note         = {Roll No. MM21B044}
}
```

---

*For questions regarding the codebase, contact: mm21b044@smail.iitm.ac.in*
