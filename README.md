# Dual Degree Project: Active Learning for Insurance Fraud Detection with Tweedie Risk Modeling

**Type:** Dual Degree Project (DDP), IIT Madras

## Overview

This project combines **Active Learning (AL)** for insurance fraud detection with **Tweedie risk modeling** for claim severity/frequency prediction. The goal is to reduce labeling cost in fraud investigation pipelines by intelligently querying the most informative claims, while jointly modeling claim risk using the Tweedie distribution commonly used in actuarial pricing. A reinforcement learning (PPO-based) component is also explored as part of the query strategy.

Full methodology, experiments, and results are documented in `DDP_Thesis_MM21B044.pdf`. A presentation summary is available in `DDP_PPT_MM21B044.pdf`, and a video walkthrough in `DDP_Video_MM21B044.mp4`.

## Repository Structure

```
Dual-Degree-Project/
├── data/                       # Raw, processed, and synthetic datasets
├── docs/                       # Literature review, meeting notes, reference papers
├── notebooks/                  # All experiment notebooks (see below)
├── results/                    # Saved experiment outputs (pickled results, trained agents)
├── submission_01/              # First milestone submission
├── submission_02/              # Second milestone submission
├── submission_mid_review/      # Mid-review poster + LaTeX source
├── DDP_Thesis_MM21B044.pdf     # Final thesis
├── DDP_PPT_MM21B044.pdf        # Final presentation
└── DDP_Video_MM21B044.mp4      # Final video presentation
```

### Notebooks (run roughly in order)

| # | Notebook | Purpose |
|---|----------|---------|
| 0 | `0_Insurance_Fraud_Algo_FLow_With_Real_Numbers.ipynb` | End-to-end pipeline walkthrough with real numeric examples |
| 1 | `1_Cost_Sensitive_RL_paper_implementation.ipynb` | Implementation of reference cost-sensitive RL paper |
| 2 | `2_data_cleaning_scaling.ipynb` | Data cleaning and feature scaling |
| 3 | `3_AL_implementation.ipynb` | Core Active Learning implementation |
| 4 | `4_AL_monthly_implementation.ipynb` | AL adapted to monthly/sequential data batches |
| 5 | `5_Clean_compilation.ipynb` | Consolidated, cleaned-up pipeline |
| 6 | `6_AL_template.ipynb` | Reusable AL template/framework |
| 7 | `7_Last_data_generation_with_results_final.ipynb` | Final synthetic data generation with results |
| 8 | `8_Modelling_after_last_data_generation_with_accuracy_AL.ipynb` | Final fraud model + AL accuracy evaluation |
| 9 | `9_Tweedie_and_dual_modeling.ipynb` | Tweedie risk model + dual (fraud + risk) modeling |
| 10 | `10_Tweedie_final.ipynb` | Final Tweedie modeling results |

## Data

- `freMTPL2freq.csv`, `freMTPL2sev.csv` — French Motor Third-Party Liability dataset (frequency & severity), used for Tweedie modeling.
- `synthetic_insurance_claims_with_fraud_3%_label.csv` — synthetic claims data with a 3% fraud label rate, used for AL experiments.
- `train_flagged.csv`, `train_flagged_plus_queried.csv`, `test_set.csv` — training/test splits used across AL iterations.
- `raw_data.zip`, `data_5L_diff.zip`, `data_5L_new.zip` — archived raw/intermediate datasets (extract before use).

> Some CSVs are large; ensure you have sufficient disk space when extracting the `.zip` archives.

## Setup

**Requirements:** Python 3.9+, pip, Jupyter

```bash
# 1. Clone/copy the project, then navigate into it
cd Dual-Degree-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Extract zipped data files
cd data
unzip raw_data.zip
unzip data_5L_new.zip
cd ..

# 5. Launch Jupyter
jupyter notebook notebooks/
```

### Core dependencies

If `requirements.txt` is not present, install manually:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn \
            stable-baselines3 gymnasium torch \
            jupyter notebook tqdm
```

- `scikit-learn` — base ML models, Tweedie regressor
- `stable-baselines3` + `gymnasium` — PPO surrogate agent for RL-based query strategy
- `torch` — backend for SB3 / any deep learning components

## Results

Pre-computed results are stored in `results/` for reference without rerunning experiments:

- `al_experiment_results.pkl` — Active Learning experiment outputs
- `ensemble_ablation_results.pkl` — ablation study on ensemble model choices
- `rl_phase2_results.pkl` — phase 2 RL experiment outputs
- `ppo_surrogate_agent.zip` — trained PPO agent (load via `stable_baselines3.PPO.load(...)`)

## Documentation

The `docs/` folder contains supporting literature review, reference papers, and meeting notes used during the project — useful for understanding the background and design choices, but not required to run the code.

## Submissions

- `submission_01/`, `submission_02/` — intermediate milestone deliverables
- `submission_mid_review/` — mid-review poster (LaTeX source + compiled PDF)

## Citation

If referencing this work, please cite:

> Nikshay Jain, "Active Learning for Insurance Fraud Detection with Tweedie Risk Modeling," Dual Degree Project, IIT Madras, 2025.
