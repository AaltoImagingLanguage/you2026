# Predictive coding narrows the condition-level gap between convolutional networks and human brain function in misspelled-word reading

This repo contains Python scripts for dataset generation, Predictive coding model training/evaluation, model-brain comparison and figure producing. The predictive-coding components in this work rely on  [Predify](https://github.com/miladmozafari/predify).

## Data

Project data (non-personal) are stored on OSF: https://osf.io/8472e

After downloading, set `data_dir` in `config.py` to the folder where you extracted the data.

## Installation

The required Python packages are listed in `requirements.txt`. One way to install them is through pip:

```bash
pip install -r requirements.txt
```

## Main analyses

Run all scripts from the repository root. Scripts named after a figure or table (`Fig*`,
`Tab*`, `S*`) are in `figures_tables/` and produce it from the OSF data. **MEG**: needs the
participant-level MEG data, which cannot be shared; the outputs of these steps are included
in the OSF data.

| Step | Scripts | Result |
|---|---|---|
| 1. Images | `01_make_image_datasets.py` | |
| 2. Feedforward backbone | `02_train_backbone.py` | |
| 3. Predictive-coding model | `04_make_pc_model.py`, `05_train_feedback_weights.py` | |
| 4. Hyperparameters, RL2 + RL3 | `06_optimize_hyperparameters.py --sti_type RL23PW --val_fold k`, once per validation fold k = 0–4. Nested cross-validation (held-out predictions, Fig 3A): also add `--test_fold f` for each test fold f = 0–4, with the next fold as validation fold (f = 4: fold 0) | Table 1 |
| 5. Human behavior | `Fig2_human_behavior.py` | Fig 2 |
| 6. Model behavior | `12_eval_runs_behavior.py --run orig` → `Fig3A_heldout_accuracy.py`; `Fig3BC_model_behavior.py` → `Fig3_compose.py` | Fig 3 |
| 7. Condition-wise RSA | `16_model_rdms.py`, `17_meg_trial_patterns.py` (**MEG**) → `18_rsa_parcels.py` → `19_rsa_wholecortex.py`; `21_rsa_controls.py` (cortex average) → `Fig5_Fig6_compose.py` | Figs 5, 6 |
| 8. Effect sizes | `Tab2_effect_sizes_predefined_windows.py` | Table 2 |
| 9. Ridge regression | `28_ridge_grand_average.py` (**MEG**) → `Fig7_ridge.py` | Fig 7 |

Fig 1 is a schematic, and Fig 4 is reproduced from earlier work.

## Supporting analyses

| Analysis | Scripts | Result |
|---|---|---|
| Human catch-trial reaction times | `11_human_rt_stats.py` → `S1Tab_human_rt.py` | S1 Table |
| Update-term contributions and prediction errors | `15_update_terms_dynamics.py` | S1 Fig |
| Retraining with four random seeds | `10_train_backbone_seeds.py`, then `05_`–`07_` with `--out_tag _seedN`; `12_`, `13_` → `S2Tab_seeds.py` | S2 Table, S2 Fig |
| One-hot training control | `02_train_backbone.py --target onehot`, `03_test_backbone_on_stimuli.py` → `S3Tab_soft_vs_onehot.py` | S3 Table |
| RNN and GRU controls and PCoder ablations | `07_`–`09_`, `14_controls_ablations_behavior.py`, `20_`, `21_rsa_controls.py`; specifications in `tables/S6_Tab.tex` | S3, S12 Figs, S6 Table |
| Hyperparameter perturbation | `31_hp_perturbation.py` → `32_hp_perturbation_rsa.py` → `S4Fig_S11Fig_hp_perturbation.py` | S4, S11 Figs |
| Effect sizes in vOT and ST; correction families | `22_effect_sizes_clusters.py` → `S4Tab_effect_sizes_vOT_ST.py`, `S5Tab_correction_families.py` | S4, S5 Tables |
| MEG and model RDMs | `16_model_rdms.py`, `S5Fig_meg_rdms.py` → `make_supporting_information.py` (composite) | S5 Fig |
| Whole-cortex RSA of PCoders 1 and 2 | `19_rsa_wholecortex.py` | S7 Fig |
| Condition-label baselines, severity-partialled RSA, noise ceiling, bootstrap and cross-validated RDMs | `23_`–`25_` → `S8FigC_severity_partial_maps.py`, `S6Fig_S8Fig_rsa_baselines.py`, `S9Fig_reliability.py` | S6, S8, S9 Figs |
| RSA over recurrent timesteps | `30_dynamics_rsa.py` → `S10Fig_rsa_timesteps.py` | S10 Fig |
| Stimulus-level RSA and RDMs | `26_meg_vertex_time_courses.py` (**MEG**) → `27_stimulus_rsa_partial.py --combine_rois`, `S14Fig_stimulus_rdms.py` | S13, S14 Figs |
| Participant-level ridge regression | `29_ridge_participant_level.py` → `S15Fig_ridge_participant_level.py` | S15 Fig |
