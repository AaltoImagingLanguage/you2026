# Predictive coding narrows the gap between convolutional networks and human brain function in misspelled-word reading

Jiaxin You, Riitta Salmelin, and Marijn van Vliet (2026). "Predictive coding narrows the gap between convolutional networks and human brain function in misspelled-word reading". 

This repo contains Python scripts for dataset generation, Predictive coding model training/evaluation, model-brain comparison and figure producing.  The predictive-coding components in this work rely on [Predify (GitHub)](https://github.com/miladmozafari/predify).

**Data**
- Project data (non-personal) are stored on OSF: https://osf.io/8472e
- After downloading, set `data_dir` in `config.py` to the folder where you extracted the data.

**Installation**
The required Python packages are listed in `requirements.txt`. One way to install them is through pip:

`pip install -r requirements.txt`

**Repository layout**
- `00_dataset_generation.py` : Prepare image stimuli and dataset for training and validating.
- `01_train_backbone_model.py` : Train the backbone feedforward CNN model.
- `01_test_stimuli_backbone.py` : Test the experimental stimuli on the trained backbone model.
- `02_generate_pred_model.py` : Generate predictive coding model architecture.
- `03_train_feedback_weights.py` : Train feedback components of predictive coding model.
- `04_train_pcoder_hps.py` : Hyperparameter optimation for predictive coding model using five-fold cross validation.
- `05_Fig2b_model_behavior_mean_folds.py` : Produce model behaviorly performance with hyperparameters averaged across folds (Figure 2b).
- `05_Fig2c-d_model_behavior_median_fold.py` : Produce model behavior with meidan hyperparameters across folds.
- `06_meg_rois_rdms.py` / `06_model_rdms.py` : RDM computations for MEG ROIs and model representations.
- `07_*` : Computate RSA time courses between RDMs of model and parcel representation and plot the results.
- `08_*` : Linear mapping / ridge mapping from model to parcel representations and plot the results.
- `config.py` : Central configuration values used by scripts.
- `pnet/` : Predictive coding model architecture.





<!-- Reproducibility notes
- Configuration files in `config_pnet/` and `config_train_fb/` contain training and model hyperparameters used to reproduce published results.
- Cross-validation fold assignments are stored in `data/cv_fold_assignments.json`.

Development
- Use `pnet/` for model implementations. To run or modify model code, edit `pnet/pvgg16v1.py` and other modules in `pnet/` accordingly.
- Utility helpers are in `utility.py` and `ridge.py` for mapping/ridge regression functionality.

Testing and running on HPC
- Example session for interactive runs on a cluster:

```bash
# activate environment
## Predictive coding narrows the gap between convolutional networks and human brain function in misspelled-word reading

Jiaxin You, Riitta Salmelin, and Marijn van Vliet (2026).

This repository contains Python scripts for dataset generation, predictive-coding model training and evaluation, model–brain comparisons (RSA/RDM), and figure generation for the project. The predictive-coding components in this work rely on Predify: [Predify (GitHub)](https://github.com/miladmozafari/predify).

**Data**
- Project data (non-personal) are hosted on OSF: https://osf.io/8472e
- After downloading, set `data_dir` in `config.py` to the folder where you extracted the data.

**Installation**
- The required Python packages are listed in `requirements.txt`.
- Recommended environment (conda):

```bash
conda create -n you2026 python=3.10 -y
conda activate you2026
pip install -r requirements.txt
```

**Repository layout**
- `00_dataset_generation.py` — prepare image stimuli and dataset metadata.
- `01_train_backbone_model.py` — train the backbone (feedforward) CNN.
- `01_test_stimuli_backbone.py` — test stimuli preprocessing and backbone behavior.
- `02_generate_pred_model.py` — build predictive-coding model instances.
- `03_train_feedback_weights.py` — train feedback components for predictive-coding models.
- `04_train_pcoder_hps.py` — hyperparameter optimization (PCoder) using cross-validation.
- `05_Fig2b_model_behavior_mean_folds.py` — produce model behavior (mean across folds) for Figure 2b.
- `05_Fig2c-d_model_behavior_median_fold.py` — median-fold behavior figures (Figure 2c–d).
- `06_meg_rois_rdms.py`, `06_model_rdms.py` — compute RDMs for MEG ROIs and model layers.
- `07_*` — RSA time-course plotting and whole-brain visualization scripts.
- `08_*` — linear / ridge mapping between model and parcel representations and plotting.
- `config.py` — central configuration and path settings.
- `pnet/` — local package implementing model architectures (e.g. predictive-coding wrappers).

**Quick usage**
- Inspect script options (many scripts accept CLI args or read `.toml`/`config.py` files):

```bash
python 00_dataset_generation.py --help
python 01_train_backbone_model.py --help
```

- Example pipeline (adjust to your cluster and paths):

```bash
python 00_dataset_generation.py
python 01_train_backbone_model.py
python 02_generate_pred_model.py
python 05_Fig2b_model_behavior_mean_folds.py
```

**Reproducibility notes**
- Configuration files in `config_pnet/` and `config_train_fb/` record model and training hyperparameters used for the experiments.
- Cross-validation fold assignments are in `data/cv_fold_assignments.json`.

**Development**
- Implementations for the predictive-coding/backbone models live under `pnet/`. Edit `pnet/pvgg16v1.py` and related modules to modify model code.
- Helpers for mapping and ridge regression are in `utility.py` and `ridge.py`.

**Running on HPC / Example session**

```bash
# activate environment
conda activate you2026
# run training or evaluation scripts (adapt GPU/SLURM options as needed)
python 01_train_backbone_model.py
```

**Contributing**
- Fork the repo and open a pull request. When adding features, include a short runnable example or test where feasible.

**License**
- No license file is included in this repository. Confirm reuse terms with the project owner before redistribution.

**Contact**
- For questions about reproducing results or the code, contact the repository owner or project lead listed in the repository metadata.

---

If you'd like, I can:
- add a short `examples/` folder with an end-to-end demo, or
- expand the README with CLI examples for the main scripts and common argument values.

Note: I added a markdown link to Predify at the top. If you prefer a different URL (for example the Predify project page or a documentation site), tell me the preferred link and I will update it.
