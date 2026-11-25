"""Parameters and file locations shared by all scripts.

Data: download the data release from OSF (https://osf.io/8472e) into ../data (the
folder next to this repository), or set the environment variable YOU2026_DATA to its
location. Steps that start from
the participant-level MEG data need access to the private data; set the environment
variable YOU2026_PRIVATE_DATA to its folder.
"""
import os

import matplotlib.pyplot as plt
import torch
from filename_templates import FileNames

plt.style.use("seaborn-v0_8-whitegrid")
# DejaVu Sans (bundled with matplotlib) as fallback where Arial is not installed
plt.rcParams["font.sans-serif"] = list(plt.rcParams.get("font.sans-serif", [])) + ["DejaVu Sans"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- model
model_name = "vgg16"
version = "v1"
temperature = 1        # temperature of the soft-label targets
max_timestep = 29      # recurrent timesteps evaluated: t = 0..28
fb_timestep = 25       # readout timestep of the predictive-coding model
SAME_PARAM = False     # separate hyperparameters for every PCoder
FF_START = True        # recurrent inference starts from the feedforward state

# hyperparameter optimization
k = 5                  # cross-validation folds
seed = 0               # seed of the fold assignment
epochs_hps = 5000      # maximum number of epochs

# ---------------------------------------------------------------- experiment
event_id = {"RW": 1, "RL1PW": 2, "RL2PW": 3, "RL3PW": 4}
n = 540                # non-catch stimuli (135 per condition)
subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-09",
            "sub-11", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17", "sub-18", "sub-19",
            "sub-20", "sub-21", "sub-22", "sub-24", "sub-25", "sub-26", "sub-27"]

# ---------------------------------------------------------------- MEG
SUBJECT = "fsaverage"
parc = "aparc.a2009s_custom_gyrus_sulcus_800mm2"   # 137 cortical parcels
rois = {"ST": 65, "vOT": 40}                       # post hoc follow-up ROIs (parcel index)
f_down_sampling = 100                              # Hz
baseline_window = (-0.2, 0)
time_len = 1.1
snr_epoch = 3.0                                    # lambda2 = 1 / snr**2 for epochs
snr_sti = 1.0                                      # ... and for single stimuli
pick_ori = "normal"
time_step = 2                                      # 20-ms steps between analysis windows
time_interval = 0.1                                # 100-ms analysis windows
time_windows = [[0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1.1]]  # 200-ms test windows

# model-brain comparison
metric_rsa = "spearman"
metric_rdm = "correlation"
n_splits = 10          # cross-validation folds of the ridge regression
nc = 0                 # PCA components before the ridge regression (0: none)

# ---------------------------------------------------------------- figures
fb_colors = ["lightblue", "lightcoral"]                  # without / with feedback
category_colors = ["#000000", "#0072B2", "#E69F00", "#D55E00"]  # RW, RL1, RL2, RL3
figures_dir = "./figures"
for _sub in ("behav", "rsa", "ridge", "dynamics", "effect_sizes", "rsa_baselines", "supporting_information"):
    os.makedirs(f"{figures_dir}/{_sub}", exist_ok=True)

# ---------------------------------------------------------------- file names
fname = FileNames()
fname.add("data_dir", os.environ.get("YOU2026_DATA", "../data"))
fname.add("figures_dir", figures_dir)
fname.add("log_dir", "./logs/")

# stimuli and training images
fname.add("dataset_dir", "{data_dir}/images_dataset/")
fname.add("fonts_dir", "{data_dir}/fonts/")
fname.add("exper_data_dir", "{data_dir}/dataset/")
fname.add("word2idx_dir", "{exper_data_dir}/word2idx.pkl")
fname.add("stimuli_con_dir", "{exper_data_dir}/rep_stimuli.csv")
fname.add("cv_folds", "{data_dir}/cv_fold_assignments.json")

# trained models
fname.add("net_dir", "{data_dir}/model_ckps/")
fname.add("bb_dir", "{net_dir}/ff_backbone/")
fname.add("pnet_dir", "{net_dir}/fb_pnet/")
fname.add("hps_dir", "{net_dir}/hps/")
fname.add("ff_ckpt", "{bb_dir}/vgg16_checkpoint_best.pth.tar")
fname.add("pcoder_ckpt", "{pnet_dir}/p_vgg16_Separate_v1_best_pc")
fname.add("hps_ckpt", "{hps_dir}/vgg16_v1_best_hps_fold{n_fold}.pth")
fname.add("hps_ckpt_RL23", "{hps_dir}/vgg16_v1_best_hps_RL23_fold{n_fold}.pth")
fname.add("log_hps", "{log_dir}/hps/runs_train_hps_{n_step}ts_{sti_type}")

# behavior and model outputs
fname.add("stimuli_dir", "{data_dir}/behavior/")
fname.add("human_behav", "{data_dir}/behavior/performance.csv", mkdir=True)
fname.add("accs", "{data_dir}/behavior/model_iterate_accs_fold{n_fold}.pkl", mkdir=True)
fname.add("out_atvs", "{data_dir}/behavior/output_activations.pkl", mkdir=True)
fname.add("pcoder_reps", "{data_dir}/behavior/pcoder_reps_conds.pkl", mkdir=True)

# MEG (processed, shared) and model-brain comparison
fname.add("mri_subjects_dir", "{data_dir}/mris/")
fname.add("sp", "ico4")
fname.add("fsaverage_src", "{mri_subjects_dir}/fsaverage/fsaverage-{sp}-src.fif")
fname.add("meg_tc_sti", "{data_dir}/meg_tcs/meg_tc_{roi}_sti.nc", mkdir=True)
fname.add("meg_tc_sti_ver", "{data_dir}/meg_tc/{roi}_ds{fs}Hz_sti_ver.nc", mkdir=True)
fname.add("rsa_tc", "{data_dir}/rsa_tcs/rsa_tcs_parcel{roi}.nc", mkdir=True)
fname.add("ridge_tc_sti", "{data_dir}/ridge_tcs/ridge_tc_{roi}_perm{perm}.nc", mkdir=True)

# participant-level MEG data (private)
fname.add("private_data_dir", os.environ.get("YOU2026_PRIVATE_DATA", "nonexisting"))
fname.add("private_mri_subjects_dir", "{private_data_dir}/mri_subjects/")
fname.add("subjects_dir", "{private_data_dir}/subjects/")
fname.add("inv", "{subjects_dir}/{subject}-{sp}-inv.fif")
fname.add("epo_con", "{subjects_dir}/{subject}-{condition}-epo.fif")
fname.add("epo", "{subjects_dir}/{subject}-epo.fif")

# figures
fname.add("fig_acc_fold", "{figures_dir}/behav/iterate_accs_fold{n_fold}.pdf", mkdir=True)
fname.add("fig_pair_out", "{figures_dir}/behav/pair_out_activations.pdf", mkdir=True)
fname.add("fig_model_rdms", "{figures_dir}/rsa/model_pcoders_rdms.pdf", mkdir=True)
fname.add("fig_meg_rdms", "{figures_dir}/rsa/meg_{roi}_rdms.pdf", mkdir=True)
fname.add("fig_rsa_whole", "{figures_dir}/rsa/rsa_whole_brain_pcoder{p}", mkdir=True)
fname.add("fig_ridge_tc", "{figures_dir}/ridge/ridge_{roi}.pdf", mkdir=True)
