#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config parameters
"""
from filename_templates import FileNames
import getpass
import pickle
import pandas as pd

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# Feedback parameters
max_timestep =29 # max time steps to consider in the recurrent feedback
fb_timestep = 25# time step to selct as stable feedback connection state
fb_colors = ["lightblue", "lightcoral"]
temperature = 1

k=5  # number of folds for cross-validation for hyperparameter tuning of the predictive coding model
seed=0 # random seed for fold assignment

SAME_PARAM = False  # to use the same parameters for all pcoders or not
FF_START = True  # to start from feedforward initialization

# time course parameters
time_step = 2  # 20 ms
time_interval = 0.1  # 10 ms
# Time window parameters
time_windows = [[0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1.1]]

# source reconstruction parameters
snr_epoch = 3.0
snr_sti = 1.0
pick_ori = "normal"  # 'normal' for source space time course

# metrics
metric_rsa = "spearman"  # spearman, pearson, kendalltaua, regression
metric_rdm = "correlation"  # euclidean, correlation, cosine


# liear regression parameters
n_splits = 10
nc = 0  # Number of components for PCA, 0 means no PCA

ctn_tps = 3  # number of continuous time points to consider in the ridge regression for whole-brain mapping

epochs_hps = 5000
num_workers = 8
n = 540
cmap = mpl.cm.magma
category_colors = [cmap(i) for i in np.linspace(0, 0.8, 4)]

# Folder you have downloaded the OSF public data package to: https://osf.io/yzqtw
data_dir = "./data"

# Folder to place the figures in
figures_dir = "./figures"

# If you also have access to the private data, set the path here
private_data_dir = None

# These users have access to the private data.
user = getpass.getuser()  # username of the user running the scripts
if user == "youj2" or user == "vanvlm1":
    private_data_dir = "/m/nbe/scratch/flexwordrec/"


# parcellation
parc = "aparc.a2009s_custom_gyrus_sulcus_800mm2"

# ROIs for the analysis and the corresponding index from the parcellation
rois = {"ST": 65, "vOT": 40}

vOT_id = 40


f_down_sampling = 100

event_id = {
    "RW": 1,
    "RL1PW": 2,
    "RL2PW": 3,
    "RL3PW": 4,
}

# subject IDs
subjects = [
    "sub-01",
    "sub-02",
    "sub-03",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-09",
    "sub-11",
    "sub-12",
    "sub-13",
    "sub-15",
    "sub-16",
    "sub-17",
    "sub-18",
    "sub-19",
    "sub-20",
    "sub-21",
    "sub-22",
    "sub-24",
    "sub-25",
    "sub-26",
    "sub-27",
]
SUBJECT = "fsaverage"
baseline_window = (-0.2, 0)  # Proper baseline window
time_len = 1.1

cmaps3 = [
    (0, 0, 0),
    (128 / 255, 0 / 255, 128 / 255),
    # (128/255, 0/255, 128/255),
    (0.994738, 0.62435, 0.427397),
]


model_name = "vgg16"  # vgg16, cornet_z
type_hp = "Separate"
version = "v1"


# %% Filenames for various things


fname = FileNames()

# dataset and model paths

fname.add("data_dir", "data")  # whcere the data is downloadeddata_dirm
fname.add("dataset_dir", "{data_dir}/images_dataset/")

fname.add("cv_folds", "{data_dir}/cv_fold_assignments.json")

fname.add("exper_data_dir", "{data_dir}/dataset/")
fname.add("word2idx_dir", "{exper_data_dir}/word2idx.pkl")
fname.add("stimuli_dir", "{exper_data_dir}/stimuli.csv")
fname.add("stimuli_con_dir", "{exper_data_dir}/rep_stimuli.csv")


fname.add("fonts_dir", "{data_dir}/fonts/")
fname.add("log_dir", "./tmp_train_feedbacks/")
fname.add("log_hps", "{log_dir}/hps/runs_train_hps_{n_step}ts_{sti_type}")
fname.add("net_dir", "{data_dir}/model_ckps/")
fname.add("bb_dir", "{net_dir}/ff_backbone/")
fname.add("pnet_dir", "{net_dir}/fb_pnet/")
fname.add("hps_dir", "{net_dir}/hps/")
fname.add("ff_ckpt", "{bb_dir}/vgg16_checkpoint_best.pth.tar")
fname.add("pcoder_ckpt", "{pnet_dir}/p_vgg16_Separate_v1_best_pc")



fname.add("hps_ckpt", "{hps_dir}/vgg16_v1_best_hps_fold{n_fold}.pth") # best hyperparameters
# behavioral data
fname.add("stimuli_dir", "{data_dir}/behavirior/")
fname.add("accs", "{data_dir}/behavirior/model_iterate_accs_fold{n_fold}.pkl", mkdir=True)
fname.add("out_atvs", "{data_dir}/behavirior/output_activations.pkl", mkdir=True)
fname.add("pcoder_reps", "{data_dir}/behavirior/pcoder_reps_conds.pkl", mkdir=True)


# MRI data

fname.add("mri_subjects_dir", "{data_dir}/mris/")


fname.add("sp", "ico4")  # add this so we can use it in the filenames below

fname.add("src", "{mri_subjects_dir}/{subject}/{subject}-{sp}-src.fif")


fname.add("fsaverage_src", "{mri_subjects_dir}/fsaverage/fsaverage-{sp}-src.fif")

# save processed data
fname.add("meg_tc_con", "{data_dir}/meg_tcs/meg_tc_{roi}_con.nc", mkdir=True)
fname.add("meg_tc_sti", "{data_dir}/meg_tcs/meg_tc_{roi}_sti.nc", mkdir=True)

# condition-wise RSA time courses
fname.add("rsa_tc", "{data_dir}/rsa_tcs/rsa_tcs_parcel{roi}.nc", mkdir=True)

# stimulu-wise ridge regression time courses
fname.add(
    "ridge_tc_sti", "{data_dir}/ridge_tcs/ridge_tc_{roi}_perm{perm}.nc", mkdir=True
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fname.add("meg_tc_sti_ver", "{data_dir}/meg_tc/{roi}_ds{fs}Hz_sti_ver.nc", mkdir=True)
fname.add(
    "meg_tc_sti1", "{data_dir}/tc_meg/{roi}_ds{fs}Hz_sti1.nc", mkdir=True
)  # vertex-wise
fname.add(
    "corr_tc_sti",
    "{data_dir}/corr_tc/{roi}_sti_ts{time_step}_ti{time_interval}_pca{nc}.nc",
    mkdir=True,
)
private_data_dir = "/m/nbe/scratch/flexwordrec/"
fname.add(
    "private_data_dir", "nonexisting" if private_data_dir is None else private_data_dir
)
fname.add("private_mri_subjects_dir", "{private_data_dir}/mri_subjects/")
fname.add("subjects_dir", "{private_data_dir}/subjects/")
fname.add("inv", "{subjects_dir}/{subject}-{sp}-inv.fif")
fname.add("epo_con", "{subjects_dir}/{subject}-{condition}-epo.fif")
fname.add("epo", "{subjects_dir}/{subject}-epo.fif")


# Figures
figures_dir = "./figures"
fname.add("figures_dir", figures_dir)  #  where the figures are saved
fname.add(
    "fig_model_rdms",
    "{figures_dir}/rsa/model_pcoders_rdms.pdf",
    mkdir=True,
)
fname.add(
    "fig_meg_rdms",
    "{figures_dir}/rsa/meg_{roi}_rdms.pdf",
    mkdir=True,
)

fname.add(
    "fig_rsa_whole", "{figures_dir}/rsa/rsa_whole_brain_pcoder{p}.pdf", mkdir=True
)
fname.add("fig_rsa_roi", "{figures_dir}/rsa/rsa_parcel{roi}_tcs.pdf", mkdir=True)


fname.add("fig_rsa_tc", "{figures_dir}/rsa/rsa_{roi}.pdf", mkdir=True)
fname.add("fig_ridge_tc", "{figures_dir}/ridge/ridge_{roi}.pdf", mkdir=True)
fname.add(
    "fig_pair_out",
    "{figures_dir}/behav/pair_out_activations.pdf",
    mkdir=True,
)
fname.add(
    "fig_acc_fold",
    "{figures_dir}/behav/iterate_accs_fold{n_fold}.pdf",
    # "{figures_dir}/behav/iterate_accs_fold{n_fold}.pgf",
    mkdir=True,
)
fname.add(
    "fig_acc_mean",
    "{figures_dir}/behav/iterate_accs_mean.pdf",
    # "{figures_dir}/behav/iterate_accs_mean.pgf",
    mkdir=True,
)
