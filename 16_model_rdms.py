########################################################################
# Condition-wise representations of the PCoders and model RDMs (S5 Fig C).
# With --compute, the predictive-coding model (original model, median hyperparameters
# of the five folds) is run on all experimental stimuli for t = 0..28 and the
# representations of PCoders 1-3 are stored per stimulus and condition (ReLU applied
# to PCoder 3). The RDMs are computed on the condition means (4 x 4; correlation
# distance) without feedback (t = 0) and with feedback (t = 25).
#
# Inputs: ../data/images_dataset/stimuli.tar, original model,
#         ../data/model_ckps/hps/vgg16_v1_best_hps_RL23_fold{0-4}.pth (--hps RL23)
# Outputs: ../data/behavior/pcoder_reps_conds_RL23.pkl (--hps RL23; input of the RSA
#          scripts 22, 23, 33, ...), ../data/dataset/rep_stimuli.csv (stimulus order of
#          the representations, used to align the MEG trials in 17_meg_trial_patterns.py),
#          figures/rsa/model_pcoders_rdms.pdf (S5 Fig C)
#
#   python 16_model_rdms.py --hps RL23 --compute 1   # representations (GPU) + RDMs
#   python 16_model_rdms.py --hps RL23               # RDMs from the saved representations
########################################################################
import os
import sys
import glob
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import webdataset as wds

from utility import plot_rdms_model, get_pmodel, transform
from config import fname, event_id, max_timestep, fb_timestep,k

device = torch.device("cuda:0")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import argparse

parser = argparse.ArgumentParser(description="model representations and RDMs")
parser.add_argument(
    "--batchsize",
    type=int,
    default=64,
    help="batch size for computing the model representations",
)
parser.add_argument(
    "--metric",
    type=str,
    default="correlation",
    help="RDM distance metric: cosine, correlation, euclidean",
)
parser.add_argument(
    "--compute",
    type=bool,
    default=False,
    help="whether to compute the model representations or load from saved files",
)
parser.add_argument(
    "--hps",
    type=str,
    default="RL23",
    choices=["RL1", "RL23"],
    help="hyperparameters: RL1-optimized or RL2+RL3-optimized (RL23, reported)",
)
args, _ = parser.parse_known_args()


sti_types = list(event_id.keys())
# RL23 representations are stored in a separate file (suffix _RL23)
REPS_PATH = (str(fname.pcoder_reps).replace(".pkl", "_RL23.pkl")
             if args.hps == "RL23" else fname.pcoder_reps)


def get_model_reps(net, max_timestep,  features="rep"):

    net.eval()
    rep_dict = {f"pcoder{p+1}": {} for p in range(num_pcoders)}
    for p in range(num_pcoders):
        name = f"pcoder{p+1}"
        stimuli_all = {sti_type: [] for sti_type in sti_types}
        for inputs, json in val_loader:

            stimulus_types = json["type"]
            stimuli_list = json["word"]

            inputs = inputs.to(device, non_blocking=True)

            time_steps_reps = []
            for t in range(max_timestep):
                if t == 0:
                    with torch.no_grad():
                        _ = net(inputs)

                else:
                    with torch.no_grad():
                        _ = net(None)

                pcoder_curr = getattr(net, name)
                
                if p != num_pcoders - 1:
                    time_steps_reps.append(
                        getattr(pcoder_curr, features).cpu().numpy()
                    )
                else:
                    time_steps_reps.append(
                        (
                            nn.ReLU(inplace=False)(getattr(pcoder_curr, features))
                            .cpu()
                            .numpy()
                        )
                    )
            time_steps_reps = np.array(time_steps_reps)  # (timesteps, batch, features)
            time_steps_reps = np.transpose(
                time_steps_reps, (1, 0, 2)
            )  # (batch, timesteps, features)
            for time_step_rep, sti_type, stimulus in zip(
                time_steps_reps, stimulus_types, stimuli_list
            ):
                if sti_type not in rep_dict[name]:
                    rep_dict[name][sti_type] = []
                rep_dict[name][sti_type].append(time_step_rep)
                # also store stimuli
                stimuli_all[sti_type].append(stimulus)

    return rep_dict, stimuli_all


tstart = datetime.now()
if args.compute:
    hps_cvs=[]
    for n_fold in range(k):
        hps_path = (fname.hps_ckpt_RL23(n_fold=n_fold) if args.hps == "RL23"
                    else fname.hps_ckpt(n_fold=n_fold))
        hps_data = torch.load(hps_path, weights_only=False)
        hps = hps_data["hps"]
        hps_cvs.append(hps)
    hps = np.median(hps_cvs, axis=0).round(3)
    print(f"using {args.hps} median hyperparameters:", hps)
    assert hps[:3].sum()==1, "first coder ffm, fbm, erm should sum to 1"
    assert hps[4:7].sum()==1, "second coder ffm, fbm, erm should sum to 1"
    assert hps[8:11].sum()==1, "third coder ffm, fbm, erm should sum to 1"
    
    print(hps)
    hps = [
        {"ffm": hps[0], "fbm": hps[1], "erm": hps[3]},
        {"ffm": hps[4], "fbm": hps[5], "erm": hps[7]},
        {"ffm": hps[8], "fbm": hps[9], "erm": hps[11]},
    ]

    backbone_path = fname.ff_ckpt
    pnet_path = fname.pcoder_ckpt

    model = get_pmodel(
        pnet_path=pnet_path,
        backbone_path=backbone_path,
        hyperparams=hps,
    ).to(device)
    num_pcoders = model.number_of_pcoders
    val_wrdset = (
        wds.WebDataset(glob.glob(f"{fname.dataset_dir}/stimuli.tar"))
        .decode("pil")
        .map_dict(png=transform)
        .to_tuple("png", "json")
    )
    val_loader = torch.utils.data.DataLoader(
        val_wrdset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    print("getting rep for model...")
    print(max_timestep,)
    rep_dict, stimuli_all = get_model_reps(model, max_timestep,)

    with open(REPS_PATH, "wb") as f:
        pickle.dump(rep_dict, f)
    print("saved reps ->", REPS_PATH)

    pd.DataFrame(stimuli_all).to_csv(
        fname.stimuli_con_dir, index=False
    )  # for further alignment with MEG trials

    print("done")


else:
    with open(REPS_PATH, "rb") as f:
        rep_dict = pickle.load(f)

import mne_rsa

metric = args.metric  # cosine, correlation, euclidean
rdms_all = []
for p in rep_dict:
    rdms = []
    names = []
    for sti_type in sti_types:
        rep_dict[p][sti_type] = np.mean(
            rep_dict[p][sti_type], 0
        )  # average across trials
    data_p = np.array([rep_dict[p][sti] for sti in sti_types])

    for it in [0,fb_timestep]:  # two states: w/o feedback (t = 0), w/ feedback (t = 25)

        reps_array = data_p[:, it, :]
        rdm = mne_rsa.compute_rdm(reps_array, metric=metric)
        rdms.extend([rdm])
        names.extend(["w/ feedback" if it else "w/o feedback"])
    rdms_all.append(rdms)
# Plot the RDM
fig = plot_rdms_model(
    rdms_all,
    condition_labels=[sti[:3] for sti in sti_types],
    names=names,
    plot_size=1.5,
    colorbar_label=f"{metric} distance",
    vmax=0.32,  # as in S5 Fig C; PCoder 3 w/o RW-RL3 (0.330) and RL1-RL3 (0.345) saturate
)
plt.savefig(
    fname.fig_model_rdms,
    bbox_inches="tight",
)
plt.show()
