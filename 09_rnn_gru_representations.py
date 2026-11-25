########################################################################
# Condition-wise representations of the RNN/GRU controls over recurrent timesteps, in
# the format of the predictive-coding model's pcoder_reps_conds.pkl, for the RSA of the
# controls (21_rsa_controls.py; S12 Fig).
#
# Stored at TIMESTEPS = [0, 5, 10, 15, 20, 25] for the three recurrent sites
# (rep1/rep2/rep3; rep3 corresponds to the PCoder 3 site), per stimulus and condition:
#   rep_dict["rnn1"|"rnn2"|"rnn3"][RW|RL1PW|RL2PW|RL3PW] = (n_stim, 6, features)
# t = 0 is the feedforward pass; t > 0 are the states of the learned recurrence.
#
# Inputs: ../data/images_dataset/stimuli.tar,
#         ../data/model_ckps/rnn_baseline/reconrec_{cell}_bn128_best.pth
# Outputs: ../data/behavior/reconrec_{cell}_bn128_reps_conds.pkl,
#          ../data/behavior/reconrec_{cell}_bn128_rep_stimuli.csv (stimulus order)
#
#   python 09_rnn_gru_representations.py --cell rnn        # (GPU)
#   python 09_rnn_gru_representations.py --cell gru
########################################################################
import argparse
import glob
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import webdataset as wds

from config import device, event_id, fname
from utility import get_model, transform
from pnet.rvgg16_baseline import ClassifierRecurrentVGG16Baseline

TIMESTEPS = [0, 5, 10, 15, 20, 25]

p = argparse.ArgumentParser()
p.add_argument("--cell", type=str, default="rnn", choices=["rnn", "gru"])
p.add_argument("--bottleneck", type=int, default=128)
p.add_argument("--batchsize", type=int, default=64)
args, _ = p.parse_known_args()

sti_types = list(event_id.keys())
bn = args.bottleneck if args.bottleneck and args.bottleneck > 0 else None
tag = f"reconrec_{args.cell}" + (f"_bn{bn}" if bn else "")
ckpt_path = f"{fname.net_dir}/rnn_baseline/{tag}_best.pth"
reps_path = f"{fname.data_dir}/behavior/{tag}_reps_conds.pkl"
stim_csv = f"{fname.data_dir}/behavior/{tag}_rep_stimuli.csv"


def load_net():
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    a = ckpt["args"]
    net = ClassifierRecurrentVGG16Baseline(
        backbone=get_model(model="vgg16", trained_root=fname.ff_ckpt),
        cell_type=a["cell"], use_topdown=bool(a["use_topdown"]),
        input_adapter=bool(a["input_adapter"]), residual_mix=bool(a["residual_mix"]),
        init_mix=a["init_mix"], bottleneck=(a["bottleneck"] if a["bottleneck"] > 0 else None),
        freeze_backbone=True).to(device)
    miss, unexp = net.load_state_dict(ckpt["state_dict"], strict=False)
    assert all(k.startswith("backbone.") for k in miss) and not unexp
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']}, val recon {ckpt['loss']:.4f})")
    return net.eval()


net = load_net()
val_wrdset = (wds.WebDataset(glob.glob(f"{fname.dataset_dir}/stimuli.tar"))
              .decode("pil").map_dict(png=transform).to_tuple("png", "json"))
val_loader = torch.utils.data.DataLoader(val_wrdset, batch_size=args.batchsize,
                                         shuffle=False, num_workers=1, pin_memory=True)

REP_NAMES = ["rnn1", "rnn2", "rnn3"]
rep_dict = {n: {s: [] for s in sti_types} for n in REP_NAMES}
stimuli_all = {s: [] for s in sti_types}
relu = nn.ReLU(inplace=False)

with torch.no_grad():
    for inputs, meta in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        types_, words = meta["type"], meta["word"]
        # collect rep1/rep2/rep3 at each requested timestep
        per_step = {n: [] for n in REP_NAMES}   # each -> list over timesteps of (batch, feat)
        for t in range(max(TIMESTEPS) + 1):
            _ = net(inputs) if t == 0 else net(None)
            if t in TIMESTEPS:
                per_step["rnn1"].append(net.rep1.cpu().numpy())
                per_step["rnn2"].append(net.rep2.cpu().numpy())
                per_step["rnn3"].append(relu(net.rep3).cpu().numpy())  # ReLU on the output, as for the PCoder readout
        # (n_ts, batch, feat) -> (batch, n_ts, feat)
        per_step = {n: np.transpose(np.array(v), (1, 0, 2)) for n, v in per_step.items()}
        for n in REP_NAMES:
            for rep, sti in zip(per_step[n], types_):
                rep_dict[n][sti].append(rep)
        for sti, w in zip(types_, words):
            stimuli_all[sti].append(w)

# number of stimuli per condition
n_per = {s: len(rep_dict["rnn1"][s]) for s in sti_types}
print("stimuli per condition:", n_per,
      "| stored timesteps:", TIMESTEPS,
      "| shapes:", {n: np.array(rep_dict[n][sti_types[0]]).shape for n in REP_NAMES})

with open(reps_path, "wb") as f:
    pickle.dump({"rep_dict": rep_dict, "timesteps": TIMESTEPS}, f)
pd.DataFrame(stimuli_all).to_csv(stim_csv, index=False)
print("saved reps ->", reps_path)
print("saved stimuli order ->", stim_csv)
