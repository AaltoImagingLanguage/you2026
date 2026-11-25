########################################################################
# Condition-mean representations of the control models, for their RSA
# (21_rsa_controls.py; S12 Fig). The controls are those of S3 Fig
# (14_controls_ablations_behavior.py), built in the same way (median RL2/RL3
# hyperparameters, backbone and PCoders of the original model):
#   full PC   - the full predictive-coding model (recomputed with the same code;
#               21_rsa_controls.py asserts that its RDMs equal those of the main analysis)
#   mem_only  - memory-only recurrence (gamma=beta=alpha=0, mu=1)
#   beta0     - prediction-error correction without feedback (beta=0)
#   alpha0    - feedback without prediction-error correction (alpha=0)
#   rand_fb   - random (untrained) feedback weights, five seeds
# The parameter-matched RNN/GRU representations are read out by
# 09_rnn_gru_representations.py and loaded in 21_rsa_controls.py.
#
# For each model, PCoder and timestep t = 0..28, the representations (ReLU on
# PCoder 3, as in 16_model_rdms.py) are averaged over the stimuli of each condition.
# Input: ../data/images_dataset/stimuli.tar, original model,
#        ../data/model_ckps/hps/vgg16_v1_best_hps_RL23_fold{0-4}.pth
# Output: ../data/behavior/controls_condmeans_RL23.pkl
#   {"timesteps": (T,), "models": {name: [ {pcoder: (4, T, features)} per seed ]}}
#
#   python 20_controls_representations.py   # (GPU)
########################################################################
import glob
import pickle

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds

from config import device, event_id, fname, k, max_timestep
from utility import get_pmodel, patch_pcoders, transform

SEEDS = [11, 23, 37, 51, 71]  # as in 14_controls_ablations_behavior.py
OUT = f"{fname.data_dir}/behavior/controls_condmeans_RL23.pkl"
sti_types = list(event_id.keys())


def load_median():
    cv = [torch.load(fname.hps_ckpt_RL23(n_fold=n), weights_only=False)["hps"] for n in range(k)]
    return np.median(cv, axis=0).round(3)


def control_sets(med):
    """As in 14_controls_ablations_behavior.py: (name, hyperparameter vector, random feedback)."""
    def patch(assignments):
        v = np.array(med, float).copy()
        for i, val in assignments.items():
            v[i] = val
        return v
    B, M, A = [1, 5, 9], [2, 6, 10], [3, 7, 11]
    return [
        ("full PC", np.array(med, float), False),
        ("mem_only", patch({**{i: 0.0 for i in B + A}, **{i: 1.0 for i in M}}), False),
        ("beta0", patch({i: 0.0 for i in B}), False),
        ("alpha0", patch({i: 0.0 for i in A}), False),
        ("rand_fb", np.array(med, float), True),
    ]


def to_hp_dicts(v):
    return [{"ffm": 1.0 - v[o + 2] - v[o + 1], "fbm": v[o + 1], "erm": v[o + 3]}
            for o in (0, 4, 8)]


def build(hp_vec, random_seed=None):
    model = get_pmodel(pnet_path=fname.pcoder_ckpt, backbone_path=fname.ff_ckpt,
                       hyperparams=to_hp_dicts(hp_vec)).to(device)
    if random_seed is not None:  # replace only the generative (feedback) modules
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        for n in range(1, model.number_of_pcoders + 1):
            pcoder = getattr(model, f"pcoder{n}")
            pcoder.pmodule.reset_parameters()
            pcoder.C_sqrt = torch.tensor(-1.0, device=device)
        patch_pcoders(model)
    return model.eval()


def condition_means(model, loader):
    relu = nn.ReLU(inplace=False)
    sums, counts = {}, np.zeros(4)
    with torch.no_grad():
        for inputs, meta in loader:
            inputs = inputs.to(device, non_blocking=True)
            cond = np.array([sti_types.index(t) for t in meta["type"]])
            steps = {p: [] for p in range(1, 4)}
            for t in range(max_timestep):
                _ = model(inputs) if t == 0 else model(None)
                for p in steps:
                    rep = getattr(model, f"pcoder{p}").rep
                    steps[p].append((relu(rep) if p == 3 else rep).cpu().numpy())
            for p, reps in steps.items():
                reps = np.stack(reps, 1).reshape(len(cond), max_timestep, -1)  # (batch, T, feat)
                onehot = np.eye(4)[cond].T                                      # (4, batch)
                sums[p] = sums.get(p, 0) + np.einsum("cb,btf->ctf", onehot, reps)
            counts += np.bincount(cond, minlength=4)
    return {f"pcoder{p}": s / counts[:, None, None] for p, s in sums.items()}


val_wrdset = (wds.WebDataset(glob.glob(f"{fname.dataset_dir}/stimuli.tar"))
              .decode("pil").map_dict(png=transform).to_tuple("png", "json"))
loader = torch.utils.data.DataLoader(val_wrdset, batch_size=64, shuffle=False,
                                     num_workers=1, pin_memory=True)

models = {}
for name, hp, randomize in control_sets(load_median()):
    runs = []
    for seed in (SEEDS if randomize else [None]):
        print(f"[{name}{'' if seed is None else f', seed={seed}'}]", flush=True)
        runs.append(condition_means(build(hp, seed), loader))
    models[name] = runs

with open(OUT, "wb") as f:
    pickle.dump({"timesteps": np.arange(max_timestep), "models": models}, f)
print("saved ->", OUT)
