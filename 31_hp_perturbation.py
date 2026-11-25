########################################################################
# Hyperparameter perturbation of the predictive-coding model (S4, S11 Figs).
#
# The distribution was fixed, together with its random seed (20261116), before it
# was evaluated. Every optimized (median) hyperparameter is multiplied by a factor
# drawn uniformly from [1 - m, 1 + m], m = 0.2, 0.4, 0.6 (a perturbation of up to
# +-20%, +-40%, +-60%; 100 sets per magnitude). For every PCoder, gamma, beta and mu
# are each multiplied by their own factor, clipped at 0 and renormalized to sum to 1;
# alpha is multiplied by its own factor. Multiplication keeps every coefficient that
# was optimized to zero at zero, so the analysis tests the sensitivity to the
# optimized values, not to the structure of the model (PCoder 2 stays memory-only,
# PCoder 3 receives no feedback). References: the median hyperparameters ("median")
# and the five fold-specific sets of the RL2/RL3 cross-validation.
#
# For every set and all 540 stimuli, the network is run for t = 0..28 and the script
# stores
#   correct    top-1 correctness per stimulus and timestep
#   eps        condition means of the per-stimulus prediction error of each PCoder
#   E_below0   fraction of stimuli (per condition) whose global prediction error
#              E(t) = sum_i n_i eps_i(t) is below E(0)
#   E_cond     condition means of E(t)
#   condmeans  condition means of the PCoder 3 representations (ReLU) at t = 0, 25, 28,
#              for the RSA in 32_hp_perturbation_rsa.py
#   diverged   whether any representation or error became non-finite
# The field "rho" holds the magnitude m of each set.
#
# The sets can be split over array tasks; each task saves a part file:
#   python 31_hp_perturbation.py --task 0 --n_tasks 6
#   -> {data_dir}/behavior/hp_perturbation/part{task}.npz
########################################################################
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds

import predify.modules.base as pbase
from config import device, event_id, fname, k, max_timestep
from utility import get_pmodel, set_hyperparams, transform

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", type=int, default=0)
p.add_argument("--n_tasks", type=int, default=1)
p.add_argument("--batchsize", type=int, default=180)
p.add_argument("--max_sets", type=int, default=None, help="quick test only")
args = p.parse_args()
args.n_steps = max_timestep

SEED = 20261116
SCALE_M = (0.2, 0.4, 0.6)
N_PER_SCALE = 100
RSA_STEPS = [0, 25, max_timestep - 1]
N_UNITS = np.array([25088, 4096, 4096])
sti_types = list(event_id.keys())
OUT_DIR = Path(f"{fname.data_dir}/behavior/hp_perturbation{'_test' if args.max_sets else ''}")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- the prespecified hyperparameter sets ------------------------------------
def fold_hps():
    return [np.asarray(torch.load(fname.hps_ckpt_RL23(n_fold=n), weights_only=False)["hps"], float)
            for n in range(k)]


def scale(v, m, rng):
    """Multiplicative perturbation with magnitude m."""
    v = v.copy()
    for o in (0, 4, 8):                                            # PCoder 1/2/3: [g, b, m, a]
        gbm = np.clip(v[o:o + 3] * (1 + rng.uniform(-m, m, 3)), 0, None)
        v[o:o + 3] = gbm / gbm.sum() if gbm.sum() > 0 else v[o:o + 3]
        v[o + 3] = max(0.0, v[o + 3] * (1 + rng.uniform(-m, m)))
    return v


def scale_sets():
    folds = fold_hps()
    median = np.median(folds, axis=0).round(3)
    sets = [("reference", "median", median, 0.0)]
    sets += [("reference", f"fold {f}", h, np.nan) for f, h in enumerate(folds)]
    rng = np.random.default_rng(SEED)
    for m in SCALE_M:
        for n in range(N_PER_SCALE):
            sets.append((f"scale{int(round(100 * m))}", f"m {m:.1f} #{n}", scale(median, m, rng), m))
    return sets


def to_hp_dicts(v):
    return [{"ffm": v[o], "fbm": v[o + 1], "erm": v[o + 3]} for o in (0, 4, 8)]


# ---- per-stimulus prediction error of each PCoder ----------------------------
def make_forward(orig):
    def forward(self, ff, fb, target, build_graph=False, ffm=None, fbm=None, erm=None):
        out = orig(self, ff, fb, target, build_graph, ffm, fbm, erm)
        self.eps_ps = ((self.prd - target.detach()) ** 2).flatten(1).mean(1)
        return out
    return forward


pbase.PCoderN.forward = make_forward(pbase.PCoderN.forward)


def load_stimuli():
    ds = (wds.WebDataset(f"{fname.dataset_dir}/stimuli.tar", shardshuffle=False)
          .decode("pil").map_dict(png=transform).to_tuple("__key__", "png", "cls", "json"))
    keys, imgs, labels, conds = [], [], [], []
    for key, img, label, meta in ds:
        keys.append(key)
        imgs.append(img)
        labels.append(label)
        conds.append(sti_types.index(meta["type"]))
    return keys, torch.stack(imgs), torch.tensor(labels), np.array(conds)


def evaluate(model, images, labels, cond):
    T, relu = args.n_steps, nn.ReLU()
    onehot = torch.tensor(np.eye(4)[cond].T, dtype=torch.float64, device=device)  # (4, N)
    counts = onehot.sum(1)
    correct = np.zeros((T, len(labels)), bool)
    eps_all = torch.zeros(T, len(labels), 3, dtype=torch.float64, device=device)
    condmeans = torch.zeros(len(RSA_STEPS), 4, 1000, dtype=torch.float64, device=device)
    finite = True
    for start in range(0, len(labels), args.batchsize):
        sl = slice(start, start + args.batchsize)
        x, y = images[sl].to(device), labels[sl].to(device)
        for t in range(T):
            with torch.no_grad():
                out = model(x) if t == 0 else model(None)
            correct[t, sl] = (out.argmax(1) == y).cpu().numpy()
            eps_all[t, sl] = torch.stack([getattr(model, f"pcoder{i}").eps_ps for i in (1, 2, 3)], 1).double()
            finite &= bool(torch.isfinite(out).all())
            if t in RSA_STEPS:
                condmeans[RSA_STEPS.index(t)] += onehot[:, sl] @ relu(model.pcoder3.rep.detach()).double()
    finite &= bool(torch.isfinite(eps_all).all())
    E = (eps_all * torch.tensor(N_UNITS, dtype=torch.float64, device=device)).sum(-1)  # (T, N)
    return {
        "correct": correct,
        "eps": ((eps_all.permute(0, 2, 1) @ onehot.T) / counts).cpu().numpy(),      # (T, 3, 4)
        "E_cond": ((E @ onehot.T) / counts).cpu().numpy(),                          # (T, 4)
        "E_below0": ((((E < E[:1]).double()) @ onehot.T) / counts).cpu().numpy(),   # (T, 4)
        "condmeans": (condmeans / counts[None, :, None]).cpu().numpy(),             # (3, 4, 1000)
        "diverged": not finite,
    }


sets = scale_sets()
mine = list(range(len(sets)))[args.task::args.n_tasks]
if args.max_sets:
    mine = mine[:args.max_sets]
print(f"task {args.task}/{args.n_tasks}: {len(mine)} of {len(sets)} sets", flush=True)

keys, images, labels, cond = load_stimuli()
model = get_pmodel(pnet_path=fname.pcoder_ckpt, backbone_path=fname.ff_ckpt,
                   hyperparams=to_hp_dicts(sets[0][2])).to(device).eval()
results = []
for n in mine:
    level, name, v, _ = sets[n]
    set_hyperparams(model, to_hp_dicts(v))
    res = evaluate(model, images, labels, cond)
    results.append(res)
    gain = [res["correct"][min(25, args.n_steps - 1), cond == c].mean() - res["correct"][0, cond == c].mean()
            for c in range(4)]
    print(f"set {n:3d} {name:12s} diverged={res['diverged']}  gain t25 (RW, RL1-3) = "
          + " ".join(f"{g:+.3f}" for g in gain), flush=True)

np.savez_compressed(
    OUT_DIR / f"part{args.task}.npz",
    set_index=np.array(mine), level=np.array([sets[n][0] for n in mine]),
    name=np.array([sets[n][1] for n in mine]), hps=np.array([sets[n][2] for n in mine]),
    rho=np.array([sets[n][3] for n in mine], float),
    keys=np.array(keys), condition=cond, label=labels.numpy(), rsa_steps=np.array(RSA_STEPS),
    conditions=np.array(sti_types),
    **{key: np.array([r[key] for r in results]) for key in results[0]})
print("saved ->", OUT_DIR / f"part{args.task}.npz")
