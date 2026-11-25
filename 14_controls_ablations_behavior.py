########################################################################
# Accuracy of the control models and ablations over recurrent timesteps t = 0..28
# (S3 Fig). Ablations of the trained predictive-coding model (median RL2/RL3
# hyperparameters, no retraining):
#   mem_only  - memory-only recurrence (gamma=beta=alpha=0, mu=1)
#   beta0     - prediction-error correction without explicit feedback mixing
#               (beta=0, alpha>0; the generative map still defines the error)
#   alpha0    - feedback without prediction-error correction (alpha=0, beta>0)
#   rand_fb   - random (untrained) feedback weights, i.e. recurrence without trained
#               feedback; averaged over five initializations (--random_seeds), band = SD
# The full predictive-coding model is evaluated with the same code; the accuracies of
# the parameter-matched RNN/GRU controls are loaded from 08_eval_rnn_gru_controls.py.
# The feedforward state at t = 0 is the soft-label backbone without predictive coding.
#
# Inputs: ../data/images_dataset/stimuli.tar, original model,
#         ../data/model_ckps/hps/vgg16_v1_best_hps_RL23_fold{0-4}.pth,
#         ../data/behavior/control_reconrec_{rnn,gru}_bn128_accs.pkl
# Outputs: ../data/behavior/controls_RL23.pkl,
#          figures/behav/controls_acc_RL23.{png,pdf,tif} (S3 Fig)
#
#   python 14_controls_ablations_behavior.py --hps RL23
#   python 14_controls_ablations_behavior.py --hps RL23 --controls beta0  # one control only
#   python 14_controls_ablations_behavior.py --hps RL23 --plot_only
########################################################################
import argparse
import glob
import pickle

import numpy as np
import torch
import webdataset as wds
import matplotlib.pyplot as plt

from utility import get_pmodel, patch_pcoders, transform
from config import (
    fname, event_id, device, k, max_timestep, fb_timestep,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--hps", type=str, default="RL23", choices=["RL1", "RL23"])
p.add_argument("--batchsize", type=int, default=64)
p.add_argument("--plot_only", action="store_true")
p.add_argument(
    "--controls",
    nargs="+",
    choices=["full PC", "mem_only", "beta0", "alpha0", "rand_fb"],
    default=None,
    help="recompute only the selected controls and merge them into the existing "
         "pickle; default: recompute all controls",
)
p.add_argument(
    "--random_seeds",
    type=int,
    nargs="+",
    default=[11, 23, 37, 51, 71],
    help="independent random-feedback initializations to average",
)
args, _ = p.parse_known_args()

sti_types = list(event_id.keys())
OUT = f"{fname.data_dir}/behavior/controls_{args.hps}.pkl"
# Full PC and the ablations are evaluated together; RNN/GRU are loaded.
ABLATIONS = ["mem_only", "beta0", "alpha0", "rand_fb"]
MODEL_ORDER = ["full PC", "rnn", "gru"] + ABLATIONS
MODEL_COLORS = {"full PC": "#000000", "rnn": "#555555", "gru": "#999999",
                "mem_only": "#0072B2", "beta0": "#009E73", "alpha0": "#E69F00",
                "rand_fb": "#CC79A7"}
MODEL_LABELS = {
    "full PC": "full predictive-coding model",
    "rnn": "parameter-matched RNN",
    "gru": "parameter-matched GRU",
    "mem_only": "memory only",
    "beta0": "prediction-error correction without feedback",
    "alpha0": "feedback without error correction",
    "rand_fb": "random feedback weights",
}


def load_median():
    cv = [torch.load(fname.hps_ckpt_RL23(n_fold=n) if args.hps == "RL23" else fname.hps_ckpt(n_fold=n),
                     weights_only=False)["hps"] for n in range(k)]
    return np.median(cv, axis=0).round(3)


def control_sets(med):
    def patch(assignments):
        v = np.array(med, float).copy()
        for i, val in assignments.items():
            v[i] = val
        return v
    B, M, A = [1, 5, 9], [2, 6, 10], [3, 7, 11]
    return [
        ("full PC",  np.array(med, float),            False),
        ("mem_only", patch({
            **{i: 0.0 for i in B + A},
            **{i: 1.0 for i in M},
        }),                                             False),
        ("beta0",    patch({i: 0.0 for i in B}),       False),
        ("alpha0",   patch({i: 0.0 for i in A}),       False),
        ("rand_fb",  np.array(med, float),            True),   # reinitialize only generative weights
    ]


def to_hp_dicts(v):
    return [
        {
            "ffm": 1.0 - v[offset + 2] - v[offset + 1],
            "fbm": v[offset + 1],
            "erm": v[offset + 3],
        }
        for offset in (0, 4, 8)
    ]


def effective_hyperparameters(v):
    """Return the values actually used by each PCoder."""
    values = []
    for pcoder, offset in enumerate((0, 4, 8), start=1):
        beta = float(v[offset + 1])
        mu = float(v[offset + 2])
        gamma = 1.0 - mu - beta
        values.append({
            "pcoder": pcoder,
            "gamma": gamma,
            "beta": beta,
            "mu": mu,
            "alpha": float(v[offset + 3]),
        })
    return values


def print_hyperparameters(name, values):
    print(f"\n{name} effective hyperparameters:")
    for hp in values:
        print(
            f"  PCoder {hp['pcoder']}: "
            f"gamma={hp['gamma']:.6f}, "
            f"beta={hp['beta']:.6f}, "
            f"mu={hp['mu']:.6f}, "
            f"alpha={hp['alpha']:.6f}"
        )


def evaluate(hp_vec, randomize_feedback, val_loader, random_seed=None):
    # Always load the trained model first. For the random-feedback control, only
    # the generative modules are then replaced; backbone and all other settings
    # remain identical to the full model.
    model = get_pmodel(pnet_path=fname.pcoder_ckpt, backbone_path=fname.ff_ckpt,
                       hyperparams=to_hp_dicts(hp_vec)).to(device)
    if randomize_feedback:
        if random_seed is None:
            raise ValueError("random_seed is required for random feedback")
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        for n in range(1, model.number_of_pcoders + 1):
            pcoder = getattr(model, f"pcoder{n}")
            pcoder.pmodule.reset_parameters()
            pcoder.C_sqrt = torch.tensor(-1.0, device=device)
        patch_pcoders(model)
    model.eval()
    correct = {c: np.zeros(max_timestep) for c in sti_types}
    total = {c: 0 for c in sti_types}
    for inputs, labels, js in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        types = js["type"]
        for t in range(max_timestep):
            with torch.no_grad():
                out = model(inputs) if t == 0 else model(None)
            corr = (out.argmax(1) == labels).float().cpu().numpy()
            for i, ty in enumerate(types):
                correct[ty][t] += corr[i]
        for ty in types:
            total[ty] += 1
    acc = {c: correct[c] / max(total[c], 1) * 100 for c in sti_types}
    return acc


def load_reference_behav():
    """{model: {cond: (max_timestep,) accuracy %}} for loaded models."""
    ext = {}
    def to_traj(d):
        return {c: np.array([np.mean(d[c][t]) * 100 if (c in d and t in d[c]) else np.nan
                             for t in range(max_timestep)]) for c in sti_types}
    try:
        ext["full PC"] = to_traj(pickle.load(open(str(fname.accs(n_fold=5)), "rb")))
        print("loaded reference behav: full PC")
    except Exception as e:
        print("skip full PC behav:", e)
    for cell in ("rnn", "gru"):
        try:
            a = pickle.load(open(f"{fname.data_dir}/behavior/control_reconrec_{cell}_bn128_accs.pkl", "rb"))
            ext[cell] = to_traj(a)
            print(f"loaded reference behav: {cell}")
        except Exception as e:
            print(f"skip {cell} behav:", e)
    return ext


# ---- run ---------------------------------------------------------------------
if not args.plot_only:
    med = load_median()
    val_wrdset = (wds.WebDataset(glob.glob(f"{fname.dataset_dir}/stimuli.tar"))
                  .decode("pil").map_dict(png=transform).to_tuple("png", "cls", "json"))
    val_loader = torch.utils.data.DataLoader(val_wrdset, batch_size=args.batchsize,
                                             shuffle=False, num_workers=1, pin_memory=True)
    selected = set(args.controls) if args.controls is not None else None
    if selected is not None:
        try:
            with open(OUT, "rb") as f:
                results = pickle.load(f)["results"]
            print("loaded existing results for selective update ->", OUT)
        except FileNotFoundError:
            results = {}
            print("no existing results found; creating a new partial payload")
    else:
        results = {}

    for name, hp, randomize in control_sets(med):
        if selected is not None and name not in selected:
            continue
        hp_values = effective_hyperparameters(hp)
        print_hyperparameters(MODEL_LABELS[name], hp_values)
        seeds = args.random_seeds if randomize else [None]
        runs = []
        for seed in seeds:
            seed_text = f", seed={seed}" if seed is not None else ""
            print(f"\n[{name}{seed_text}]")
            acc = evaluate(
                hp, randomize, val_loader, random_seed=seed
            )
            runs.append(acc)

        # Average random-feedback results over seeds. Preserve the individual
        # seed results in the payload so their variability can also be reported.
        result = {
            "acc": {
                c: np.mean([run[c] for run in runs], axis=0)
                for c in sti_types
            },
            "hyperparameters": hp_values,
        }
        if randomize:
            result["random_seeds"] = list(seeds)
            result["seed_results"] = runs
        results[name] = result
        print(
            f"{name:9s} RW-RL1 gain(t25)="
            f"{result['acc']['RW'][fb_timestep] - result['acc']['RL1PW'][fb_timestep]:+.1f}"
        )
    payload = {"args": vars(args), "results": results}
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)
    print("saved ->", OUT)
else:
    payload = pickle.load(open(OUT, "rb"))
    results = payload["results"]
    for name in MODEL_ORDER:
        if name in results and "hyperparameters" in results[name]:
            print_hyperparameters(
                MODEL_LABELS[name], results[name]["hyperparameters"]
            )

# Combine evaluated controls with the saved RNN/GRU behavioral references.
ref_behav = load_reference_behav()
behav = {**ref_behav, **{nm: results[nm]["acc"] for nm in results}}
present = [m for m in MODEL_ORDER if m in behav]
steps = np.arange(max_timestep)


# ---- behavioral: accuracy over timesteps, one panel per condition ------------
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
for ax, c in zip(axes.flat, sti_types):
    for nm in present:
        if nm not in behav:
            continue
        y = behav[nm][c]
        ax.plot(steps[:len(y)], y, color=MODEL_COLORS[nm], lw=2 if nm == "full PC" else 1.3,
                label=MODEL_LABELS[nm], alpha=1 if nm == "full PC" else 0.85)
        if nm == "rand_fb" and results.get(nm, {}).get("seed_results"):
            seed_y = np.stack(
                [run[c] for run in results[nm]["seed_results"]], axis=0
            )
            seed_sd = seed_y.std(axis=0, ddof=1)
            ax.fill_between(
                steps[:len(y)],
                y - seed_sd,
                y + seed_sd,
                color=MODEL_COLORS[nm],
                alpha=0.15,
                linewidth=0,
            )
    ax.axvline(fb_timestep, color="grey", ls="--", lw=1)  # readout timestep
    ax.set_title(c[:3]); ax.set_ylabel("Accuracy (%)")
    ax.spines[["right", "top"]].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("Timesteps")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.15),
    frameon=False,
    fontsize=10,
    ncol=4,
)
# fig.suptitle("Control models: behavioral accuracy over recurrent timesteps", y=1.0)
fig.tight_layout(rect=(0, 0.20, 1, 1))
b = f"{fname.figures_dir}/behav/controls_acc_{args.hps}"
for ext in ("png", "pdf", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(f"{b}.{ext}", bbox_inches="tight", dpi=200, **kw)
print("saved ->", f"{b}.[png|pdf|tif]")
