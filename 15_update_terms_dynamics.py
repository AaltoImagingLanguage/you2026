########################################################################
# Contribution of each PCoder update term over recurrent timesteps and the
# prediction-error trajectory (S1 Fig). For each PCoder and recurrent timestep we
# decompose the update
#   m_i(t+1) = mu*m_i(t) + gamma*ff + beta*fb - alpha*error_scale*grad
# into its four terms (feed-forward, feedback, memory, error-correction), record
# the L2-norm share of each, and record the reconstruction/prediction error
# eps_i(t). Uses the full PC model with the optimized median hyperparameters.
# The S1 Fig outputs (term shares, batch-mean errors) cover t = 0..max_timestep-1.
#
# Global prediction error and convergence: with --n_steps N the network
# is run for N timesteps (t = 0..N-1) and, per stimulus, layer and timestep, the
# script additionally records
#   eps        prediction error of the PCoder, MSE(W_b m_i(t), m_{i-1}(t))
#              (the quantity whose gradient the error-correction term follows)
#   eps_fixed  the same prediction compared with the t = 0 target m_{i-1}(0)
#              (for PCoder 1 the target, the backbone input, does not change)
#   drift      ||m_i(t) - m_i(0)|| / ||m_i(0)||
#   step       ||m_i(t) - m_i(t-1)|| / ||m_i(t-1)||  (0 at t = 0)
#   grad       ||d eps_i / d m_i(t)||
# together with the top-1 prediction and its correctness at every timestep, and
# the condition means of the PCoder representations (ReLU on PCoder 3, as in
# 16_model_rdms.py and 20_controls_representations.py) at every timestep, for the
# RSA in 30_dynamics_rsa.py (S1 and S10 Figs).
# Inputs: ../data/images_dataset/stimuli.tar, original model, median RL2/RL3
#         hyperparameters (../data/model_ckps/hps/vgg16_v1_best_hps_RL23_fold{0-4}.pth)
# Outputs: ../data/behavior/term_contributions_{hps}.pkl (term norms, prediction errors),
#          ../data/behavior/dynamics_long_{hps}.npz (per-stimulus dynamics),
#          figures/behav/Fig_S1_term_contributions.{png,pdf,tif} (S1 Fig)
# The global (whole-hierarchy) prediction error is computed from eps in
# 30_dynamics_rsa.py: E(t) = sum_i n_i eps_i(t), the summed squared error of all
# three PCoders (n_i = number of predicted units: 25088, 4096, 4096).
#
#   python 15_update_terms_dynamics.py --hps RL23 --batchsize 64   # S1 Fig + dynamics outputs
#   python 15_update_terms_dynamics.py --plot_only                   # S1 Fig only
########################################################################
import argparse
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
import matplotlib.pyplot as plt

import predify.modules.base as pbase
from utility import get_pmodel, transform
from config import fname, event_id, device, k, max_timestep, fb_timestep

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--hps", type=str, default="RL23", choices=["RL1", "RL23"])
p.add_argument("--batchsize", type=int, default=64)
p.add_argument("--n_steps", type=int, default=max_timestep,
               help="timesteps to run (t = 0..n_steps-1); the per-stimulus dynamics "
                    "outputs are saved for all of them")
p.add_argument("--max_batches", type=int, default=None, help="smoke test only")
p.add_argument("--plot_only", action="store_true")
args, _ = p.parse_known_args()
assert args.n_steps >= max_timestep, "n_steps must cover the S1 Fig timesteps"

sti_types = list(event_id.keys())
TERMS = ["feed-forward", "feedback", "memory", "error"]
TERM_COLORS = {"feed-forward": "#0072B2", "feedback": "#D55E00",
               "memory": "#999999", "error": "#009E73"}
OUT = f"{fname.data_dir}/behavior/term_contributions_{args.hps}.pkl"
OUT_DYN = f"{fname.data_dir}/behavior/dynamics_long_{args.hps}.npz"
if args.max_batches:  # smoke tests never overwrite the real outputs
    OUT = OUT.replace(".pkl", "_smoke.pkl")
    OUT_DYN = OUT_DYN.replace(".npz", "_smoke.npz")
METRICS = ["eps", "eps_fixed", "drift", "step", "grad"]

NPC = 3
# accumulators over (pcoder, timestep, term) and (pcoder, timestep) for prediction error
_norm_sum = np.zeros((NPC, max_timestep, len(TERMS)))
_perr_sum = np.zeros((NPC, max_timestep))
_count = np.zeros((NPC, max_timestep))
_batch_metrics = [[] for _ in range(NPC)]   # per PCoder: list over t of (batch, metric)


def _mean_l2(t):
    """mean over batch of the per-sample L2 norm of a tensor."""
    return float(t.reshape(t.shape[0], -1).norm(dim=1).mean().item())


def _norm(x):
    return x.reshape(x.shape[0], -1).norm(dim=1)


def make_recording_forward(orig):
    def recording(self, ff, fb, target, build_graph=False, ffm=None, fbm=None, erm=None):
        first = self.rep is None
        if first:
            self._t = 0
        else:
            self._t = getattr(self, "_t", -1) + 1
            prev_rep, prev_grd = self.rep, self.grd
            prd_numel = self.prd.numel()
            csq = float(self.C_sqrt)
        out = orig(self, ff, fb, target, build_graph, ffm, fbm, erm)
        pi, t = self._pc_idx, self._t
        if t < max_timestep:
            ti = {name: i for i, name in enumerate(TERMS)}
            if first:
                # rep = ff : the representation is entirely the feed-forward drive
                _norm_sum[pi, t, ti["feed-forward"]] += _mean_l2(ff)
            else:
                g, b, a = float(ffm), float(fbm) if fbm is not None else 0.0, float(erm)
                _norm_sum[pi, t, ti["feed-forward"]] += _mean_l2(g * ff)
                if self.has_feedback and fb is not None:
                    _norm_sum[pi, t, ti["feedback"]] += _mean_l2(b * fb)
                    mem = (1 - g - b) * prev_rep
                else:
                    mem = (1 - g) * prev_rep
                _norm_sum[pi, t, ti["memory"]] += _mean_l2(mem)
                _norm_sum[pi, t, ti["error"]] += _mean_l2(a * (prd_numel / csq) * prev_grd)
            _perr_sum[pi, t] += float(self.prediction_error)
            _count[pi, t] += 1

        # per-stimulus quantities at every timestep
        rep, tgt = self.rep.detach(), target.detach()
        if first:
            self._rep0, self._tgt0, self._prev = rep.clone(), tgt.clone(), rep.clone()
        eps = ((self.prd - tgt) ** 2).flatten(1).mean(1)
        eps_fixed = ((self.prd - self._tgt0) ** 2).flatten(1).mean(1)
        drift = _norm(rep - self._rep0) / _norm(self._rep0)
        step = _norm(rep - self._prev) / _norm(self._prev)
        _batch_metrics[pi].append(
            torch.stack([eps, eps_fixed, drift, step, _norm(self.grd)], 1).cpu().numpy())
        self._prev = rep.clone()
        return out
    return recording


# ---- run ---------------------------------------------------------------------
if not args.plot_only:
    cv = [torch.load(fname.hps_ckpt_RL23(n_fold=n) if args.hps == "RL23" else fname.hps_ckpt(n_fold=n),
                     weights_only=False)["hps"] for n in range(k)]
    med = np.median(cv, axis=0).round(3)
    hps = [{"ffm": med[0], "fbm": med[1], "erm": med[3]},
           {"ffm": med[4], "fbm": med[5], "erm": med[7]},
           {"ffm": med[8], "fbm": med[9], "erm": med[11]}]
    model = get_pmodel(pnet_path=fname.pcoder_ckpt, backbone_path=fname.ff_ckpt,
                       hyperparams=hps).to(device)
    model.eval()
    for i in range(model.number_of_pcoders):
        getattr(model, f"pcoder{i+1}")._pc_idx = i

    pbase.PCoderN.forward = make_recording_forward(pbase.PCoderN.forward)   # instrument

    val_wrdset = (wds.WebDataset(f"{fname.dataset_dir}/stimuli.tar", shardshuffle=False)
                  .decode("pil").map_dict(png=transform).to_tuple("__key__", "png", "cls", "json"))
    val_loader = torch.utils.data.DataLoader(val_wrdset, batch_size=args.batchsize,
                                             shuffle=False, num_workers=1, pin_memory=True)
    relu = nn.ReLU(inplace=False)
    T = args.n_steps
    keys, conds, labels, metrics, preds = [], [], [], [], []
    cm_sums, cm_counts = {}, np.zeros(4)
    for bi, (bkeys, inputs, blabels, meta) in enumerate(val_loader):
        if args.max_batches and bi >= args.max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        cond = np.array([sti_types.index(c) for c in meta["type"]])
        onehot = np.eye(4)[cond].T                                   # (4, batch)
        for lst in _batch_metrics:
            lst.clear()
        bpred = []
        for t in range(T):
            with torch.no_grad():
                out = model(inputs) if t == 0 else model(None)
            bpred.append(out.argmax(1).cpu().numpy())
            for p_ in range(1, NPC + 1):
                rep = getattr(model, f"pcoder{p_}").rep.detach()
                rep = (relu(rep) if p_ == 3 else rep).double().cpu().numpy()
                key = f"pcoder{p_}"
                if key not in cm_sums:
                    cm_sums[key] = np.zeros((4, T, rep.shape[1]))
                cm_sums[key][:, t] += onehot @ rep
        cm_counts += np.bincount(cond, minlength=4)
        # (T, batch, pcoder, metric)
        metrics.append(np.stack([np.stack(_batch_metrics[p_], 0) for p_ in range(NPC)], 2))
        preds.append(np.stack(bpred, 0))
        keys += list(bkeys)
        conds.append(cond)
        labels.append(blabels.numpy())
        print(f"batch {bi} done", flush=True)

    cnt = np.maximum(_count[..., None], 1)
    term_norm = _norm_sum / cnt                                   # (npc, t, 4) mean norm
    share = term_norm / np.maximum(term_norm.sum(-1, keepdims=True), 1e-12)
    perr = _perr_sum / np.maximum(_count, 1)                      # (npc, t)
    payload = {"term_norm": term_norm, "share": share, "perr": perr, "hps": args.hps}
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)
    print("saved ->", OUT)

    label = np.concatenate(labels)
    pred = np.concatenate(preds, 1)                               # (T, N)
    np.savez_compressed(
        OUT_DYN, keys=np.array(keys), condition=np.concatenate(conds), label=label,
        metrics=np.concatenate(metrics, 1).astype(np.float32),     # (T, N, pcoder, metric)
        metric_names=np.array(METRICS), pred=pred, correct=pred == label[None],
        n_units=np.array([25088, 4096, 4096]), hps=med, conditions=np.array(sti_types),
        **{f"condmeans_{p_}": (s / cm_counts[:, None, None]).astype(np.float32)
           for p_, s in cm_sums.items()})
    print("saved ->", OUT_DYN)
else:
    payload = pickle.load(open(OUT, "rb"))
    term_norm, share, perr = payload["term_norm"], payload["share"], payload["perr"]

# ---- plot: term shares (top) + prediction-error trajectory (bottom) ----------
steps = np.arange(max_timestep)
# A (rows 1-2): update-term shares and prediction error of each PCoder.
# B, C (row 3): global prediction error E(t) = sum of the three PCoders' MSEs
# (the feedback-weight training objective), and the PCoder 3 error against the
# feedforward (t = 0) state of PCoder 2 (eps_fixed, solid) vs its current state
# (dashed; same curve as A, bottom right); mean and 95% CI
# over the 540 stimuli, from dynamics_long_{hps}.npz.
from scipy import stats
fig = plt.figure(figsize=(12, 9.4))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.15], hspace=0.45, wspace=0.9)
axes = np.array([[fig.add_subplot(gs[r, 2 * c:2 * c + 2]) for c in range(NPC)] for r in range(2)])
for pi in range(NPC):
    ax = axes[0, pi]
    for tname in TERMS:
        ti = TERMS.index(tname)
        label = {"feed-forward": "feedforward drive", "feedback": "feedback prediction",
                 "memory": "memory", "error": "prediction-error correction"}[tname]
        ax.plot(steps, share[pi, :, ti], color=TERM_COLORS[tname], lw=1.8, label=label)
    # ax.axvline(fb_timestep, color="grey", ls="--", lw=1)
    ax.set_title(f"PCoder {pi+1}")
    if pi == 0:
        ax.set_ylabel("update-term share")
    if pi == NPC - 1:
        ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2 = axes[1, pi]
    ax2.plot(steps, perr[pi], color="k", lw=1.8)
    # ax2.axvline(fb_timestep, color="grey", ls="--", lw=1)
    ax2.set_xlabel("Timesteps")
    if pi == 0:
        ax2.set_ylabel("prediction error")
dyn = np.load(OUT_DYN)
dm = dyn["metrics"][:max_timestep].astype(float)
names = list(dyn["metric_names"])
eps, fixed = dm[..., names.index("eps")], dm[..., 2, names.index("eps_fixed")]
N = eps.shape[1]
def mean_ci(y):
    return y.mean(1), stats.t.ppf(0.975, N - 1) * y.std(1, ddof=1) / np.sqrt(N)
axB = fig.add_subplot(gs[2, 0:3])
axC = fig.add_subplot(gs[2, 3:6])
mu, h = mean_ci(eps.sum(-1))
axB.fill_between(steps, mu - h, mu + h, color="k", alpha=0.15, lw=0)
axB.plot(steps, mu, color="k", lw=1.8)
axB.axhline(mu[0], color="k", lw=0.6, ls=":")
axB.set_ylabel("global prediction error E")
for y, ls, lab in [(fixed, "-", "feedforward state of PCoder 2 (t = 0)"),
                   (eps[..., 2], "--", "current state of PCoder 2 (as in A)")]:
    mu, h = mean_ci(y)
    axC.fill_between(steps, mu - h, mu + h, color="k", alpha=0.15, lw=0)
    axC.plot(steps, mu, color="k", lw=1.8, ls=ls, label=lab)
axC.set_ylim(0, 1.1)
axC.set_ylabel("PCoder 3 prediction error")
axC.legend(title="Target of PCoder 3", fontsize=8, title_fontsize=8, frameon=False, loc="upper left")
for ax_ in (axB, axC):
    ax_.axvline(fb_timestep, color="grey", ls="--", lw=0.8)
    ax_.set_xlabel("Timesteps")
for ax_, lab in [(axes[0, 0], "A"), (axB, "B"), (axC, "C")]:
    ax_.text(-0.2 if ax_ is axes[0, 0] else -0.1, 1.05, lab, transform=ax_.transAxes,
             fontsize=14, fontweight="bold")
for ax in list(axes.flat) + [axB, axC]:
    ax.spines[["right", "top"]].set_visible(False)
# fig.suptitle(f"Per-term contribution and prediction-error trajectory ({args.hps})", y=1.0)
base = f"{fname.figures_dir}/behav/Fig_S1_term_contributions"
if args.max_batches:
    base += "_smoke"
Path(base).parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200, **kw)
print("saved ->", f"{base}.[png|pdf|tif]")
