# %%
#########################################################################
# Accuracy of the RNN/GRU controls (pnet/rvgg16_baseline.py, trained by
# 07_train_rnn_gru_controls.py) over recurrent timesteps t = 0..28 (S3 Fig), with the
# same stimuli and readout as the predictive-coding model. t = 0 is the feedforward
# pass; the learned recurrence runs for t > 0. The table printed reports t = 0, the
# readout timestep t = 25, the last timestep and the timestep with the highest mean
# pseudoword accuracy. No misspelled words were used for training.
# The architecture is rebuilt from the arguments stored in the checkpoint, so only
# --cell (and the tag flags, if non-default) are needed to locate it.
#
# Inputs: ../data/images_dataset/stimuli.tar,
#         ../data/model_ckps/rnn_baseline/reconrec_{cell}_bn128_best.pth
# Outputs: ../data/behavior/control_reconrec_{cell}_bn128_accs.pkl (per-stimulus top-1
#          correctness per condition and timestep; read by 14_controls_ablations_behavior.py)
#          figures/behav/control_reconrec_{cell}_bn128_trajectory.[tif|pdf]
#
#   python 08_eval_rnn_gru_controls.py --cell rnn --compute   # evaluate and plot
#   python 08_eval_rnn_gru_controls.py --cell rnn             # plot from the saved pkl
#########################################################################
import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import webdataset as wds

from config import category_colors, device, event_id, fb_timestep, fname, max_timestep
from utility import accuracy, get_model, transform
from pnet.rvgg16_baseline import ClassifierRecurrentVGG16Baseline

parser = argparse.ArgumentParser(description="Accuracy of the RNN/GRU controls over timesteps")
parser.add_argument("--cell", type=str, default="gru", choices=["rnn", "gru"])
parser.add_argument("--bottleneck", type=int, default=128)
parser.add_argument("--use_topdown", type=int, default=0)
parser.add_argument("--input_adapter", type=int, default=0)
parser.add_argument("--batchsize", type=int, default=60)
parser.add_argument("--compute", action="store_true")
args, _ = parser.parse_known_args()

sti_types = list(event_id.keys())
bottleneck = args.bottleneck if args.bottleneck and args.bottleneck > 0 else None
# tag identical to the trainer (07_train_rnn_gru_controls.py)
tag = (f"reconrec_{args.cell}" + (f"_bn{bottleneck}" if bottleneck else "")
       + ("_td" if args.use_topdown else "") + ("_ad" if args.input_adapter else ""))
ckpt_path = f"{fname.net_dir}/rnn_baseline/{tag}_best.pth"
out_path = f"{fname.data_dir}/behavior/control_{tag}_accs.pkl"


def load_net():
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    a = ckpt.get("args", {})  # architecture arguments stored in the checkpoint
    bn = a.get("bottleneck", bottleneck)
    bn = bn if bn and bn > 0 else None
    backbone = get_model(model="vgg16", trained_root=fname.ff_ckpt)
    net = ClassifierRecurrentVGG16Baseline(
        backbone=backbone,
        cell_type=a.get("cell", args.cell),
        use_topdown=bool(a.get("use_topdown", args.use_topdown)),
        input_adapter=bool(a.get("input_adapter", args.input_adapter)),
        residual_mix=bool(a.get("residual_mix", 1)),
        init_mix=a.get("init_mix", 0.10),
        bottleneck=bn,
        freeze_backbone=True,
    ).to(device)
    missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=False)
    assert all(k.startswith("backbone.") for k in missing), missing
    assert not unexpected, unexpected
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']}, val recon {ckpt['loss']:.4f})")
    return net.eval()


def make_loader():
    ds = (wds.WebDataset(f"{fname.dataset_dir}/stimuli.tar")
          .decode("pil").map_dict(png=transform)
          .to_tuple("png", "cls", "json"))
    return torch.utils.data.DataLoader(ds, batch_size=args.batchsize,
                                       shuffle=False, num_workers=1, pin_memory=True)


def evaluate(net):
    stim_acc = {sti: {t: [] for t in range(max_timestep)} for sti in sti_types}
    loader = make_loader()
    tstart = time.time()
    for i, (inputs, labels, meta) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        types = meta["type"]
        for t in range(max_timestep):
            with torch.no_grad():
                outputs = net(inputs) if t == 0 else net(None)
            for out, label, sti in zip(outputs, labels, types):
                stim_acc[sti][t].append(accuracy(out[None, :], label[None], topk=(1,))[0])
        print(f"  batch {i} ({time.time()-tstart:.1f}s)")
    return stim_acc


if args.compute:
    net = load_net()
    results = evaluate(net)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"saved -> {out_path}")
else:
    with open(out_path, "rb") as f:
        results = pickle.load(f)


# ------------------------------------------------------------ trajectories
traj = {sti: np.array([100.0 * float(np.mean(results[sti][t]))
                       for t in range(max_timestep)]) for sti in sti_types}
pseudo = np.mean([traj[s] for s in sti_types if s != "RW"], axis=0)
plateau_t = int(np.argmax(pseudo))


def at(sti, t):
    return traj[sti][t]


print("\n" + "=" * 74)
print(f"Recurrent control ({tag}): top-1 accuracy (%)")
print(f"plateau (max mean-pseudoword accuracy) at t={plateau_t}")
print("=" * 74)
print(f"{'cond':<8}{'t=0 (FF)':<12}{f'plateau t={plateau_t}':<16}"
      f"{f't={fb_timestep}':<10}{'final t='+str(max_timestep-1):<12}")
for sti in sti_types:
    print(f"{sti[:3]:<8}{at(sti,0):<12.1f}{at(sti,plateau_t):<16.1f}"
          f"{at(sti,fb_timestep):<10.1f}{at(sti,max_timestep-1):<12.1f}")
print("=" * 74)

# -------------------------------------------------------------- trajectory plot
fig, ax = plt.subplots(figsize=(9, 6))
tt = np.arange(max_timestep)
for i, sti in enumerate(sti_types):
    ax.plot(tt, traj[sti], marker="o", ms=6, lw=2.5, color=category_colors[i], label=sti[:3])
# ax.axvline(fb_timestep, color="grey", ls=":", lw=1, label=f"t={fb_timestep}")
# ax.axvline(plateau_t, color="k", ls="--", lw=2, label=f"plateau t={plateau_t}")
ax.set_xlabel("Timestep", fontsize=18)
ax.set_ylabel("Accuracy (%)", fontsize=18)
ax.set_title(f"{args.cell.upper()} baseline: accuracy over inference", fontsize=18)
ax.tick_params(axis="both", labelsize=15)
ax.legend(frameon=False, fontsize=14, title="Condition", title_fontsize=14)
fig.tight_layout()
base = f"{fname.figures_dir}/behav/control_{tag}_trajectory"
# .tif (300 dpi, LZW compression) and .pdf
fig.savefig(f"{base}.tif", bbox_inches="tight", dpi=300,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.pdf", bbox_inches="tight")
print(f"saved trajectory plot -> {base}.[tif|pdf]")
