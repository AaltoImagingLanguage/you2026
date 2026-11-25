#########################################################################
# Train the parameter-matched recurrent controls (RNN, GRU; S6 Table, S3 Fig, S12 Fig)
# defined in pnet/rvgg16_baseline.py, with the same unsupervised objective as the
# predictive-coding model: each generative head reconstructs the representation of the
# layer below (MSE), averaged over recurrent timesteps t = 0..max_step, on the 1000-word
# training images, with the backbone frozen and gradient clipping.
#
# The models differ from the predictive-coding model only in the update rule: a learned
# RNN/GRU cell (anchored to the feedforward state by a residual mixing gate) instead of
# the error-minimizing predictive-coding update. Defaults: no adapters, no separate
# top-down input, 128-d cell bottleneck, max_step 5, Adam (lr 1e-3, weight decay 5e-4),
# batch 32, at most 200 epochs, early stopping after 10 epochs without improvement.
#
# Inputs: ../data/images_dataset/train*.tar, val*.tar and the backbone
# ../data/model_ckps/ff_backbone/vgg16_checkpoint_best.pth.tar.
# Output: ../data/model_ckps/rnn_baseline/reconrec_{cell}_bn128{out_tag}_best.pth
#
#   python 07_train_rnn_gru_controls.py --cell rnn
#   python 07_train_rnn_gru_controls.py --cell gru
#########################################################################
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.optim as optim
import webdataset as wds
from torch.utils.tensorboard import SummaryWriter

from config import device, fname
from utility import get_model, transform
from pnet.rvgg16_baseline import (ClassifierRecurrentVGG16Baseline,
                                  count_trainable_params,
                                  pcoder_feedback_param_count)

p = argparse.ArgumentParser(description="Train recurrent baseline (reconstruction objective)")
# Defaults: the parameter-matched configuration reported in the paper, a plain
# RNN/GRU cell (no adapters / no separate top-down) with residual anchoring and a
# 128-d cell bottleneck, so total added params (gen heads + small cell) match PC.
p.add_argument("--cell", type=str, default="rnn", choices=["rnn", "gru"])
p.add_argument("--use_topdown", type=int, default=0)
p.add_argument("--input_adapter", type=int, default=0)
p.add_argument("--residual_mix", type=int, default=1)
p.add_argument("--init_mix", type=float, default=0.10)
p.add_argument("--bottleneck", type=int, default=128,
               help="cell bottleneck dim for parameter matching (0 = full-width)")
p.add_argument("--max_step", type=int, default=5,
               help="recurrent timesteps unrolled during training")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--num_workers", type=int, default=8)
p.add_argument("--num_epochs", type=int, default=200)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight_decay", type=float, default=5e-4)
p.add_argument("--grad_clip", type=float, default=1.0, help="max grad-norm (0 disables)")
p.add_argument("--patience", type=int, default=10)
p.add_argument("--random_seed", type=int, default=42)
p.add_argument("--val_batches", type=int, default=None, help="cap val batches for speed")
# Training seeds: seed-specific backbone and checkpoint tag (defaults: original model).
p.add_argument("--backbone", type=str, default="", help="backbone checkpoint (default: fname.ff_ckpt)")
p.add_argument("--out_tag", type=str, default="", help="checkpoint suffix, e.g. _seed5")
args, _ = p.parse_known_args()

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

bottleneck = args.bottleneck if args.bottleneck and args.bottleneck > 0 else None

save_dir = f"{fname.net_dir}/rnn_baseline"
os.makedirs(save_dir, exist_ok=True)
tag = (f"reconrec_{args.cell}" + (f"_bn{bottleneck}" if bottleneck else "")
       + ("_td" if args.use_topdown else "") + ("_ad" if args.input_adapter else "")
       + args.out_tag)
ckpt_path = f"{save_dir}/{tag}_best.pth"
sumwriter = SummaryWriter(os.path.join(fname.log_dir, f"reconrec_baseline_{tag}_{args.lr}"))

# ------------------------------------------------------------------ model
backbone = get_model(model="vgg16", trained_root=args.backbone or fname.ff_ckpt)
net = ClassifierRecurrentVGG16Baseline(
    backbone=backbone,
    cell_type=args.cell,
    use_topdown=bool(args.use_topdown),
    input_adapter=bool(args.input_adapter),
    residual_mix=bool(args.residual_mix),
    init_mix=args.init_mix,
    bottleneck=bottleneck,
    freeze_backbone=True,
).to(device)
print(f"trainable params: {count_trainable_params(net)/1e6:.2f} M   "
      f"(PC feedback pathway: {pcoder_feedback_param_count()/1e6:.2f} M)")

trainable = [pp for pp in net.parameters() if pp.requires_grad]
optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

# ------------------------------------------------------------------ data
def make_loader(pattern, shuffle):
    ds = (wds.WebDataset(sorted(glob.glob(f"{fname.dataset_dir}/{pattern}")),
                         shardshuffle=shuffle)
          .decode("pil").map_dict(png=transform).to_tuple("png"))
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       num_workers=args.num_workers, pin_memory=True)

train_loader = make_loader("train*.tar", shuffle=100)
val_loader = make_loader("val*.tar", shuffle=False)


def recon_loss_over_time(images):
    """Average summed-reconstruction MSE across t=0..max_step."""
    net.reset()
    total = 0.0
    for t in range(args.max_step + 1):
        _ = net(images) if t == 0 else net(None)
        total = total + net.reconstruction_loss()
    return total / (args.max_step + 1)


def train_epoch(epoch):
    net.train()
    tstart = time.time()
    for bidx, (images,) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = recon_loss_over_time(images)
        loss.backward()
        gnorm = (torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                 if args.grad_clip and args.grad_clip > 0 else torch.tensor(float("nan")))
        optimizer.step()
        if bidx % 100 == 0:
            print(f"epoch {epoch} [{bidx*args.batch_size}]  recon {loss.item():.4f}"
                  f"  gnorm {float(gnorm):.2f}  ({time.time()-tstart:.1f}s)")
        sumwriter.add_scalar("train/recon", loss.item(), epoch * 100000 + bidx)


@torch.no_grad()
def validate(epoch):
    net.eval()
    total, n = 0.0, 0
    for bidx, (images,) in enumerate(val_loader):
        if args.val_batches is not None and bidx >= args.val_batches:
            break
        images = images.to(device, non_blocking=True)
        total += recon_loss_over_time(images).item()
        n += 1
    val = total / max(n, 1)
    sumwriter.add_scalar("val/recon", val, epoch)
    print(f"[val] epoch {epoch}  recon {val:.4f}")
    return val


# ------------------------------------------------------------------ train loop
best, bad = float("inf"), 0
for epoch in range(args.num_epochs):
    train_epoch(epoch)
    v = validate(epoch)
    if best - v > 1e-3:
        best, bad = v, 0
        torch.save({"epoch": epoch, "loss": best, "cell": args.cell,
                    "args": vars(args),
                    "state_dict": {k: val for k, val in net.state_dict().items()
                                   if not k.startswith("backbone.")}},
                   ckpt_path)
        print(f"  saved best -> {ckpt_path} (val recon {best:.4f})")
    else:
        bad += 1
        if bad >= args.patience:
            print(f"early stop at epoch {epoch}; best {best:.4f}")
            break
sumwriter.close()
print("done ->", ckpt_path)
