#########################################################################
# Training of the soft-label VGG16 backbones of the training seeds (5, 15, 25, 35),
# numerically
# equivalent to 02_train_backbone.py (--target soft), on the webdataset shards
# of 01_make_image_datasets.py (fname.dataset_dir: train*.tar, val*.tar).
#
# Two changes, neither of which alters the training protocol:
#  1. Soft targets are computed once as a 1000 x 1000 table (row = target word) and
#     indexed per batch, instead of recomputing 100 x 1000 Levenshtein similarities
#     per batch on the CPU (identical values; utility.compute_soft_target).
#  2. The frozen part of VGG16_FT (features[:24]: conv1_1-conv4_3 + pool4, the 20
#     frozen parameter tensors, ImageNet weights) is deterministic: no batch norm, no
#     random augmentation (CenterCrop 224 only), weights identical for every seed. Its
#     output (512 x 14 x 14) is computed once for all training and validation images
#     ("cache" stage, float16, with each image's tar key and label) and shared by all
#     seeds; training then runs only the
#     trainable layers (features[24:], avgpool, classifier) on the cached features.
# Unchanged: model and initialization (VGG16_FT, classifier initialized after seeding),
# loss (cross-entropy against the soft target, temperature 1), Adam (lr 1e-4, weight
# decay 5e-4), batch size 100, dropout, a new random order of the training images in
# every epoch, validation top-1 after every epoch, early stopping after 10 epochs
# without improvement, at most 300 epochs. The checkpoint is the FULL VGG16_FT state
# dict in the format of 02_train_backbone.py (epoch, state_dict, best_prec1, optimizer, args), saved to
# the same path (vgg16_checkpoint_best_soft{tag}.pth.tar), so 05_, 06_, 07_ and 12_ load it
# unchanged. The only numerical difference is float16 storage of the cached pool4
# features (relative rounding ~5e-4), checked by the "verify" stage.
#
# Stages:
#   python 10_train_backbone_seeds.py --stage cache                 # once (GPU, ~30-60 min)
#   python 10_train_backbone_seeds.py --stage verify                # once: original backbone,
#        full network from images vs trainable part on cached features (validation images)
#   python 10_train_backbone_seeds.py --stage train --seed 5 --tag _seed5
#########################################################################
import argparse
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import glob
import tarfile

import webdataset as wds

from config import fname
from utility import VGG16_FT, compute_soft_target, transform

p = argparse.ArgumentParser()
p.add_argument("--stage", required=True, choices=["cache", "verify", "train"])
p.add_argument("--cache_dir", default=f"{fname.data_dir}/backbone_cache/images_dataset_pool4")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--tag", default="", help="checkpoint suffix, e.g. _seed5")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight_decay", type=float, default=5e-4)
p.add_argument("--batch_size", type=int, default=100)
p.add_argument("--max_epochs", type=int, default=300)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--temp", type=float, default=1.0)
p.add_argument("--num_workers", type=int, default=10)
p.add_argument("--out_path", default=None, help="override the checkpoint path (testing)")
p.add_argument("--no_optimizer", action="store_true", help="do not store the optimizer state (saves ~0.5 GB)")
p.add_argument("--verify_n", type=int, default=5000, help="validation images used by --stage verify")
FLAGS = p.parse_args()

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
CACHE = Path(FLAGS.cache_dir)
SPLIT_FROZEN = 24                       # features[:24] frozen (conv1_1 ... conv4_3, pool4)
SHAPE = (512, 14, 14)
words = list(pickle.load(open(fname.word2idx_dir, "rb")).keys())


def frozen_part(net):
    return net.features[:SPLIT_FROZEN]


def head_forward(net, x):
    """Trainable part of VGG16_FT.forward, from the pool4 output."""
    x = net.features[SPLIT_FROZEN:](x)
    x = net.avgpool(x)
    return net.classifier(torch.flatten(x, 1))


def check_frozen_split(net):
    frozen = [q.requires_grad for q in frozen_part(net).parameters()]
    rest = [q.requires_grad for q in net.features[SPLIT_FROZEN:].parameters()]
    assert len(frozen) == 20 and not any(frozen) and all(rest), "frozen split differs from VGG16_FT"


def shards(split):
    return sorted(glob.glob(f"{fname.dataset_dir}/{split}*.tar"))


def n_images(split):
    """Number of .png samples in the shards of a split (read from the tar headers)."""
    n = 0
    for f in shards(split):
        with tarfile.open(f) as t:
            n += sum(m.name.endswith(".png") for m in t)
    return n


def tar_split(split, keys=None):
    """(key, image, label) of a split as loaded by 02_train_backbone.py (labels = word2idx order of the soft
    targets); keys: keep only these samples. No shuffling."""
    ds = wds.WebDataset(shards(split), shardshuffle=False)
    if keys is not None:
        ds = ds.select(lambda smp: smp["__key__"] in keys)
    return ds.decode("pil").map_dict(png=transform).to_tuple("__key__", "png", "cls")


def load_cache(split):
    meta = np.load(CACHE / f"{split}_labels.npy")
    feats = np.memmap(CACHE / f"{split}_pool4.f16", dtype=np.float16, mode="r", shape=(len(meta),) + SHAPE)
    return feats, meta


# ----------------------------------------------------------------------- cache
if FLAGS.stage == "cache":
    CACHE.mkdir(parents=True, exist_ok=True)
    net = VGG16_FT().to(device).eval()
    check_frozen_split(net)
    for split in ("train", "val"):
        n = n_images(split)
        out = np.memmap(CACHE / f"{split}_pool4.f16.part", dtype=np.float16, mode="w+", shape=(n,) + SHAPE)
        labels = np.zeros(n, dtype=np.int16)
        keys = []
        # workers read different shards, so the order is interleaved; each row keeps its key and label
        loader = data.DataLoader(tar_split(split), batch_size=200, num_workers=FLAGS.num_workers, pin_memory=True)
        i, t0 = 0, time.time()
        with torch.no_grad():
            for k, x, y in loader:
                out[i:i + len(y)] = frozen_part(net)(x.to(device, non_blocking=True)).half().cpu().numpy()
                labels[i:i + len(y)] = y.numpy()
                keys += list(k)
                i += len(y)
                if (i // 200) % 100 == 0:
                    print(f"{split}: {i}/{n} images, {time.time() - t0:.0f} s", flush=True)
        assert i == n, f"{split}: {i} images read, {n} in the tar headers"
        assert len(set(keys)) == n, "duplicate tar keys"
        out.flush()
        del out
        os.replace(CACHE / f"{split}_pool4.f16.part", CACHE / f"{split}_pool4.f16")
        np.save(CACHE / f"{split}_labels.npy", labels)
        (CACHE / f"{split}_files.txt").write_text("\n".join(keys))
        print(f"cached {split}: {n} images -> {CACHE}/{split}_pool4.f16", flush=True)
    raise SystemExit

# ----------------------------------------------------------------------- verify
if FLAGS.stage == "verify":
    net = VGG16_FT().to(device)
    ck = torch.load(fname.ff_ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    feats, labels = load_cache("val")
    all_keys = (CACHE / "val_files.txt").read_text().split("\n")
    row_of = {k: r for r, k in enumerate(all_keys)}
    rng = np.random.default_rng(0)
    pick = set(rng.choice(all_keys, min(FLAGS.verify_n, len(all_keys)), replace=False))
    full, cached, rows = [], [], []
    loader = data.DataLoader(tar_split("val", pick), batch_size=100, num_workers=FLAGS.num_workers)
    with torch.no_grad():
        for k, x, y in loader:
            j = np.array([row_of[q] for q in k])
            assert np.array_equal(labels[j], y.numpy()), "cached labels differ from the tars"
            full.append(net(x.to(device)).float().cpu())
            cached.append(head_forward(net, torch.from_numpy(feats[j].astype(np.float32)).to(device)).cpu())
            rows.append(j)
    full, cached, idx = torch.cat(full), torch.cat(cached), np.concatenate(rows)
    y = torch.from_numpy(labels[idx].astype(np.int64))
    print(f"verify on {len(idx)} validation images (original backbone {fname.ff_ckpt}):")
    print(f"  top-1 full network {100 * (full.argmax(1) == y).float().mean():.2f}%, "
          f"from cache {100 * (cached.argmax(1) == y).float().mean():.2f}%")
    print(f"  identical top-1 predictions: {100 * (full.argmax(1) == cached.argmax(1)).float().mean():.3f}%")
    print(f"  max |logit difference| {(full - cached).abs().max():.2e} (logit range "
          f"{full.min():.1f} to {full.max():.1f})")
    raise SystemExit

# ----------------------------------------------------------------------- train
print("loading cached features into memory ...", flush=True)
t0 = time.time()
tr_x, tr_y = load_cache("train")
va_x, va_y = load_cache("val")
tr_x = torch.from_numpy(np.array(tr_x))     # float16, in RAM (~40 GB)
va_x = torch.from_numpy(np.array(va_x))     # (~10 GB)
tr_y = torch.from_numpy(tr_y.astype(np.int64))
va_y = torch.from_numpy(va_y.astype(np.int64))
print(f"loaded {len(tr_y)} training and {len(va_y)} validation images in {time.time() - t0:.0f} s", flush=True)
soft_table = torch.stack([compute_soft_target(w, words, FLAGS.temp) for w in words]).to(device)  # (1000, 1000)
criterion = nn.CrossEntropyLoss()


def train(net, seed, out_path, lr, weight_decay, max_epochs, patience, label=""):
    """Train the trainable part of net on the cached features; keep the best validation epoch in out_path.
    Returns the best state dict."""
    optimizer = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed if seed is not None else 0)
    best_prec1, patience_counter, best_state = -float("inf"), 0, None
    B = FLAGS.batch_size
    for epoch in range(max_epochs):
        t0 = time.time()
        net.train()                                   # dropout active, as in 02_train_backbone.py
        order = torch.randperm(len(tr_y), generator=gen)
        loss_sum = correct = n_seen = 0
        for i in range(0, len(order), B):
            j = order[i:i + B]
            x = tr_x[j].to(device, non_blocking=True).float()
            y = tr_y[j].to(device, non_blocking=True)
            out = head_forward(net, x)
            loss = criterion(out, soft_table[y])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(j)
            correct += (out.argmax(1) == y).sum().item()
            n_seen += len(j)
            if (i // B) % 500 == 0:
                print(f"{label}Epoch: [{epoch}][{i // B:04d}/{len(order) // B:04d}] Loss {loss_sum / n_seen:.4f} "
                      f"Prec@1 {correct / n_seen:.3f}", flush=True)
        net.eval()
        with torch.no_grad():
            val_correct = sum((head_forward(net, va_x[i:i + B].to(device).float()).argmax(1)
                               == va_y[i:i + B].to(device)).sum().item() for i in range(0, len(va_y), B))
        prec1 = val_correct / len(va_y)
        print(f"{label} * Prec@1 {prec1:.3f}  (epoch {epoch}, {time.time() - t0:.0f} s)", flush=True)
        if prec1 > best_prec1:
            best_prec1, patience_counter = prec1, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            ck = {"epoch": epoch + 1, "state_dict": best_state, "best_prec1": best_prec1,
                  "args": argparse.Namespace(**vars(FLAGS), target="soft", data="tar_cached", seed_run=seed,
                                             lr_run=lr, wd_run=weight_decay)}
            if not FLAGS.no_optimizer:
                ck["optimizer"] = optimizer.state_dict()
            torch.save(ck, out_path)
            print(f"{label}===============save best model at epoch: ", epoch, "->", out_path, flush=True)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"{label}patience limit reached, stopping training {epoch}")
            break
    return best_state


def ckpt_path(tag):
    return FLAGS.out_path or str(fname.ff_ckpt).replace(".pth.tar", f"_soft{tag}.pth.tar")


def new_net(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    net = VGG16_FT().to(device)            # classifier initialized after seeding, as in 02_train_backbone.py
    check_frozen_split(net)
    return net


net = new_net(FLAGS.seed)
train(net, FLAGS.seed, ckpt_path(FLAGS.tag), FLAGS.lr, FLAGS.weight_decay, FLAGS.max_epochs, FLAGS.patience)
