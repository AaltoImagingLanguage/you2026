import torch
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import gc
from collections import defaultdict
import json, random

import numpy as np
from utility import load_pnet, accuracy, compute_soft_target, transform
from torch.utils.tensorboard import SummaryWriter
from config import fname, device, epochs_hps, temperature, SAME_PARAM, FF_START, seed, k
import webdataset as wds
from pathlib import Path
import pickle

########################
## GLOBAL CONFIGURATIONS
########################

gc.collect()
torch.cuda.empty_cache()

with open(fname.word2idx_dir, "rb") as file:
    word2idx10k = pickle.load(file)

words1k = list(word2idx10k.keys())


import argparse

parser = argparse.ArgumentParser(description="Training hyper-parameters")
parser.add_argument("--net", type=str, default="vgg16", help="pnet")  # Separate
parser.add_argument("--version", type=str, default="v1", help="pnet version: vx")
parser.add_argument(
    "--hp_type",
    type=str,
    default="Separate",
    help="Separate of Same hyper parameters for each predictive-coding layers",
)
parser.add_argument(
    "--sti_type",
    type=str,
    default="RL1PW",
    help="which stimulus type to tune the hyperparameters on",
)
parser.add_argument(
    "--max_step",
    type=int,
    default=5,
    help="max time step to consider for training and evaluation",
)
parser.add_argument(
    "--val_fold",
    type=int,
    default=0,
    help="which fold to use as validation fold",
)

args, _ = parser.parse_known_args()

MAX_TIMESTEP = args.max_step
sti_type = args.sti_type
type_hp = args.hp_type

TASK_NAME = args.net
version = args.version
# leng will be set after fold split is created (line ~90)
feedforward = False
LOG_DIR = fname.log_hps(n_step=MAX_TIMESTEP, sti_type=sti_type)

hps = None


backbone_path = fname.ff_ckpt
WEIGHT_PATTERN_N = fname.pcoder_ckpt
# time of we run the script
TIME_NOW = datetime.now().isoformat()
loss_function = nn.CrossEntropyLoss()

print("pnet: ", TASK_NAME, version)
########################
########################


def evaluate(
    net,
    epoch,
    dataloader,
    timesteps,
    writer=None,
    tag="Clean",
):
    test_loss = np.zeros((timesteps + 1,))
    correct = np.zeros((timesteps + 1,))
    for images, target, json in dataloader:
        images = images.cuda()
        base = json["base"]

        if type(target) == list:
            target = [t.to(device, non_blocking=True) for t in target]
        else:
            target = target.to(device, non_blocking=True)
        with torch.no_grad():
            for tt in range(timesteps + 1):
                if tt == 0:
                    outputs = net(images)
                else:
                    outputs = net()

                soft_targets = [
                    compute_soft_target(
                        y_word.lower(), words1k, temperature=temperature
                    )
                    for y_word in base
                ]
                soft_targets = torch.stack(soft_targets).to(
                    device
                )  # (batch_size, 1000)
                loss = loss_function(outputs, soft_targets)

                out = accuracy(outputs, target, topk=(1,))
                acc5 = out[0]

                test_loss[tt] += loss.item()
                correct[tt] += acc5

    for tt in range(timesteps + 1):
        test_loss[tt] /= val_leng
        correct[tt] /= val_leng
        print(
            "Test set t = {:02d}: Average loss: {:.4f}, Accuracy: {:.4f}".format(
                tt, test_loss[tt], correct[tt]
            )
        )
        if writer is not None:
            writer.add_scalar(f"{tag}Perf//Epoch#{epoch}", correct[tt], tt)
            writer.add_scalar(f"{tag}Loss//Epoch#{epoch}", test_loss[tt], tt)
    return correct, test_loss


def train(net, epoch, dataloader, timesteps, writer=None):
    for batch_index, (images, target, json) in enumerate(dataloader):

        base = json["base"]
        net.reset()

        if type(target) == list:
            target = [t.to(device, non_blocking=True) for t in target]
        else:
            target = target.to(device, non_blocking=True)
        images = images.cuda()

        ttloss = np.zeros((timesteps + 1))
        optimizer.zero_grad()

        soft_targets = [
            compute_soft_target(y_word.lower(), words1k, temperature=temperature)
            for y_word in base
        ]
        soft_targets = torch.stack(soft_targets).to(device)  # (batch_size, 1000)

        for tt in range(timesteps + 1):
            if tt == 0:
                outputs = net(images)
                loss = loss_function(outputs, soft_targets)

                ttloss[tt] = loss.item()
                acc0 = accuracy(outputs, target, topk=(1,))[0]
            else:
                outputs = net()
                loss1 = loss_function(outputs, soft_targets)
                ttloss[tt] = loss1.item()
                loss += loss1
        acc5 = accuracy(outputs, target, topk=(1,))[0]
        loss.backward()
        optimizer.step()
        net.update_hyperparameters()
        print(
            f"Training Epoch: {epoch} [{batch_index*len(images) + len(images)}/{leng}]\thit0: {acc0}\thit: {acc5}"
        )
        for tt in range(timesteps + 1):
            print(f"\t{ttloss[tt]:0.1f}\t", end="")

        if writer is not None:
            writer.add_scalar(
                f"TrainingLoss/CE",
                loss.item(),
                (epoch - 1) * leng + batch_index,
            )


def _loads_json(maybe_bytes_or_str):
    if isinstance(maybe_bytes_or_str, (bytes, bytearray)):
        maybe_bytes_or_str = maybe_bytes_or_str.decode("utf-8")
    return json.loads(maybe_bytes_or_str)


def _norm_key(k):
    return k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k


def log_hyper_parameters(net, epoch, sumwriter, same_param=True):
    if same_param:
        sumwriter.add_scalar(
            f"HyperparamRaw/feedforward", getattr(net, f"ff_part").item(), epoch
        )
        sumwriter.add_scalar(
            f"HyperparamRaw/feedback", getattr(net, f"fb_part").item(), epoch
        )
        sumwriter.add_scalar(
            f"HyperparamRaw/error", getattr(net, f"errorm").item(), epoch
        )
        sumwriter.add_scalar(
            f"HyperparamRaw/memory", getattr(net, f"mem_part").item(), epoch
        )

        sumwriter.add_scalar(
            f"Hyperparam/feedforward", getattr(net, f"ffm").item(), epoch
        )
        sumwriter.add_scalar(f"Hyperparam/feedback", getattr(net, f"fbm").item(), epoch)
        sumwriter.add_scalar(f"Hyperparam/error", getattr(net, f"erm").item(), epoch)
        sumwriter.add_scalar(
            f"Hyperparam/memory",
            1 - getattr(net, f"ffm").item() - getattr(net, f"fbm").item(),
            epoch,
        )
    else:
        for i in range(1, net.number_of_pcoders + 1):
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_feedforward",
                getattr(net, f"ffm{i}").item(),
                epoch,
            )
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(
                    f"Hyperparam/pcoder{i}_feedback",
                    getattr(net, f"fbm{i}").item(),
                    epoch,
                )
            else:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", 0, epoch)
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_error", getattr(net, f"erm{i}").item(), epoch
            )
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(
                    f"Hyperparam/pcoder{i}_memory",
                    1 - getattr(net, f"ffm{i}").item() - getattr(net, f"fbm{i}").item(),
                    epoch,
                )
            else:
                sumwriter.add_scalar(
                    f"Hyperparam/pcoder{i}_memory",
                    1 - getattr(net, f"ffm{i}").item(),
                    epoch,
                )


# Load or generate fold assignments

tar = f"{fname.dataset_dir}/{sti_type}.tar"
fold_path = fname.cv_folds

if fold_path.exists():
    fold_of_key = json.load(open(fold_path))["fold_of_key"]
else:

    groups = defaultdict(list)

    for key, meta_json in wds.WebDataset(tar).to_tuple("__key__", "json"):
        meta = _loads_json(meta_json)
        groups[int(meta["type_idx"])].append(_norm_key(key))

    rng = random.Random(seed)
    fold_of_key = {}

    for type_idx, keys in groups.items():
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            fold_of_key[key] = i % k

    out = {"k": k, "seed": seed, "fold_of_key": fold_of_key}
    with open(fold_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {fold_path}")

val_fold = args.val_fold

# Create base dataset
base = wds.WebDataset(tar).decode("pil").map_dict(png=transform)


def _fold_for_sample(s):
    return fold_of_key.get(_norm_key(s["__key__"]))


# Split into train and validation using fold assignments
train_wrdset = base.select(
    lambda s: (f := _fold_for_sample(s)) is not None and f != val_fold
).to_tuple("png", "cls", "json")

val_wrdset = base.select(lambda s: (_fold_for_sample(s) == val_fold)).to_tuple(
    "png", "cls", "json"
)

train_loader = torch.utils.data.DataLoader(
    train_wrdset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    val_wrdset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

# Compute training set size from fold counts
# For stratified CV: 4/5 of data in train, 1/5 in validation
# Assuming 135 total samples per sti_type, train_set ≈ 108, val_set ≈ 27
# Count samples in each split by scanning fold_of_key
train_count = sum(1 for f in fold_of_key.values() if f != val_fold)
val_count = sum(1 for f in fold_of_key.values() if f == val_fold)
leng = train_count  # Use training set size for progress reporting
val_leng = val_count  # Use validation set size for progress reporting
print(f"Fold split: train={train_count}, val={val_count}, val_fold={val_fold}")

sumwriter = SummaryWriter(f"{LOG_DIR}/", filename_suffix=f"")
start = datetime.now()


# feedforward for baseline
if feedforward:
    pnet_fw = load_pnet(
        model=TASK_NAME,
        pnet_path=WEIGHT_PATTERN_N,
        backbone_path=backbone_path,
        type_hp=type_hp,
        version=version,
        build_graph=False,
        random_init=(not FF_START),
        ff_multiplier=1.0,
        fb_multiplier=0.0,
        er_multiplier=0.0,
    ).to(device)

    acc = evaluate(
        pnet_fw,
        0,
        val_loader,
        timesteps=10,
        writer=sumwriter,
        tag="FeedForward",
    )
    print("ff acc:", acc)
    del pnet_fw
gc.collect()

# train hps


pnet = load_pnet(
    model=TASK_NAME,
    pnet_path=WEIGHT_PATTERN_N,
    backbone_path=backbone_path,
    type_hp=type_hp,
    version=version,
    build_graph=True,
    random_init=(not FF_START),
    ff_multiplier=0.33,
    fb_multiplier=0.33,
    er_multiplier=0.1,
    hyperparams=hps,
).to(device)
hyperparams = [
    *pnet.get_hyperparameters()
]  # -torch.log((1-x)/x) hyperparams[4*pc1, 4*pc2, 4*pc3, ] 4: ff, fb, mem,er,
if SAME_PARAM:
    optimizer = optim.Adam(
        [
            {"params": hyperparams[:-1], "lr": 0.01},
            {"params": hyperparams[-1:], "lr": 0.0001},
        ],
        weight_decay=0.00001,
    )

else:
    fffbmem_hp = []
    erm_hp = []
    for pc in range(pnet.number_of_pcoders):
        fffbmem_hp.extend(hyperparams[pc * 4 : pc * 4 + 3])
        erm_hp.append(hyperparams[pc * 4 + 3])
    optimizer = optim.Adam(
        [
            {"params": fffbmem_hp, "lr": 0.001},
            {"params": erm_hp, "lr": 0.0001},
        ],  ##################################
        weight_decay=0.00001,
    )


log_hyper_parameters(pnet, 0, sumwriter, same_param=SAME_PARAM)
hps = pnet.get_hyperparameters_values()
print(hps)

acc, loss = evaluate(
    pnet,
    0,
    val_loader,
    timesteps=MAX_TIMESTEP,
    writer=sumwriter,
    tag="Noisy",
)
print(datetime.now() - start)

best_loss = np.sum(loss)  # initial loss with initial hyperparamters

print("Initial loss:", best_loss)
patients = 1000
best_acc = np.mean(acc)  # acc under best loss
best_hyps = 0

# results to save
for epoch in range(1, epochs_hps + 1):
    train(pnet, epoch, train_loader, timesteps=MAX_TIMESTEP, writer=sumwriter)
    print(datetime.now() - start)
    log_hyper_parameters(pnet, epoch, sumwriter, same_param=SAME_PARAM)

    hps = pnet.get_hyperparameters_values()
    print(hps)

    acc, loss = evaluate(
        pnet,
        epoch,
        val_loader,
        timesteps=MAX_TIMESTEP,
        writer=sumwriter,
        tag="Noisy",
    )
    print(datetime.now() - start)

    sum_acc = np.mean(acc)
    is_best = best_acc < sum_acc
    if is_best:
        patients = patients
        best_hyps = hps
        best_acc = sum_acc
        best_loss_list = loss
        torch.save(
            {
                "loss": loss,
                "acc": best_acc,
                "hps": best_hyps,
            },
            fname.hps_ckpt(n_fold=val_fold),
        )
        print("current_acc:", best_acc)
        print("best hyps:", best_hyps)
    else:
        patients -= 1
        print("patients:", patients)

    if patients == 0:
        print("best_acc:", best_acc)
        print("best hyps:", best_hyps)
        print("best hyps:", best_hyps, "DONE")

        break

sumwriter.close()
del pnet
gc.collect()
print(datetime.now() - start)
