#########################################################################
# Train the feedback (generative) weights of the PCoders of the predictive-coding VGG16
# (pnet/pvgg16v1.py, 04_make_pc_model.py); the feedforward backbone stays fixed.
# Loss: sum over PCoders of the prediction error (MSE between the feedback prediction
# and the representation of the layer below), on the 1000-word training images.
# SGD (no momentum), lr 1e-4, weight decay 5e-4, batch 50, at most 200 epochs; training
# stops after 10 epochs in which the validation loss does not fall more than min_delta
# below the best loss. The checkpoints of the original model record optim_name='Adam'
# in their stored arguments, but their stored optimizer state is SGD; the defaults
# here are the SGD settings.
#
# Inputs: ../data/images_dataset/train*.tar, val*.tar (01_make_image_datasets.py) and the
# backbone ../data/model_ckps/ff_backbone/vgg16_checkpoint_best.pth.tar (--backbone).
# Outputs (../data/model_ckps/fb_pnet/, N = PCoder): untagged runs write
# p_vgg16_Separate_v1_epoch-EE_pcN.pth at every improvement (the released weights of
# the original model are p_vgg16_Separate_v1_best_pcN.pth); tagged runs (--out_tag)
# write p_vgg16_Separate_v1{out_tag}_best_pcN.pth, _last.pth and a _DONE marker.
#
#   python 05_train_feedback_weights.py                       # original model
#   python 05_train_feedback_weights.py --backbone ../data/model_ckps/ff_backbone/vgg16_checkpoint_best_onehot.pth.tar --out_tag _onehot
#        one-hot control (S3 Table)
#   training seeds 5, 15, 25, 35: (--out_tag _seedSEED
#        --random_seed SEED --auto_resume)
#########################################################################
# %%
import torch
import torchvision.transforms as transforms
import webdataset as wds
import glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from config import fname, device
import os
import time
from utility import get_model, fetch_pmodel, transform, patch_pcoders
################################################
#       Global configs
################################################


import argparse

parser = argparse.ArgumentParser(description="Training Feedback Weights")

# General settings
parser.add_argument(
    "--net", type=str, default="vgg16", help="pnet for training"
)  # Separate
parser.add_argument(
    "--version", type=str, default="v1", help="pnet version: vx"
)  # Separate
parser.add_argument(
    "--hp_type",
    type=str,
    default="Separate",
    help="Separate of Same hyper parameters for each predictive-coding layers",
)
parser.add_argument(
    "--random_seed", type=int, default=42, help="Random seed for training"
)
parser.add_argument(
    "--cudnn_deterministic", action="store_true", help="Use CUDNN deterministic mode"
)  # Running the script with --cudnn_deterministic --> true, otherwise false
parser.add_argument(
    "--cudnn_benchmark", action="store_true", help="Use CUDNN benchmark mode"
)
parser.add_argument("--gpu_to_use", type=int, default=0, help="GPU index to use")
parser.add_argument(
    "--backbone", type=str, default=str(fname.ff_ckpt),
    help="backbone checkpoint (default: soft-label ff_ckpt; the _onehot backbone for the "
    "one-hot control)",
)
parser.add_argument(
    "--out_tag", type=str, default="",
    help="suffix of the saved PCoder filenames, e.g. _onehot or _seed5",
)

# TensorBoard
parser.add_argument(
    "--start_epoch", type=int, default=0, help="Epoch from which training starts"
)


# Training
parser.add_argument("--batch_size", type=int, default=50, help="Batch size for training")
parser.add_argument(
    "--num_workers", type=int, default=8, help="Number of workers for data loading"
)
parser.add_argument(
    "--num_epochs", type=int, default=200, help="Number of training epochs"
)
parser.add_argument(
    "--train_samples",
    type=int,
    default=None,
    help="Number of training samples per epoch for WebDataset.with_length",
)
parser.add_argument(
    "--val_samples",
    type=int,
    default=None,
    help="Number of validation samples per epoch for WebDataset.with_length",
)

# Optimization
parser.add_argument("--optim_name", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer name")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
parser.add_argument("--ckpt_every", action="store_true", help="checkpoint every epoch (not used by the script)")
parser.add_argument(
    "--min_delta", type=float, default=0.05,
    help="an epoch counts as an improvement (weights saved, patience reset) only if the "
    "validation loss falls more than min_delta below the best loss so far; training stops "
    "after 10 epochs without improvement. 0.05 for the training seeds "
    "(training seeds): with this value training stops near epoch 170, the "
    "training length of the original model",
)
parser.add_argument(
    "--auto_resume", action="store_true",
    help="(tagged runs only) continue from <base>_last.pth if it exists, including the "
    "best loss and patience counter, so a run split over several jobs equals one run",
)

# Resuming Training
parser.add_argument(
    "--resume_training",
    type=bool,
    default=False,
    help="Resume training from checkpoint",
)
parser.add_argument(
    "--resume_ckpts",
    type=str,
    nargs="+",
    default=[
        f"{fname.pnet_dir}/p_vgg16_Separate_v1_best_pc1.pth",
        f"{fname.pnet_dir}/p_vgg16_Separate_v1_best_pc2.pth",
        f"{fname.pnet_dir}/p_vgg16_Separate_v1_best_pc3.pth",
    ],
    help="Checkpoint file paths",
)  # accept one or more values as a list


args, _ = parser.parse_known_args()


# Setup the training
if args.random_seed:
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


if not os.path.exists(f"{fname.log_dir}"):
    os.makedirs(f"{fname.log_dir}")

# summarywriter
tensorboard_path = os.path.join(
    f"{fname.log_dir}",
    f"{args.net}_{args.version}_{args.lr}",
)
sumwriter = SummaryWriter(tensorboard_path, filename_suffix="")

main_str = ""
for x in vars(args):
    main_str += f"{x:<20}: {getattr(args,x)}\n"
sumwriter.add_text("Parameters", f"{main_str}", 0)


################################################
#          Net , optimizers
################################################


pmodel, pnet_name = fetch_pmodel(args.net, args.hp_type, version=args.version)

train_root = args.backbone  # default soft-label ff_ckpt; --backbone for the one-hot control
# train_root=None
print("pretrained ckpts:", train_root)
print("pnet name:", pnet_name)
print("batchsize:", args.batch_size)
print("learning rate:", args.lr)
net = get_model(pretrained=False, trained_root=train_root, model=args.net)
pnet = pmodel(net, build_graph=True, random_init=False).to(
    device
) 
#patch all PCoderN instances with 1D-compatible compute_C_sqrt
patch_pcoders(pnet)

NUMBER_OF_PCODERS = pnet.number_of_pcoders

best_loss = float("inf")

if args.resume_training:

    assert len(args.resume_ckpts) == NUMBER_OF_PCODERS
    "the number os ckpts provided is not equal to the number of pcoders"

    print("-" * 30)
    print(f"Loading checkpoint from {args.resume_ckpts}")
    print("-" * 30)

    for x in range(NUMBER_OF_PCODERS):
        checkpoint = torch.load(args.resume_ckpts[x], weights_only=False)
        args.start_epoch = checkpoint["epoch"] + 1
        getattr(pnet, f"pcoder{x+1}").pmodule.load_state_dict(
            {
                k[len("pmodule.") :]: v
                for k, v in checkpoint["pcoderweights"].items()
                if k != "C_sqrt"
            }
        )
    best_loss = checkpoint["loss"]
    print(
        f"Checkpoint loaded from epoch { checkpoint['epoch']} with loss {checkpoint['loss']}"
    )
    print("-" * 30)

else:
    print("Training from scratch...")

loss_function = nn.MSELoss()

if args.optim_name == "SGD":
    optimizer = optim.SGD(
        [
            {
                "params": getattr(pnet, f"pcoder{x+1}").pmodule.parameters(),
            }
            for x in range(NUMBER_OF_PCODERS)
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
elif args.optim_name == "Adam":  # not used for any reported model
    optimizer = optim.Adam(
        [
            {
                "params": getattr(pnet, f"pcoder{x+1}").pmodule.parameters(),
            }
            for x in range(NUMBER_OF_PCODERS)
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


################################################
#       Dataset and train-test helpers
################################################
transform_val = transforms.Compose(
    [
        # transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# training/validation images: webdataset shards in fname.dataset_dir
train_wrdset = (
    wds.WebDataset(glob.glob(f"{fname.dataset_dir}/train*.tar"), shardshuffle=True)
    .decode("pil")
    .map_dict(png=transform)
    .to_tuple("png", "cls")
    # .batched(args.batch_size)
)


training_gen = torch.utils.data.DataLoader(
    train_wrdset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
)

val_wrdset = (
    wds.WebDataset(glob.glob(f"{fname.dataset_dir}/val*.tar"), shardshuffle=False)
    .decode("pil")
    .map_dict(png=transform)
    .to_tuple("png", "cls")
    # .batched(args.batch_size)
)
if args.val_samples is not None:
    val_wrdset = val_wrdset.with_length(args.val_samples)

validation_gen = torch.utils.data.DataLoader(
    val_wrdset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)


def train_pcoders(net, epoch, sumwriter, train_loader):
    """A training epoch"""

    net.train()

    tstart = time.time()
    n = 0
    for batch_index, (images, _) in enumerate(train_loader):
        n += 1
        net.reset()  # rep: None; prd: None
        images = images.to(device)  # (64,3,224,224)
        # print("images shape:", images.shape)
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                a = net.pcoder1.prediction_error
                loss = a
            else:
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                a = pcoder_curr.prediction_error
                loss += a

            sumwriter.add_scalar(
                f"MSE Train/PCoder{i+1}",
                a.item(),
                epoch * n + batch_index,
            )

        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            print(
                "Training Epoch: {epoch} [{trained_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                    epoch=epoch,
                    trained_samples=batch_index * args.batch_size + len(images),
                )
            )
            print("Time taken:", time.time() - tstart)
        sumwriter.add_scalar("MSE Train/Sum", loss.item(), epoch * n + batch_index)


def test_pcoders(net, epoch, sumwriter, test_loader, verbose=True):
    """A testing epoch"""

    net.eval()

    tstart = time.time()
    final_loss = [0 for i in range(NUMBER_OF_PCODERS)]
    n = 0
    for batch_index, (images, _) in enumerate(test_loader):
        n += 1
        net.reset()
        images = images.to(device)
        with torch.no_grad():
            outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                final_loss[i] += net.pcoder1.prediction_error.item()
            else:
                # pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                final_loss[i] += pcoder_curr.prediction_error.item()

    loss_sum = 0
    for i in range(NUMBER_OF_PCODERS):
        final_loss[i] /= n
        loss_sum += final_loss[i]
        sumwriter.add_scalar(f"MSE Test/PCoder{i+1}", final_loss[i], int(epoch))
    sumwriter.add_scalar("MSE Test/Sum", loss_sum, int(epoch))

    print(
        "Testing Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
            loss_sum,
            optimizer.param_groups[-1]["lr"],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=n * args.batch_size,
        )
    )
    print("Time taken:", time.time() - tstart)

    return loss_sum


################################################
#              Train loops
################################################
patience = 10

patience_counter = 0

# Tagged runs (e.g. --out_tag _seed5, training seeds) also keep a
# per-epoch state file for exact resumption, a <base>_best_pcN.pth copy of the
# best weights for 06_optimize_hyperparameters.py, and a <base>_DONE marker.
TAGGED = bool(args.out_tag)
BASE = f"{fname.pnet_dir}/p_{args.net}_{args.hp_type}_{args.version}{args.out_tag}"
if TAGGED and args.auto_resume and os.path.exists(f"{BASE}_last.pth"):
    last = torch.load(f"{BASE}_last.pth", weights_only=False)
    for x in range(NUMBER_OF_PCODERS):
        getattr(pnet, f"pcoder{x+1}").load_state_dict(last["pcoderweights"][x])
    args.start_epoch = last["epoch"] + 1
    best_loss, patience_counter = last["best_loss"], last["patience_counter"]
    print(f"auto-resume from epoch {last['epoch']} (best loss {best_loss}, "
          f"patience counter {patience_counter})")


def atomic_save(obj, filename):
    # write to a temporary file and rename it: a job killed mid-write leaves the
    # previous checkpoint intact
    tmp = f"{filename}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, filename)


def save_checkpoint(state, save_path, epoch=None):
    # args.out_tag keeps a variant run (e.g. --out_tag _onehot) separate from the
    # soft-label PCoder weights; get_pmodel's pnet_path points at the same base.
    base = f"{save_path}/p_{args.net}_{args.hp_type}_{args.version}{args.out_tag}"
    if epoch is not None:
        filename = f"{base}_epoch-{epoch:02d}_pc{pcod_idx+1}.pth"
    else:
        filename = f"{base}_best_pc{pcod_idx+1}.pth"
    atomic_save(state, filename)


for epoch in range(args.start_epoch, args.num_epochs):

    train_pcoders(pnet, epoch, sumwriter, training_gen)

    loss = test_pcoders(pnet, epoch, sumwriter, validation_gen)

    is_best = (best_loss - loss) > args.min_delta

    if is_best:
        best_loss = min(loss, best_loss)
        patience_counter = 0
        print("save best model at epoch:", epoch, "best Loss: ", best_loss)
        # tagged runs keep only <base>_best_pcN.pth and <base>_last.pth (per-epoch
        # copies take ~0.5 GB per epoch)
        for pcod_idx in range(0 if TAGGED else NUMBER_OF_PCODERS):
            save_checkpoint(
                {
                    "pcoderweights": getattr(pnet, f"pcoder{pcod_idx+1}").state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "loss": best_loss,
                },
                fname.pnet_dir,
                epoch,
            )
        if TAGGED:
            for pcod_idx in range(NUMBER_OF_PCODERS):
                save_checkpoint(
                    {
                        "pcoderweights": getattr(pnet, f"pcoder{pcod_idx+1}").state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "loss": best_loss,
                    },
                    fname.pnet_dir,
                )
    else:
        patience_counter += 1
        print(patience_counter, "patience counter")

    if TAGGED:
        atomic_save(
            {
                "pcoderweights": [getattr(pnet, f"pcoder{x+1}").state_dict()
                                  for x in range(NUMBER_OF_PCODERS)],
                "epoch": epoch,
                "best_loss": best_loss,
                "patience_counter": patience_counter,
            },
            f"{BASE}_last.pth",
        )

    if patience_counter >= patience:
        print(f"patience limit reached, stopping training {epoch}")
        print("Best Loss:", best_loss, "Current Loss:", loss)
        break

    # scheduler.step(loss)

if TAGGED:
    open(f"{BASE}_DONE", "w").write(f"best loss {best_loss}\n")
# %%
