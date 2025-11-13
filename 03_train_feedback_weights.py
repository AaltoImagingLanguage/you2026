#########################
# In this script we train PVGG feedback weights only.
#########################
# %%
import torch
import torchvision.transforms as transforms
import webdataset as wds
import glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from config import fname
import os
import time
from utility import get_model, fetch_pmodel, transform

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

# TensorBoard
parser.add_argument(
    "--start_epoch", type=int, default=0, help="Epoch from which training starts"
)


# Training
parser.add_argument("--batchsize", type=int, default=50, help="Batch size for training")
parser.add_argument(
    "--num_workers", type=int, default=8, help="Number of workers for data loading"
)
parser.add_argument(
    "--num_epochs", type=int, default=200, help="Number of training epochs"
)

# Optimization
parser.add_argument("--optim_name", type=str, default="Adam", help="Optimizer name")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
parser.add_argument("--ckpt_every", action="store_true", help="Checkpointing frequency")

# Resuming Training
parser.add_argument(
    "--resume_training",
    type=bool,
    default=True,
    help="Resume training from checkpoint",
)
parser.add_argument(
    "--resume_ckpts",
    type=str,
    nargs="+",
    default=[
        "save/fb_pnet/p_vgg16_Separate_v1_best_pc1.pth",
        "save/fb_pnet/p_vgg16_Separate_v1_best_pc2.pth",
        "save/fb_pnet/p_vgg16_Separate_v1_best_pc3.pth",
    ],
    help="Checkpoint file paths",
)  # accept one or more values as a list


args, _ = parser.parse_known_args()


# %%

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_to_use)


# Setup the training
if args.random_seed:
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


device = torch.device("cuda:0")


################################################
#          Net , optimizers
################################################
## Change this to change the network


pmodel, pnet_name = fetch_pmodel(args.net, args.hp_type, version=args.version)

train_root = fname.ff_ckpt
# train_root=None
print("pretrained ckpts:", train_root)
print("pnet name:", pnet_name)
print("batchsize:", args.batchsize)
print("learning rate:", args.lr)
net = get_model(pretrained=True, trained_root=train_root, ngpus=1, model=args.net)
pnet = pmodel(net, build_graph=True, random_init=False).to(
    device
)  # build_graph: gradients


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
elif args.optim_name == "Adam":

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

train_wrdset = (
    wds.WebDataset(glob.glob(f"{fname.dataset_dir}/train*.tar"), shardshuffle=True)
    .decode("pil")
    .map_dict(png=transform)
    .to_tuple("png", "cls")
    .batched(args.batch_size)
)
training_gen = torch.utils.data.DataLoader(
    train_wrdset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)

val_wrdset = (
    wds.WebDataset(glob.glob(f"{fname.dataset_dir}/val*.tar"), shardshuffle=False)
    .decode("pil")
    .map_dict(png=transform)
    .to_tuple("png", "cls")
    .batched(args.batch_size)
)
validation_gen = torch.utils.data.DataLoader(
    val_wrdset,
    batch_size=args.num_val_items,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)


def train_pcoders(net, epoch, sumwriter, train_loader, verbose=True):
    """A training epoch"""

    net.train()

    tstart = time.time()
    for batch_index, (images, _) in enumerate(train_loader):
        net.reset()  # rep: None; prd: None
        images = images.to(device)  # (64,3,224,224)
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                a = net.pcoder1.prediction_error  # net.pcoder1.prd (-14218, -1064)
                # print("net.pcoder1 loss:", a.item())
                loss = a
            else:
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                a = pcoder_curr.prediction_error
                loss += a

            sumwriter.add_scalar(
                f"MSE Train/PCoder{i+1}",
                a.item(),
                epoch * len(train_loader) + batch_index,
            )

        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            print(
                "Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                    epoch=epoch,
                    trained_samples=batch_index * args.batchsize + len(images),
                    total_samples=len(train_loader.dataset),
                )
            )
            print("Time taken:", time.time() - tstart)
        sumwriter.add_scalar(
            f"MSE Train/Sum", loss.item(), epoch * len(train_loader) + batch_index
        )


def test_pcoders(net, epoch, sumwriter, test_loader, verbose=True):
    """A testing epoch"""

    net.eval()

    tstart = time.time()
    final_loss = [0 for i in range(NUMBER_OF_PCODERS)]
    for batch_index, (images, _) in enumerate(test_loader):
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
        final_loss[i] /= len(test_loader)
        loss_sum += final_loss[i]
        sumwriter.add_scalar(f"MSE Test/PCoder{i+1}", final_loss[i], int(epoch))
    sumwriter.add_scalar(f"MSE Test/Sum", loss_sum, int(epoch))

    print(
        "Testing Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
            loss_sum,
            optimizer.param_groups[-1]["lr"],
            epoch=epoch,
            trained_samples=batch_index * args.batchsize + len(images),
            total_samples=len(test_loader.dataset),
        )
    )
    print("Time taken:", time.time() - tstart)

    return loss_sum


################################################
#        Load checkpoints if given...
################################################
if not os.path.exists(f"{fname.log_dir}/{args.tb_dir}"):
    os.makedirs(f"{fname.log_dir}/{args.tb_dir}")

# summarywriter
tensorboard_path = os.path.join(
    f"{fname.log_dir}/{args.tb_dir}",
    f"{args.net}_{args.version}_{args.lr}_{args.suffix}",
)
sumwriter = SummaryWriter(tensorboard_path, filename_suffix=f"")

main_str = ""
for x in vars(args):
    main_str += f"{x:<20}: {getattr(args,x)}\n"
sumwriter.add_text("Parameters", f"{main_str}", 0)


################################################
#              Train loops
################################################
patience = 10

patience_counter = 0


def save_checkpoint(state, save_path, epoch=None):
    if epoch is not None:
        filename = f"{save_path}/p_{args.net}_{args.hp_type}_{args.version}_{args.suffix}_epoch-{epoch:02d}_pc{pcod_idx+1}.pth"
    else:
        filename = f"{save_path}/p_{args.net}_{args.hp_type}_{args.version}_best_pc{pcod_idx+1}.pth"
    torch.save(state, filename)


for epoch in range(args.start_epoch, args.num_epochs):

    train_pcoders(pnet, epoch, sumwriter, training_gen)

    loss = test_pcoders(pnet, epoch, sumwriter, validation_gen)

    is_best = (best_loss - loss) > 0.001

    if is_best:
        best_loss = min(loss, best_loss)
        patience_counter = 0
        print("save best model at epoch:", epoch, "best Loss: ", best_loss)
        for pcod_idx in range(NUMBER_OF_PCODERS):
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
    else:
        patience_counter += 1
        print(patience_counter, "patience counter")

    if patience_counter >= patience:
        print(f"patience limit reached, stopping training {epoch}")
        print("Best Loss:", best_loss, "Current Loss:", loss)
        break

    # scheduler.step(loss)
# %%
