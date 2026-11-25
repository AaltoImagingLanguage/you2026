"""Train the feedforward backbone: ImageNet-pretrained VGG16 with frozen convolutional
layers up to pool4 (VGG16_FT), fine-tuned to classify the rendered words of the
1000-word vocabulary.

Training target (--target): "soft" (default), the soft-label distribution over the
vocabulary derived from the Levenshtein distance to the base word (temperature
1), or "onehot", the base-word label (one-hot control, S3 Table). Both use Adam
(lr 1e-4, weight decay 5e-4, batch size 100) and stop after 10 epochs without a gain in
validation accuracy; the weights with the highest validation accuracy are saved as
{bb_dir}/vgg16_checkpoint_best_{target}{tag}.pth.tar. The released backbone of the
original model is vgg16_checkpoint_best.pth.tar (validation accuracy 97.0%), the file
read by all later steps.

    python 02_train_backbone.py                  # soft-label backbone
    python 02_train_backbone.py --target onehot  # one-hot control

The backbones of the training seeds (5, 15, 25, 35) are trained with the same settings by
10_train_backbone_seeds.py, which caches the frozen layers.
"""
import torch, time
from time import strftime, localtime
from datetime import timedelta
import torch.nn as nn
from torch.utils import data
import webdataset as wds
import glob
from utility import transform
import pickle

torch.cuda.empty_cache()
import argparse
from utility import (
    accuracy,
    AverageMeter_train,
    VGG16_FT,
    compute_soft_target,
)
from config import fname

parser = argparse.ArgumentParser(description="Training")
parser.add_argument(
    "--restore_file",
    default="",
    help="optional checkpoint to continue training from (weights only)",
)
parser.add_argument(
    "--start_epoch", type=int, default=0, help="Epoch from which training starts"
)
parser.add_argument(
    "--num_val_items", default=50, help="number of validation items in each word"
)
parser.add_argument(
    "--num_workers", default=10, help="number of workers to load batches in parallel"
)
parser.add_argument(
    "--max_epochs",
    default=300,
    type=int,
    help="maximun number of epochs to run for training",
)
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument(
    "--lr", "--learning_rate", default=1e-4, type=float, help="initial learning rate"
)

parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
parser.add_argument(
    "--target",
    default="soft",
    type=str,
    choices=["onehot", "soft"],
    help="training target: 'onehot' (standard cross-entropy on the class label) or "
    "'soft' (Levenshtein-similarity soft targets, the original objective)",
)
parser.add_argument("--seed", default=None, type=int,
                    help="random seed (classifier initialization, sample shuffling); "
                    "None keeps the previous unseeded behaviour")
parser.add_argument("--tag", default="", type=str,
                    help="suffix of the saved checkpoint, e.g. _seed1 -> "
                    "vgg16_checkpoint_best_soft_seed1.pth.tar")
parser.add_argument(
    "--temp",
    default=1,
    type=float,
    help="controls how sharply the score falls off as distance increases",
)

FLAGS, _ = parser.parse_known_args()
if FLAGS.seed is not None:
    import random
    import numpy as np
    random.seed(FLAGS.seed)  # webdataset .shuffle() draws from the random module
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

with open(fname.word2idx_dir, "rb") as file:
    word2idx10k = pickle.load(file)

words1k = list(word2idx10k.keys())


# useful
def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def save_checkpoint(state):
    # e.g. vgg16_checkpoint_best_onehot.pth.tar (the --backbone path of the later steps)
    filename = str(fname.ff_ckpt).replace(".pth.tar", f"_{FLAGS.target}{FLAGS.tag}.pth.tar")
    torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch):
    print("=> training on device", device)
    batch_time = AverageMeter_train()
    data_time = AverageMeter_train()
    losses = AverageMeter_train()
    top1 = AverageMeter_train()
    top5 = AverageMeter_train()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device, non_blocking=True)
        if type(target) == list:
            target = [t.to(device, non_blocking=True) for t in target]
        else:
            target = target.to(device, non_blocking=True)

        # compute output
        output = model(input)  # (batch_size, 1000)
        if FLAGS.target == "onehot":
            loss = criterion(output, target)
        else:

        # loss = criterion(output, target)
            target_words = [words1k[i] for i in target]
            soft_targets = [
                compute_soft_target(y_word, words1k, FLAGS.temp) for y_word in target_words
            ]
            soft_targets = torch.stack(soft_targets).to(device)  # (batch_size, 1000)
            loss = criterion(output, soft_targets)

        # type_idx = json["type_idx"]
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item())
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 50 == 0:
            # print('Density:', output[0].shape[1] - torch.sum(output[0] == 0, dim=1).float().mean())
            print(
                f"Epoch: [{epoch}][{i:04d}/{2000:04d}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f}) "
                f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})",
                flush=True,
            )


def validate(test_loader, model, criterion):
    batch_time = AverageMeter_train()
    losses = AverageMeter_train()
    top1 = AverageMeter_train()
    top5 = AverageMeter_train()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):

            input = input.to(device, non_blocking=True)
            if type(target) == list:
                target = [t.to(device, non_blocking=True) for t in target]
            else:
                target = target.to(device, non_blocking=True)

            input = input.to(device, non_blocking=True)

            # compute output
            output = model(input)

            target_words = [words1k[i] for i in target]
            soft_targets = [
                compute_soft_target(y_word, words1k, FLAGS.temp)
                for y_word in target_words
            ]
            soft_targets = torch.stack(soft_targets).to(device)  # (batch_size, 1000)
            loss = criterion(output, soft_targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item())

            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print(
                    f"Test: [{i:04d}/{500:04d}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})",
                    flush=True,
                )

        print(f" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}", flush=True)

    return top1.avg


def train_all():
    start_time = time.time()

    # CUDA for PyTorch
    # device = "cpu"
    torch.backends.cudnn.benchmark = True
    # import ds2
    # train_wrdset = ds2.WordDataset1(data_path=FLAGS.wrd_path, folder="train")
    # training_gen = data.DataLoader(
    #     train_wrdset,
    #     batch_size=FLAGS.batch_size,
    #     shuffle=True,
    #     num_workers=FLAGS.num_workers,
    #     pin_memory=True,
    # )

    
    # # with open("../data/word2idx.pkl", "wb") as f:
    # #     pickle.dump(wordlist2idx, f)
    # val_wrdset = ds2.WordDataset1(data_path=FLAGS.wrd_path, folder="val")
    # validation_gen = data.DataLoader(
    #     val_wrdset,
    #     batch_size=FLAGS.num_val_items,
    #     shuffle=False,
    #     num_workers=FLAGS.num_workers,
    # )

    # training/validation images: webdataset shards in fname.dataset_dir
    train_wrdset = (
        wds.WebDataset(glob.glob(f"{fname.dataset_dir}/train*.tar"))
        # CRITICAL: sample-level shuffle. The tars are written word-by-word (all
        # renders of one word together), so without this every batch is a single
        # class and the model cannot learn (loss stays at ln(1000)=6.91, acc=chance).
        # shardshuffle only reorders shards, not samples within them.
        .shuffle(5000)
        .decode("pil")
        .map_dict(png=transform)
        .to_tuple("png", "cls")
    )
    training_gen = data.DataLoader(
        train_wrdset,
        batch_size=FLAGS.batch_size,  # DataLoader batches samples; IterableDataset forbids shuffle=
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )

    val_wrdset = (
        wds.WebDataset(glob.glob(f"{fname.dataset_dir}/val*.tar"))
        .decode("pil")
        .map_dict(png=transform)
        .to_tuple("png", "cls")
       
    )
    validation_gen = data.DataLoader(
        val_wrdset,
        batch_size=FLAGS.batch_size,  # dataset already batches via .batched()
        num_workers=FLAGS.num_workers,
    )

    net = VGG16_FT()  # load vgg16 model with early layers frozen
    net = net.cuda()

    if FLAGS.restore_file:

        print("-" * 30)
        print(f"Loading checkpoint from {FLAGS.restore_file}")
        print("-" * 30)

        checkpoint = torch.load(FLAGS.restore_file, weights_only=False)
        FLAGS.start_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["best_prec1"]
        # weights only: the optimizer is built afresh below and best_prec1 is reset
        missing, unexpected = net.load_state_dict(checkpoint["state_dict"], strict=False)
        assert not missing and not unexpected, (missing, unexpected)

        print(
            f"Checkpoint loaded from epoch { checkpoint['epoch']} when accuracy = {best_prec1}"
        )
        print("-" * 30)

    else:
        print("Training from scratch...")

    # Datasets and Generators
    print("loading datasets")

    # Model

    # net.module[0].nonlin is the fisrt Relu layer for Predive coding
    exec_time = secondsToStr(time.time() - start_time)
    print("execution time so far: ", exec_time)

    # Build loss function, model and optimizer.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
    )

    """
    train
    """
    max_epochs = FLAGS.max_epochs
    patience = 10
    best_prec1 = -float("inf")
    patience_counter = 0  # Initialize patience_counter

    # Loop over epochs
    for epoch in range(FLAGS.start_epoch, max_epochs):

        # Training
        start_time = time.time()
        train(training_gen, net, criterion, optimizer, epoch)
        prec1 = validate(validation_gen, net, criterion)
        print(prec1)
        print("|| 1 ||", flush=True)
        exec_time = secondsToStr(time.time() - start_time)
        print("execution time so far: ", exec_time)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                    "args": FLAGS,
                },
            )
            print("===============save best model at epoch: ", epoch)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"patience limit reached, stopping training {epoch}")
            break

        print("|| 2 ||", flush=True)
    return True


if __name__ == "__main__":

    train_all()
