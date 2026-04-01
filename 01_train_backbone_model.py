#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Training script for pre-trained VGG16 with frozen early layers using soft targets derived from word similarities.
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
    AverageMeter,
    VGG16_FT,
    compute_soft_target,
)
from config import fname

parser = argparse.ArgumentParser(description="Training")
parser.add_argument(
    "--restore_file",
    default=fname.ff_ckpt,
    help="fath name of file from which to restore model (ought to be located in save path)",
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
    "--lr", "--learning_rate", default=0.001, type=float, help="initial learning rate"
)

parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay ")

parser.add_argument(
    "--temp",
    default=1,
    type=float,
    help="controls how sharply the score falls off as distance increases",
)

FLAGS, _ = parser.parse_known_args()
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


def save_checkpoint(state, epoch=None):
    if epoch is not None:
        filename = f"{fname.bb_path}/vgg16_epoch-{epoch:02d}.pth.tar"
    else:
        filename = fname.ff_ckpt
    torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch):
    print("=> training on device", device)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, json) in enumerate(train_loader):
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

        # loss = criterion(output, target)
        target_words = [words1k[i] for i in target]
        soft_targets = [
            compute_soft_target(y_word, words1k, FLAGS.temp) for y_word in target_words
        ]
        soft_targets = torch.stack(soft_targets).to(device)  # (batch_size, 1000)
        loss = criterion(output, soft_targets)

        type_idx = json["type_idx"]
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item())
        top1.update(prec1, input.size(0), type_idx)
        top5.update(prec5, input.size(0), type_idx)

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
                f"Prec@1 {top1.val[type_idx]:.3f} ({top1.avg[type_idx]:.3f}) "
                f"Prec@5 {top5.val[type_idx]:.3f} ({top5.avg[type_idx]:.3f})",
                flush=True,
            )


def validate(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, json) in enumerate(test_loader):
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

            type_idx = json["type_idx"]

            top1.update(prec1, input.size(0), type_idx)
            top5.update(prec5, input.size(0), type_idx)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print(
                    f"Test: [{i:04d}/{500:04d}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val[type_idx]:.3f} ({top1.avg[type_idx]:.3f})\t"
                    f"Prec@5 {top5.val[type_idx]:.3f} ({top5.avg[type_idx]:.3f})",
                    flush=True,
                )

        print(
            f" * Prec@1 {top1.avg[type_idx]:.3f} Prec@5 {top5.avg[type_idx]:.3f}",
            flush=True,
        )

    return top1.avg


def train_all():
    start_time = time.time()

    # CUDA for PyTorch
    # device = "cpu"
    torch.backends.cudnn.benchmark = True
    train_wrdset = (
        wds.WebDataset(glob.glob(f"{fname.dataset_dir}/train*.tar"), shardshuffle=True)
        .decode("pil")
        .map_dict(png=transform)
        .to_tuple("png", "cls")
        .batched(FLAGS.batch_size)
    )
    training_gen = data.DataLoader(
        train_wrdset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )

    val_wrdset = (
        wds.WebDataset(glob.glob(f"{fname.dataset_dir}/val*.tar"), shardshuffle=False)
        .decode("pil")
        .map_dict(png=transform)
        .to_tuple("png", "cls")
        .batched(FLAGS.batch_size)
    )
    validation_gen = data.DataLoader(
        val_wrdset,
        batch_size=FLAGS.num_val_items,
        shuffle=False,
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
        optimizer.load_state_dict(checkpoint["optimizer"])
        net.load_state_dict(checkpoint["state_dict"], strict=False)

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
                fname.net_dir,
                None,
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
