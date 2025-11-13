#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import torch, time
from time import strftime, localtime
from datetime import timedelta
import torch.nn as nn

from torch.utils import data
import webdataset as wds


torch.cuda.empty_cache()
import argparse
from utility import accuracy, get_model, transform, AverageMeter
from config import fname

parser = argparse.ArgumentParser(description="Testing")
parser.add_argument(
    "--num_val_items", default=1, help="number of validation items in each category"
)
parser.add_argument(
    "--num_workers", default=1, help="number of workers to load batches in parallel"
)
parser.add_argument(
    "--ngpus",
    default=1,
    type=int,
    help="number of GPUs to use; 0 if you want to run on CPU",
)
parser.add_argument("--data_type", default="stimuli", help="data type to extract from")
FLAGS, _ = parser.parse_known_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# useful
def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def validate(
    test_loader,
    model,
):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, json) in enumerate(test_loader):
            # stimulus
            if type(target) == list:
                target = [t.to(device, non_blocking=True) for t in target]
            else:
                target = target.to(device, non_blocking=True)

            # compute output
            input = input.to(device, non_blocking=True)
            output = model(input)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            type_idx = json["type_idx"]
            top1.update(prec1, input.size(0), type_idx)
            top5.update(prec5, input.size(0), type_idx)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg


def test(
    net="vgg16",
):
    start_time = time.time()

    torch.backends.cudnn.benchmark = True

    net = get_model(
        pretrained=True,
        trained_root=fname.ff_ckpt,
        model=net,
        ngpus=FLAGS.ngpus,
    )

    # Datasets and Generators
    print("loading datasets")

    val_wrdset = (
        wds.WebDataset(f"{fname.dataset_dir}/{FLAGS.data_type}.tar")
        .decode("pil")
        .map_dict(png=transform)
        .to_tuple("png", "cls", "json")
    )

    validation_gen = data.DataLoader(
        val_wrdset,
        batch_size=FLAGS.num_val_items,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=False,
    )

    # Model

    prec1 = validate(
        validation_gen,
        net,
    )
    print("test accuracy: ", [round(p * 100, 1) for p in prec1])
    exec_time = secondsToStr(time.time() - start_time)
    print("execution time so far: ", exec_time)
    return True


if __name__ == "__main__":
    test()
