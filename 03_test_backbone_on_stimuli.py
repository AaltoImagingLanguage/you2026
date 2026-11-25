"""Top-1 and top-5 accuracy of a feedforward backbone on the experimental stimuli, per
condition (RW, RL1, RL2, RL3); used for the soft-label vs one-hot comparison (S3 Table,
assembled by S3Tab_soft_vs_onehot.py).

Input: ../data/images_dataset/stimuli.tar (01_make_image_datasets.py) and a backbone
checkpoint (default ../data/model_ckps/ff_backbone/vgg16_checkpoint_best.pth.tar).
Output: printed accuracies.

    python 03_test_backbone_on_stimuli.py
    python 03_test_backbone_on_stimuli.py --backbone ../data/model_ckps/ff_backbone/vgg16_checkpoint_best_onehot.pth.tar
"""
import torch, time
from time import strftime, localtime
from datetime import timedelta

from torch.utils import data
import webdataset as wds


torch.cuda.empty_cache()
import argparse
from utility import accuracy, get_model, transform, AverageMeter
from config import fname

parser = argparse.ArgumentParser(description="Testing")
parser.add_argument(
    "--num_val_items", default=1, type=int, help="number of validation items in each category"
)
parser.add_argument(
    "--num_workers", default=1, type=int, help="number of workers to load batches in parallel"
)
parser.add_argument(
    "--ngpus",
    default=1,
    type=int,
    help="number of GPUs to use; 0 if you want to run on CPU",
)
parser.add_argument("--data_type", default="stimuli", help="data type to extract from")
parser.add_argument(
    "--backbone", type=str, default=str(fname.ff_ckpt),
    help="backbone checkpoint to test (default: soft-label ff_ckpt; pass the "
    "_onehot backbone for the one-hot control, S3 Table)",
)
FLAGS, _ = parser.parse_known_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


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

    return top1.avg, top5.avg


def test(
    net="vgg16",
):
    start_time = time.time()

    torch.backends.cudnn.benchmark = True

    net = get_model(
        pretrained=True,
        trained_root=FLAGS.backbone,
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

    prec1, prec5 = validate(
        validation_gen,
        net,
    )
    conds = ["RW", "RL1", "RL2", "RL3"]  # order matches json["type_idx"] 0..3
    print("backbone tested:", FLAGS.backbone)
    print("per-condition top-1 (%):", dict(zip(conds, [round(p * 100, 1) for p in prec1])))
    print("per-condition top-5 (%):", dict(zip(conds, [round(p * 100, 1) for p in prec5])))
    exec_time = secondsToStr(time.time() - start_time)
    print("execution time so far: ", exec_time)
    return True


if __name__ == "__main__":
    test()
