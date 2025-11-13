# %%

import webdataset as wds
from plotnine import *

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

import torch
from utility import accuracy, compute_soft_target, get_sti_rank, transform
from config import (
    category_colors,
    event_id,
    fname,
    max_timestep,
    fb_timestep,
    fb_colors,
)
import os
import time

with open(fname.word2idx_dir, "rb") as file:
    word2idx10k = pickle.load(file)

words1k = list(word2idx10k.keys())
sti_types = list(event_id.keys())

device = torch.device("cuda:0")
import pandas as pd

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utility import get_pmodel
import argparse

parser = argparse.ArgumentParser(description="plot model behavior")
parser.add_argument(
    "--batchsize",
    type=int,
    default=60,
    help="batch size for computing model behavioral results",
)
parser.add_argument(
    "--compute",
    type=bool,
    default=False,
    help="whether to compute the model behaviral results or load from saved files",
)
parser.add_argument(
    "--plot_acc", type=bool, default=True, help="Whether to plot accuracy"
)
parser.add_argument(
    "--plot_out", type=bool, default=False, help="Whether to plot output activations"
)

args, _ = parser.parse_known_args()


def get_acc_list(net, max_timestep):

    net.eval()

    stimulus_acc = {sti: {} for sti in sti_types}
    sti_out = {"id": [], "group": [], "w/o feedback": [], "w/ feedback": []}
    tstart = time.time()

    for i, (inputs, labels, json) in enumerate(val_loader, 0):

        stimulus_types = json["type"]

        inputs = inputs.to(device, non_blocking=True)

        if type(labels) == list:
            labels = [t.to(device, non_blocking=True) for t in labels]
        else:
            labels = labels.to(device, non_blocking=True)
        # get reps for clean images
        for t in range(max_timestep):
            if t == 0:
                with torch.no_grad():
                    outputs = net(inputs)

            else:
                with torch.no_grad():
                    outputs = net(None)

            target_words = json["base"]
            soft_targets = [
                compute_soft_target(y_word, words1k, 2) for y_word in target_words
            ]
            soft_targets = torch.stack(soft_targets).to(device)  # (batch_size, 1000)
            for out, label, stimulus_type in zip(outputs, labels, stimulus_types):
                prec = accuracy(out[None, :], label[None], topk=(1,))

                # Save accuracy for the current stimulus type and time step
                if t not in stimulus_acc[stimulus_type]:
                    stimulus_acc[stimulus_type][t] = []
                stimulus_acc[stimulus_type][t].append(prec[0])

            if t == 0:
                index = get_sti_rank(outputs, target_words)
                sti_out["w/o feedback"].extend(index)
            elif t == fb_timestep:
                index = get_sti_rank(outputs, target_words)
                sti_out["w/ feedback"].extend(index)

        sti_out["id"].extend(json["word"])
        sti_out["group"].extend([type[:3] for type in json["type"]])

        print("Time taken:", time.time() - tstart, "for", i)

    df_long = pd.DataFrame(sti_out).melt(
        id_vars=["id", "group"],
        value_vars=["w/o feedback", "w/ feedback"],
        var_name="PCoder 3",
        value_name="Rank of target word",
    )

    return (stimulus_acc, df_long)


compute = False
tstart = datetime.now()
if args.compute:
    val_wrdset = (
        wds.WebDataset(f"{fname.dataset_dir}/stimuli.tar")
        .decode("pil")
        .map_dict(png=transform)
        .to_tuple("png", "cls", "json")
    )

    val_loader = torch.utils.data.DataLoader(
        val_wrdset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    hps_root = fname.hps_ckpt
    hps_data = torch.load(hps_root, weights_only=False)
    hps = hps_data["hps"]
    print("Loaded hps from:", hps_root)
    hps = [
        {"ffm": hps[0], "fbm": hps[1], "erm": hps[3]},
        {"ffm": hps[4], "fbm": hps[5], "erm": hps[7]},
        {"ffm": hps[8], "fbm": hps[9], "erm": hps[11]},
    ]

    backbone_path = fname.ff_ckpt
    pnet_path = fname.pcoder_ckpt

    model = get_pmodel(
        pnet_path=pnet_path,
        backbone_path=backbone_path,
        hyperparams=hps,
    ).to(device)

    NUMBER_OF_PCODERS = model.number_of_pcoders

    accs_sti_dict, df_long = get_acc_list(model, max_timestep)
    with open(
        fname.accs,
        "wb",
    ) as f:
        pickle.dump(accs_sti_dict, f)

    with open(
        fname.out_atvs,
        "wb",
    ) as f:
        pickle.dump(df_long, f)

    print(
        "Saved accs to:",
        fname.accs,
        " and df of output activations to:",
        fname.out_atvs,
    )
else:
    with open(
        fname.out_atvs,
        "rb",
    ) as f:
        df_long = pickle.load(f)
    print("Loaded df of output activations from:", fname.out_atvs)

    with open(
        fname.accs,
        "rb",
    ) as f:
        accs_sti_dict = pickle.load(f)


if args.plot_out:

    df_long["PCoder 3"] = pd.Categorical(
        df_long["PCoder 3"],
        categories=["w/o feedback", "w/ feedback"],
        ordered=True,
    )

    # Set desired order of group
    df_long["group"] = pd.Categorical(
        df_long["group"], categories=[type[:3] for type in sti_types], ordered=True
    )

    # Create a "count" column: number of overlapping points per x, y
    df_long["y_bin"] = df_long["Rank of target word"].round(
        1
    )  # bin y to reduce unique values
    overlap = (
        df_long.groupby(["group", "PCoder 3", "y_bin"]).size().reset_index(name="count")
    )
    df_long = df_long.merge(overlap, on=["group", "PCoder 3", "y_bin"])

    plot = (
        ggplot(
            df_long,
            aes(x="PCoder 3", y="Rank of target word", group="id", alpha="count"),
        )
        + geom_line(
            color="gray",
            alpha=0.6,
            size=0.6,
        )
        + geom_point(aes(color="PCoder 3"), size=2.5)
        + theme_bw()
        + scale_alpha(range=(0.25, 1))  # small alpha = light line, high alpha = dark
        + facet_wrap(
            "~group", nrow=1, scales="free_y"  # 👈 independent y-axis for each group
        )
        + scale_color_manual(
            values={"w/o feedback": fb_colors[0], "w/ feedback": fb_colors[1]}
        )
        + coord_cartesian(ylim=(1, None))
        + labs(x="", y="Rank of base word in the lexical units")
        + theme(
            figure_size=(8, 4),  # smaller width
            legend_position="none",
            axis_text_x=element_text(
                rotation=10,
            ),
            # subplots_adjust={"wspace": 0.05},  # reduce horizontal spacing
            panel_grid_major=element_line(color="gray", size=0.3, linetype="dotted"),
            # plot_title=element_text(size=14, weight="bold", ha="center"),
        )
    )
    plot.save(fname.fig_pair_out, dpi=300)

if args.plot_acc:
    from brokenaxes import brokenaxes

    current_color = plt.rcParams["axes.edgecolor"]

    fig = plt.figure(figsize=(7, 7))
    bax = brokenaxes(
        ylims=((13, 19), (56, 64), (84, 91), (98, 102)),
        hspace=0.2,
        fig=fig,
        diag_color=current_color,
    )

    for i, (sti, data) in enumerate(accs_sti_dict.items()):
        acc_means = {k: np.mean(v) * 100 for k, v in data.items()}

        bax.plot(
            acc_means.keys(),
            acc_means.values(),
            marker="o",
            label=sti,
            color=category_colors[i],
        )

    bax.set_xlabel("Timesteps", fontsize=18, labelpad=25)

    bax.set_ylabel("Accuracy (%)", fontsize=18, labelpad=40)
    bax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
    bax.vlines(23, 13, 102, colors="grey", linestyles="dashed")

    bax.tick_params(axis="both", labelsize=14)

    plt.savefig(
        fname.fig_acc,
        bbox_inches="tight",
    )
    plt.show()
    print("done")
