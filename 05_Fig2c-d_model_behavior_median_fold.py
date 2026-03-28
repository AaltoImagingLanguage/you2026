# %%

import webdataset as wds
from plotnine import *
import json
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
    k
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
parser.add_argument(
    "--fold", type=int, default=5, help="which fold to use for validation (0-4), or 5 to use the median hyperparameters across folds"
)

args, _ = parser.parse_known_args()

hps_cvs=[]
for n_fold in range(k):
    hps_data = torch.load(fname.hps_ckpt(n_fold=n_fold), weights_only=False)
    hps = hps_data["hps"]
    hps_cvs.append(hps)

median_hps=np.median(hps_cvs, axis=0).round(3)

print(median_hps)
n_fold=args.fold

def get_acc_list(net, max_timestep):

    net.eval()

    stimulus_acc = {sti: {} for sti in sti_types}
    sti_out = {"id": [], "group": [], "w/o feedback": [], "w/ feedback": []}
    tstart = time.time()
    df_long = pd.DataFrame()  # Placeholder for output activations dataframe

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
                compute_soft_target(y_word, words1k) for y_word in target_words
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

seed=0

fold_of_key = json.load(open(fname.cv_folds))["fold_of_key"]

def _norm_key(k):
    return k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k

def _letters_only_token(k):
    # take lexical part before final '_' and keep only letters (canonicalized)
    nk = _norm_key(k)
    if not nk:
        return ""
    base = nk.split("_")[0]
    # keep only alphabetic characters and uppercase for canonical form
    letters = "".join(ch for ch in base if ch.isalpha())
    return letters.upper()

# Build a letters-only -> fold map for fast fallback lookups
fold_letters_map = {}
for fk, fv in fold_of_key.items():
    tok = _letters_only_token(fk)
    if tok:
        # prefer first-seen mapping; assumes consistency across archives
        fold_letters_map.setdefault(tok, fv)


def _fold_for_sample(s):
    nk = _norm_key(s["__key__"])
    # exact lookup first
    f = fold_of_key.get(nk)
    if f is not None:
        return f
    # fallback: match by letters-only token (e.g. 'MAMOITUS' from 'MAMOITUS_000005')
    tok = _letters_only_token(nk)
    if tok:
        return fold_letters_map.get(tok)
    return None

tstart = datetime.now()
if args.compute:
    
    base = (
        wds.WebDataset(f"{fname.dataset_dir}/stimuli.tar")
        .decode("pil")
        .map_dict(png=transform)
    )

    val_conditions = {"RW", "RL2PW", "RL3PW"}

    def _get_condition_from_json(j):
        # json payload may contain the condition under different keys
        if isinstance(j, (bytes, bytearray)):
            try:
                j = json.loads(j.decode("utf-8"))
            except Exception:
                return None
        cond = None
        for key in ("condition", "sti_type", "set", "type"):
            cond = j.get(key) if isinstance(j, dict) else None
            if cond is not None:
                break
        if isinstance(cond, (bytes, bytearray)):
            try:
                cond = cond.decode("utf-8")
            except Exception:
                pass
        return cond

    def _is_val_sample(s):
        # If metadata indicates one of the val_conditions -> include
        j = s.get("json") if isinstance(s, dict) else s[2]
        cond = _get_condition_from_json(j)
        if cond in val_conditions:
            return True
        # Otherwise include if the sample's fold equals the chosen val fold
        return (_fold_for_sample(s) == n_fold)
    if n_fold>=len(hps_cvs):
        hps=median_hps
        val_wrdset = (
            wds.WebDataset(f"{fname.dataset_dir}/stimuli.tar")
            .decode("pil")
            .map_dict(png=transform)
            .to_tuple("png", "cls", "json")
        )
    else:   
        hps = hps_cvs[n_fold]
        val_wrdset = (
            base
            .select(_is_val_sample)
            .to_tuple("png", "cls", "json")
        )

    val_loader = torch.utils.data.DataLoader(
        val_wrdset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
   
   
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
        fname.accs(n_fold=n_fold),
        "wb",
    ) as f:
        pickle.dump(accs_sti_dict, f)
    if n_fold==len(hps_cvs):
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
        fname.accs(n_fold=n_fold),
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
    # from brokenaxes import brokenaxes

    current_color = plt.rcParams["axes.edgecolor"]

    fig = plt.figure(figsize=(7, 7))
  
    if n_fold<len(hps_cvs):

        for i, (sti, data) in enumerate(accs_sti_dict.items()):
            acc_means = {k: np.mean(v) * 100 for k, v in data.items()}
            print()

            plt.plot(
                acc_means.keys(),
                acc_means.values(),
                marker="o",
                label=sti[:3],
                color=category_colors[i],
            )

        plt.xlabel("Timesteps", fontsize=22, labelpad=25)

        plt.ylabel("Accuracy (%)", fontsize=22, labelpad=40)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=222)
        # bax.vlines(23, 13, 102, colors="grey", linestyles="dashed")

        plt.tick_params(axis="both", labelsize=22)
    else: 
        
        ylims=((8,23), (50, 67), (76, 98), (99, 101))
        from brokenaxes import brokenaxes
        bax = brokenaxes(
            ylims=ylims,
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
                label=sti[:3],
                color=category_colors[i],
            )
            print(acc_means.values())

        bax.set_xlabel("Timesteps", fontsize=20, labelpad=25)

        bax.set_ylabel("Accuracy (%)", fontsize=20, labelpad=40)
        bax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
        bax.vlines(25, 8, 101, colors="grey", linestyles="dashed",  linewidth=2)

        bax.tick_params(axis="both", labelsize=18)

    plt.savefig(
        fname.fig_acc_fold(n_fold=n_fold),
        bbox_inches="tight",
    )
    plt.show()
    print("done")
