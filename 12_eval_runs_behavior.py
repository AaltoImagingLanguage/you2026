#########################################################################
# Per-stimulus classification of one training run (Fig 3A, S2 Table, S2 Fig).
#
# For every non-catch stimulus (540; 135 per condition) and timestep
# t = 0..max_timestep-1, stores whether the top-1 output is the base word for
#   "PC median"   the predictive-coding model with the median hyperparameters of the
#                 five folds (rounded to 3 decimals, as for Fig 3B)
#   "PC fold f"   the same model with the hyperparameters of fold f (validation fold f)
#   "PC nested f" the hyperparameters of the nested cross-validation with outer test
#                 fold f (trained on three folds, selected on fold (f + 1) mod 5); for the
#                 RL2/RL3 stimuli of fold f these are strictly held-out predictions
#   "RNN", "GRU"  the recurrent controls (07_train_rnn_gru_controls.py)
#
# --run orig   original model: all models  -> seeds/item_preds_orig.csv
# --run SEED   retrained seed (5, 15, 25, 35): the PC models -> seeds/item_preds_seedSEED.csv,
#              with --rnn_gru the RNN/GRU controls -> seeds/item_preds_seedSEED_rnn_gru.csv
# All runs use the original fold assignment (cv_fold_assignments_RL23.json).
# The paired tests and the summaries across runs are in 13_runs_behavior_stats.py.
#
#   python 12_eval_runs_behavior.py --run orig
# On 16-GB GPUs, memory builds up across models within one process, so
# evaluate each model in its own process and then merges:
#   python 12_eval_runs_behavior.py --run orig --n_models        # number of models
#   python 12_eval_runs_behavior.py --run orig --model_index i    # one part file
#   python 12_eval_runs_behavior.py --run orig --merge            # parts -> csv
#########################################################################
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds

from config import device, fname, k, max_timestep
from utility import get_model, get_pmodel, transform
from pnet.rvgg16_baseline import ClassifierRecurrentVGG16Baseline

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="orig, or a seed (5, 15, 25, 35)")
p.add_argument("--rnn_gru", action="store_true", help="seeds only: evaluate the RNN and GRU controls")
p.add_argument("--batchsize", type=int, default=60)
p.add_argument("--n_models", action="store_true", help="print the number of models and exit")
p.add_argument("--model_index", type=int, default=None,
               help="evaluate only this model and save it as a part file")
p.add_argument("--merge", action="store_true", help="merge the part files into the output")
args = p.parse_args()

orig = args.run == "orig"
tag = "" if orig else f"_seed{int(args.run)}"
backbone = fname.ff_ckpt if orig else str(fname.ff_ckpt).replace(".pth.tar", f"_soft{tag}.pth.tar")
pcoders = fname.pcoder_ckpt if orig else f"{fname.pnet_dir}/p_vgg16_Separate_v1{tag}_best_pc"
hps_paths = [str(fname.hps_ckpt_RL23(n_fold=f)).replace(".pth", f"{tag}.pth") for f in range(k)]
nested_paths = [str(fname.hps_ckpt_RL23(n_fold=(f + 1) % k)).replace(".pth", f"_test{f}{tag}.pth")
                for f in range(k)]
fold_file = Path(str(fname.cv_folds).replace(".json", "_RL23.json"))
out_dir = Path(f"{fname.data_dir}/behavior/seeds")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"item_preds{tag or '_orig'}{'_rnn_gru' if args.rnn_gru else ''}.csv"
fold_of_key = json.load(open(fold_file))["fold_of_key"]


def load_hps(path):
    return np.asarray(torch.load(path, weights_only=False)["hps"], float)


def pc_model(h):
    hp = [{"ffm": h[0], "fbm": h[1], "erm": h[3]},
          {"ffm": h[4], "fbm": h[5], "erm": h[7]},
          {"ffm": h[8], "fbm": h[9], "erm": h[11]}]
    return get_pmodel(pnet_path=pcoders, backbone_path=backbone, hyperparams=hp).to(device).eval()


def rnn_model(cell):
    ckpt = torch.load(f"{fname.net_dir}/rnn_baseline/reconrec_{cell}_bn128{tag}_best.pth",
                      weights_only=False, map_location=device)
    a = ckpt["args"]
    net = ClassifierRecurrentVGG16Baseline(
        backbone=get_model(model="vgg16", trained_root=backbone), cell_type=cell,
        use_topdown=bool(a["use_topdown"]), input_adapter=bool(a["input_adapter"]),
        residual_mix=bool(a["residual_mix"]), init_mix=a["init_mix"],
        bottleneck=a["bottleneck"], freeze_backbone=True).to(device)
    missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=False)
    assert all(m.startswith("backbone.") for m in missing) and not unexpected
    return net.eval()


def classify(net, name):
    loader = torch.utils.data.DataLoader(
        wds.WebDataset(f"{fname.dataset_dir}/stimuli.tar", shardshuffle=False)
        .decode("pil").map_dict(png=transform).to_tuple("__key__", "png", "cls", "json"),
        batch_size=args.batchsize, shuffle=False, num_workers=1)
    rows = []
    for keys, inputs, labels, meta in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            for t in range(max_timestep):
                pred = (net(inputs) if t == 0 else net(None)).argmax(1)
                for i, key in enumerate(keys):
                    rows.append((name, key, meta["word"][i], meta["base"][i], meta["type"][i][:3],
                                 fold_of_key.get(key, -1), t, int(pred[i] == labels[i])))
    print(f"{name}: {len(rows)} rows")
    return rows


def run(build, name):
    # build, classify and release one model; the PC networks keep their state tensors in
    # reference cycles, so the GPU is freed before the next model is built
    net = build()
    out = classify(net, name)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return out


# the models, in output order: (name, builder)
RNN_GRU = [(cell.upper(), lambda cell=cell: rnn_model(cell)) for cell in ("rnn", "gru")]
if args.rnn_gru:
    assert not orig, "--rnn_gru is for the seeds; the original run includes the RNN/GRU controls"
    MODELS = RNN_GRU
else:
    hps_folds = [load_hps(h) for h in hps_paths]
    median_hps = np.median(hps_folds, axis=0).round(3)
    MODELS = ([("PC median", lambda: pc_model(median_hps))]
              + [(f"PC fold {f}", lambda h=h: pc_model(h)) for f, h in enumerate(hps_folds)]
              + [(f"PC nested {f}", lambda path=path: pc_model(load_hps(path)))
                 for f, path in enumerate(nested_paths)]
              + (RNN_GRU if orig else []))
if args.n_models:
    print(len(MODELS))
    raise SystemExit
COLUMNS = ["model", "key", "word", "base", "condition", "fold", "t", "correct"]


def part_path(i):
    return out_path.with_name(f"{out_path.stem}.part{i:02d}.csv")


if args.model_index is not None:
    name, build = MODELS[args.model_index]
    pd.DataFrame(run(build, name), columns=COLUMNS).to_csv(part_path(args.model_index), index=False)
    print("saved ->", part_path(args.model_index))
    raise SystemExit
if args.merge:
    missing = [i for i in range(len(MODELS)) if not part_path(i).exists()]
    assert not missing, f"missing part files for models {missing}"
    df = pd.concat([pd.read_csv(part_path(i)) for i in range(len(MODELS))], ignore_index=True)
    assert list(df.model.unique()) == [name for name, _ in MODELS]
else:
    df = pd.DataFrame([row for name, build in MODELS for row in run(build, name)], columns=COLUMNS)
df.to_csv(out_path, index=False)
if args.merge:
    for i in range(len(MODELS)):
        part_path(i).unlink()
acc = df[df.t.isin([0, 25])].groupby(["model", "condition", "t"]).correct.mean().unstack() * 100
print(acc.round(1))
print("saved ->", out_path)
