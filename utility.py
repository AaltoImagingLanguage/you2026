import pandas as pd
import pickle
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import importlib
import matplotlib.pyplot as plt
from config import fname
import torchvision.models as models
from Levenshtein import distance as DL
import torch.nn.functional as F
import matplotlib.patches as mpatches
from config import n, category_colors
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import matplotlib.gridspec as gridspec
import mne
from scipy import sparse
from config import fname

import torchvision
import torchvision.transforms as transforms

with open(fname.word2idx_dir, "rb") as file:
    word2idx10k = pickle.load(file)

stimuli = pd.read_csv(f"{fname.data_dir}/stimuli.csv")
stimuli = stimuli[stimuli["target"] == "0"]

transform = torchvision.transforms.Compose(
    [
        # torchvision.transforms.CenterCrop(224),
        transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AverageMeter(object):
    """Computes and stores the average and current value for four types"""

    def __init__(self):
        # Initialize storage for four types
        self.reset()

    def reset(self):
        # Reset the values for each type
        self.val = [0] * 4  # Current values for each type
        self.avg = [0] * 4  # Average values for each type
        self.sum = [0] * 4  # Sum values for each type
        self.count = [0] * 4  # Count values for each type

    def update(self, val, n=1, type_idx=0):
        """Updates the values for a given type (0, 1, 2, or 3)"""
        # Update the current value for the given type index
        self.val[type_idx] = val / n

        # Update the sum and count for the given type index
        self.sum[type_idx] += val
        self.count[type_idx] += n

        # Recalculate the average for the given type index
        self.avg[type_idx] = self.sum[type_idx] / self.count[type_idx]


# def get_real_label(
#     stimulus,
# ):

#     data_series = stimuli[stimuli["stimuli"] == stimulus.upper()]
#     base = data_series["base"].values[0]
#     stimulus_type = data_series["type"].values[0][:3]
#     type_idx = data_series["index"].values[0] - 1
#     label = word2idx10k[base.lower()]

#     return label, stimulus_type, type_idx, base


def Acc(out, label, Print=0):
    # out and labels are tensors
    m = nn.Softmax(dim=1)
    out = m(out)
    out, label = out.cpu(), label.cpu()
    out, label = np.argmax(out.detach().numpy(), axis=1), label.numpy()
    score = 100 * np.mean(out == label)
    # print ('out', out)
    # print ('label', label)
    # print ('')
    return score


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = [correct[:k].sum().item() for k in topk]

        return res


def accuracy1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        values, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # index = torch.nonzero(correct[:, 0]).item()

        # probabilities of the topk predictions
        # softmax=torch.nn.Softmax(dim=None)
        # p_values=softmax(values)
        # prob=p_values.cpu()[0,index].item()
        #
        # prob=values
        res = [correct[:k].sum().item() for k in topk]

        return res, pred[0]


def get_model(pretrained=False, ngpus=0, model="cornet_r", trained_root=None, times=5):
    map_location = None if ngpus > 0 else "cpu"

    # Recurrent version of CORnet-Z. Better than CORnet-Z + recurrent but slow

    if model.lower() == "vgg16":
        from torchvision.models import vgg16

        model = vgg16(pretrained=pretrained)

    if trained_root:
        ckpt_data = torch.load(trained_root, weights_only=False)
        model.load_state_dict(ckpt_data["state_dict"])
        print("saved model at epoch: ", ckpt_data["epoch"])
        print("accuracy: ", ckpt_data["best_prec1"])

    if isinstance(model, nn.DataParallel):
        model = model.module  # remove DataParallel
    if ngpus > 0:
        model = model.cuda()
    return model


def compute_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_mb = (param_size) / 1024**2
    print("model size: {:.3f} MB".format(size_all_mb))
    return size_all_mb


def get_layerwise_avg_weights(model):
    layer_avg_weights = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_avg_weights[name] = param.data.mean().item()

    return layer_avg_weights


# Example usage:
# model = YourModel()
# print(get_layerwise_avg_weights(model))


def set_hyperparams(net, hps):

    num = net.number_of_pcoders

    assert len(hps) == num

    for n in range(1, num + 1):
        setattr(net, f"ffm{n}", torch.tensor(hps[n - 1]["ffm"], dtype=torch.float64))
        setattr(net, f"fbm{n}", torch.tensor(hps[n - 1]["fbm"], dtype=torch.float64))
        setattr(net, f"erm{n}", torch.tensor(hps[n - 1]["erm"], dtype=torch.float64))


def fetch_pmodel(net_name, hp_type, version=""):
    module = importlib.import_module(f"pnet.p{net_name}{version}")
    pnet_name = f"p{net_name}{version}{hp_type}HP"
    pmodel = getattr(module, pnet_name)
    return pmodel, pnet_name


def get_pmodel(
    pnet_path,
    backbone_path,
    hyperparams=None,
    model="vgg16",
    type_hp="Separate",
    version="v1",
):

    # from pc_model import pc_modelSameHP
    pnet, _ = fetch_pmodel(model, type_hp, version=version)

    net = get_model(model=model)
    if isinstance(net, nn.DataParallel):
        net = net.module
    if backbone_path:
        checkpoint = torch.load(backbone_path, weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value

        # Load the modified state_dict
        net.load_state_dict(new_state_dict)

    pc_model = pnet(
        backbone=net, build_graph=False, random_init=False, er_multiplier=0.01
    )

    if pnet_path:
        print(f"Loading weights from {pnet_path}")
        if hasattr(pc_model, "numbre_of_pcoders"):
            for n in range(pc_model.numbre_of_pcoders):
                checkpoints = torch.load(f"{pnet_path}{n+1}.pth", weights_only=False)
                checkpoint = checkpoints["pcoderweights"]
                getattr(pc_model, f"pcoder{n+1}").load_state_dict(checkpoint)
                # getattr( pc_model,f"pcoder{n+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items() if k!="C_sqrt"})

        else:
            for n in range(pc_model.number_of_pcoders):
                checkpoints = torch.load(f"{pnet_path}{n+1}.pth", weights_only=False)
                checkpoint = checkpoints["pcoderweights"]
                getattr(pc_model, f"pcoder{n+1}").load_state_dict(checkpoint)
                # getattr( pc_model,f"pcoder{n+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items() if k!="C_sqrt"})
                # getattr(pc_model,f"pcoder{n+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items() if k!="C_sqrt"})
        print("load pnet at epoch: ", checkpoints["epoch"])
        print("load pnet at loss: ", checkpoints["loss"])
    if hyperparams is not None:
        set_hyperparams(pc_model, hyperparams)

    return pc_model.eval()


def load_pnet(
    pnet_path,
    backbone_path,
    ff_multiplier,
    fb_multiplier,
    er_multiplier,
    model="cornet_z",
    type_hp="Same",
    version="",
    build_graph=False,
    random_init=False,
    hyperparams=None,
):

    # from pc_model import pc_modelSameHP
    pnet, _ = fetch_pmodel(model, type_hp, version=version)

    net = get_model(model=model)
    if isinstance(net, nn.DataParallel):
        net = net.module
    if backbone_path:
        checkpoint = torch.load(backbone_path, weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value

        # Load the modified state_dict
        net.load_state_dict(new_state_dict)

    pc_model = pnet(
        backbone=net,
        build_graph=build_graph,
        random_init=random_init,
        ff_multiplier=ff_multiplier,
        fb_multiplier=fb_multiplier,
        er_multiplier=er_multiplier,
    )

    if pnet_path:
        print(f"Loading weights from {pnet_path}")
        if hasattr(pc_model, "numbre_of_pcoders"):
            for n in range(pc_model.numbre_of_pcoders):
                checkpoints = torch.load(f"{pnet_path}{n+1}.pth", weights_only=False)
                pc_dict = checkpoints["pcoderweights"]

                if "C_sqrt" not in pc_dict:
                    pc_dict["C_sqrt"] = torch.tensor(-1, dtype=torch.float)
                getattr(pc_model, f"pcoder{n+1}").load_state_dict(pc_dict)
                # getattr( pc_model,f"pcoder{n+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items() if k!="C_sqrt"})

        else:
            for n in range(pc_model.number_of_pcoders):
                checkpoints = torch.load(f"{pnet_path}{n+1}.pth", weights_only=False)
                pc_dict = checkpoints["pcoderweights"]
                if "C_sqrt" not in pc_dict:
                    pc_dict["C_sqrt"] = torch.tensor(-1, dtype=torch.float)
                getattr(pc_model, f"pcoder{n+1}").load_state_dict(pc_dict)
                # getattr( pc_model,f"pcoder{n+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items() if k!="C_sqrt"})
                # getattr(pc_model,f"pcoder{n+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items() if k!="C_sqrt"})
        print("load pnet at epoch: ", checkpoints["epoch"])
        print("load pnet at loss: ", checkpoints["loss"])
        print("load pnet optimizer: ", checkpoints["optimizer"])
    if hyperparams is not None:
        set_hyperparams(pc_model, hyperparams)

    return pc_model.eval()


# def plot_output(output, stimulus, base, ind, n_units=10, t_step=0):
#     with open("../data/word2idx.pkl", "rb") as file:
#         word2idx10k = pickle.load(file)

#     with torch.no_grad():
#         values, pred = output.topk(
#             n_units, dim=1, largest=True, sorted=True
#         )  # perd=[1,10], output:[1,100]

#     predicted_words = []
#     for index in pred[0]:
#         if index in word2idx10k.values():
#             word = [k for k, v in word2idx10k.items() if v == index]
#             predicted_words.extend(word)


#     colors = [
#         "orange" if word == base.lower() else "skyblue" for word in predicted_words
#     ]
#     plt.figure(figsize=(6, 4))
#     plt.bar(predicted_words, values.cpu().numpy().flatten(), color=colors)
#     plt.title(f"{ind} Stimulus: {stimulus.lower()} Base: {base.lower()}")
#     plt.xlabel(f"Word class units (top {n_units})")
#     plt.ylabel("Unit activation")
#     plt.xticks(rotation=45)
#     plt.savefig(
#         f"figures/out_bb1/{ind}_{stimulus}_{n_units}_prediction_RL1_t_step{t_step}.png",
#         bbox_inches="tight",
#     )
#     plt.savefig(
#         f"figures/out_bb1/{ind}_{stimulus}_{n_units}_prediction_RL1_t_step{t_step}.pdf",
#         bbox_inches="tight",
#     )
# def plot_output(outputs, stimulus, base, ind, n_units=10):

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

#     # Process both time steps
#     outputs
#     t_steps = [0, 23]

#     for ax, output, t_step in zip(
#         axes,
#         outputs,
#         t_steps,
#     ):
#         with torch.no_grad():
#             values, pred = output.topk(
#                 n_units, dim=1, largest=True, sorted=True
#             )  # pred=[1,10], output:[1,100]

#         predicted_words = []
#         for index in pred[0]:
#             if index in word2idx10k.values():
#                 word = [k for k, v in word2idx10k.items() if v == index]
#                 predicted_words.extend(word)

#         colors = [
#             "orange" if word == base.lower() else "skyblue" for word in predicted_words
#         ]

#         ax.bar(predicted_words, values.cpu().numpy().flatten(), color=colors)
#         ax.set_title(
#             f"{"w/" if t_step else "w/o"} feedback: {stimulus.lower()} → {predicted_words[0].lower()}"
#         )
#         ax.set_xlabel(f"Word class units (top {n_units})")
#         if t_step == 0:
#             ax.set_ylabel("Unit activation")
#         ax.tick_params(axis="x", rotation=45)

#     plt.suptitle(
#         f"Stimulus: {stimulus.lower()}; Base: {base.lower()}",
#     )
#     plt.tight_layout()

#     plt.savefig(
#         f"figures/out_bb1/{ind}_{stimulus}_{n_units}_prediction_RL1_comparison.png",
#         bbox_inches="tight",
#     )
#     plt.savefig(
#         f"figures/out_bb1/{ind}_{stimulus}_{n_units}_prediction_RL1_comparison.pdf",
#         bbox_inches="tight",
#     )
#     plt.close()


def plot_output(outputs, stimulus, base, ind, n_units=10):

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    t_steps = [0, 23]

    for ax, output, t_step in zip(axes, outputs, t_steps):
        with torch.no_grad():
            values, pred = output.topk(n_units, dim=1, largest=True, sorted=True)

        predicted_words = []
        for index in pred[0]:
            if index in word2idx10k.values():
                word = [k for k, v in word2idx10k.items() if v == index]
                predicted_words.extend(word)

        colors = [
            "orange" if word == base.lower() else "skyblue" for word in predicted_words
        ]

        y_pos = range(len(predicted_words))
        vals = values.cpu().numpy().flatten()

        # Draw lines
        ax.hlines(y_pos, 0, vals, colors=colors, alpha=0.5, linewidth=2)
        # Draw dots
        ax.scatter(
            vals, y_pos, color=colors, s=60, zorder=3, edgecolors="black", linewidth=0.5
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(predicted_words)
        ax.invert_yaxis()
        ax.set_xlabel("Unit activation", fontsize=13)
        if t_step == 0:
            ax.set_ylabel(f"Lexical units (top {n_units})", fontsize=13)
        label = "w/" if t_step else "w/o"
        ax.set_title(f"{label} feedback", fontsize=13)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.suptitle(f"Stimulus: {stimulus.lower()}; Base: {base.lower()}", fontsize=13)
    plt.tight_layout()

    plt.savefig(
        f"figures/out_bb1/{ind}_{stimulus}_{n_units}_RL1_comparison.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        f"figures/out_bb1/{ind}_{stimulus}_{n_units}_RL1_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close()


def get_sti_rank(outputs, base, n_units=1000):
    with torch.no_grad():
        values, pred = outputs.topk(n_units, dim=1, largest=True, sorted=True)
    indices = []
    for i, target in enumerate(base):
        targetindex = word2idx10k[target.lower()]
        idx = (pred[i] == targetindex).nonzero(as_tuple=True)[0].item()
        indices.append(idx + 1)  # rank starts from 1
    return indices


def extract_activations(model, input_tensor, model_name="vgg16", pc=False):
    activations = {}

    # Define a hook function
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().contiguous()

        return hook

    # Register hooks on the relevant layers

    # model.backbone.V1.register_forward_hook(get_activation('v1'))
    # model.backbone.V2.register_forward_hook(get_activation('v2'))
    # model.backbone.V4.register_forward_hook(get_activation('v4'))
    # model.backbone.IT.register_forward_hook(get_activation('it'))
    # model.backbone.decoder.flatten.register_forward_hook(get_activation('h'))
    # model.backbone.decoder.output.register_forward_hook(get_activation('out'))
    if model_name == "cornet_z":
        # model.V1.register_forward_hook(get_activation("v1"))
        # model.V2.register_forward_hook(get_activation("v2"))
        # model.V4.register_forward_hook(get_activation("v4"))
        model.IT.register_forward_hook(get_activation("it"))
        model.decoder.flatten.register_forward_hook(get_activation("h"))
        # model.decoder.output.register_forward_hook(get_activation("out"))
    elif model_name == "vgg16":
        if pc:
            model.backbone.classifier[0].register_forward_hook(get_activation("fc1"))
            model.backbone.classifier[3].register_forward_hook(get_activation("fc2"))
            model.backbone.classifier[6].register_forward_hook(get_activation("out"))
        else:
            model.classifier[0].register_forward_hook(get_activation("fc1"))
            model.classifier[3].register_forward_hook(get_activation("fc2"))
            model.classifier[6].register_forward_hook(get_activation("out"))

    # Run a forward pass to collect activations
    _ = model(input_tensor)
    # compare pnet and backbone activations
    return activations


class VGG16_FT(nn.Module):
    def __init__(self, num_classes=1000, freeze_layers=True):
        super(VGG16_FT, self).__init__()
        # Load pre-trained VGG16
        vgg16 = models.vgg16(pretrained=True)

        # Features part (convolutional layers)
        self.features = vgg16.features

        # Freeze early layers (if requested)
        if freeze_layers:
            # Freeze all feature layers except the last few
            # This example freezes all but the last 4 convolutional layers
            for i, param in enumerate(self.features.parameters()):
                if (
                    i < 20
                ):  # VGG16 has 13 conv layers (each with weights and biases), vOT from 10th layer 9/13~3/4 (v1/v2/v4/it)
                    param.requires_grad = False

        # Classifier part (fully connected layers)
        self.avgpool = vgg16.avgpool

        # Replace the classifier for your task
        # Assuming you need a custom number of output classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # Initialize the new layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def compute_soft_target(y_word: str, vocab: list, temperature: float = 1):
    """
    Given a target word, compute a soft label distribution over the vocabulary
    using Levenshtein similarity.
    """
    sims = [np.exp(-DL(y_word, word) * temperature) for word in vocab]
    sims = np.array(sims)
    soft_target = sims / sims.sum()  # normalize to make a probability distribution
    return torch.tensor(soft_target, dtype=torch.float32)  # (1000,)


def soft_label_loss(
    logits: torch.Tensor, target_words: list, vocab: list, temperature: float = 1
):
    """
    logits: (batch_size, 1000)
    target_words: list of length batch_size, ground-truth words as strings
    vocab: list of 1000 real words (fixed)
    """
    device = logits.device

    # Precompute all soft targets
    soft_targets = [
        compute_soft_target(y_word, vocab, temperature) for y_word in target_words
    ]
    soft_targets = torch.stack(soft_targets).to(device)  # (batch_size, 1000)

    # Log probabilities
    log_probs = F.log_softmax(logits, dim=1)  # (batch_size, 1000)

    # Cross-entropy with soft targets
    loss = -torch.sum(soft_targets * log_probs, dim=1).mean()

    return loss


# if __name__ == "__main__":
# from torchvision.models import vgg16
# model = "vgg16"
# model1 = get_model(model=model, pretrained=False)
# model2 = get_model(model=model, pretrained=True)
# model3 = get_model(
#     model=model, trained_root="save/ffbackbone//vgg16_checkpoint.pth.tar"
# )
# # model_size = compute_model_size (model)
# mean_weights1 = get_layerwise_avg_weights(model1)
# mean_weights2 = get_layerwise_avg_weights(model2)
# mean_weights3 = get_layerwise_avg_weights(model3)
# print()
# model = cornet_z_FT()
# model = VGG16_FT()
from scipy.spatial import distance
import matplotlib.colors as mcolors


def plot_rdms(
    rdms,
    names=None,
    items=None,
    n_rows=1,
    cmap="viridis",
    title=None,
    size=3,
    vmin=0,
    vmax=100,
    categories=["RW", "RL1", "RL2", "RL3"],
    category_size=135,
    colorbar_label="correlation",
):
    """Plot one or more RDMs.

    Parameters
    ----------
    rdms : ndarray | list of ndarray
        The RDM or list of RDMs to plot. The RDMs can either be two-dimensional (n_items
        x n_items) matrices or be in condensed form.
    names : str | list of str | None
        For each given RDM, a name to show above it. Defaults to no names.
    items : list of str | None
        The each item (row/col) in the RDM, a string description. This will be displayed
        along the axes. Defaults to None which means the items will be numbered.
    n_rows : int
        Number of rows to use when plotting multiple RDMs at once. Defaults to 1.
    cmap : str
        Matplotlib colormap to use. See
        https://matplotlib.org/gallery/color/colormap_reference.html
        for all possibilities. Defaults to 'viridis'.
    title : str | None
        Title for the entire figure. Defaults to no title.

    Returns
    -------
    fig : matplotlib figure
        The figure produced by matplotlib

    """
    if not isinstance(rdms, list):
        rdms = [rdms]

    if isinstance(names, str):
        names = [names]
    if names is not None and len(names) != len(rdms):
        raise ValueError(
            f"Number of given names ({len(names)}) does not "
            f"match the number of RDMs ({len(rdms)})"
        )

    n_cols = int(np.ceil(len(rdms) / n_rows))
    fig = plt.figure(figsize=(size * n_cols, size * n_rows))

    ax = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * n_cols + col % n_cols
            if i < len(rdms):
                rdm = rdms[i]
                if rdm.ndim == 1:
                    rdm = distance.squareform(rdm)
                elif rdm.ndim > 2:
                    raise ValueError(f"Invalid shape {rdm.shape} for RDM")
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                im = ax[row, col].imshow(rdm, cmap=cmap, norm=norm)

                if names is not None:
                    name = names[i]
                    ax[row, col].set_title(name)
                if items is not None:
                    ax[row, col].set_xticks(np.arange(len(items)))
                    ax[row, col].set_xticklabels(items)
                    ax[row, col].set_yticks(np.arange(len(items)))
                    ax[row, col].set_yticklabels(items)
            else:
                ax[row, col].set_visible(False)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            ax[row, col].axis("off")

            # Add colorbar

            # Add category markers on the axes
            # Y-axis (left side)
            for i, cat in enumerate(categories):
                start = i * category_size
                end = (i + 1) * category_size

                # Create colored blocks for y-axis labels
                rect = mpatches.Rectangle(
                    (-n * 0.06, start),
                    n * 0.03,
                    category_size,
                    facecolor=category_colors[i],
                    edgecolor=None,
                    clip_on=False,
                )
                ax[row, col].add_patch(rect)

                # Add text labels
                ax[-1, 0].text(
                    -n * 0.07,
                    start + category_size / 2,
                    cat,
                    ha="right",
                    va="center",
                    fontsize=12,
                    # fontweight="bold"
                )

            # X-axis (bottom)
            for i, cat in enumerate(categories):
                start = i * category_size
                end = (i + 1) * category_size

                # Create colored blocks for x-axis labels
                rect = mpatches.Rectangle(
                    (start, n * 1.01),
                    category_size,
                    n * 0.03,
                    facecolor=category_colors[i],
                    edgecolor=None,
                    clip_on=False,
                )
                ax[row, col].add_patch(rect)

            # Set appropriate limits to show the labels
            ax[row, col].set_xlim(-n * 0.1, n * 1.05)
            ax[row, col].set_ylim(
                n * 1.05, -n * 0.1
            )  # Inverted y-axis for correct orientation
    # Add a shared colorbar
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im)
    cbar.set_label("Correlation Distance", rotation=270, labelpad=20)
    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    if title is not None:
        plt.suptitle(title)
    return fig


def plot_rdms1(
    rdms,
    names=None,
    title=None,
    cmap="viridis",
    vmin=0,
    vmax=2,
    figsize=(10, 5),
    category_size=135,
    categories=["RW", "RL1", "RL2", "RL3"],
    colorbar_label="Correlation Distance",
):
    """
    Plot multiple RDMs with concise label style using short vertical lines.
    No blank spaces between RDMs. Colorbar only shows min and max values.
    """
    if not isinstance(rdms, list):
        rdms = [rdms]

    n_rdms = len(rdms)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create placeholder for storing image object and last RDM axis
    im = None
    last_rdm_ax = None

    # Category names
    # categories = ["RW", "RL1", "RL2", "RL3"]
    # Matrix dimensions
    n = category_size * len(categories)  # Assuming square matrix (540×540)

    # Normalize colors consistently across all plots
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Create GridSpec with no horizontal spacing between plots
    gs = GridSpec(1, n_rdms, figure=fig, wspace=0)

    # Plot each RDM
    for i in range(n_rdms):
        # Create axis for this RDM
        ax = fig.add_subplot(gs[0, i])
        rdm = rdms[i]
        if rdm.ndim == 1:
            rdm = distance.squareform(rdm)
        elif rdm.ndim > 2:
            raise ValueError(f"Invalid shape {rdm.shape} for RDM")
        # Plot the RDM
        im = ax.imshow(rdm, cmap=cmap, norm=norm)
        last_rdm_ax = ax  # Store reference to the last RDM axis

        # Remove axis ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add main title and subtitle if provided
        if names is not None and i < len(names):
            ax.set_title(names[i])

        # Add vertical lines to mark category boundaries on Y-axis (left)
        # Only for the first RDM

        for j in range(1, 4):  # Add 3 lines at category boundaries
            boundary_pos = j * category_size
            # Draw a line segment instead of using axhline
            ax.plot(
                [-n * 0.02, 0],
                [boundary_pos, boundary_pos],
                color="black",
                linewidth=1.5,
                clip_on=False,
            )

        # Add horizontal lines to mark category boundaries on X-axis (bottom)
        for j in range(1, 4):  # Add 3 lines at category boundaries
            boundary_pos = j * category_size
            # Draw a line segment instead of using axvline
            ax.plot(
                [boundary_pos, boundary_pos],
                [n, n * 1.02],
                color="black",
                linewidth=1.5,
                clip_on=False,
            )

        # Add category labels on Y-axis (only for the first RDM)
        if i == 0:
            for j, cat in enumerate(categories):
                mid_pos = j * category_size + category_size // 2
                ax.text(
                    -n * 0.03,
                    mid_pos,
                    cat,
                    ha="right",
                    va="center",
                    fontsize=12,
                    # fontweight="bold",
                )

            # Add category labels on X-axis (for all RDMs)
            # Only add labels for leftmost and rightmost categories if not the first RDM
            # This avoids overlapping labels between adjacent RDMs
            for j, cat in enumerate(categories):
                # For RDMs after the first one, only label the leftmost and rightmost categories
                if i > 0 and j > 0 and j < len(categories) - 1:
                    continue

                mid_pos = j * category_size + category_size // 2
                ax.text(
                    mid_pos,
                    n * 1.03,
                    cat,
                    ha="center",
                    va="top",
                    fontsize=12,
                    # fontweight="bold",
                )

        # Set limits to show labels
        if i == 0:
            ax.set_xlim(-n * 0.05, n)
        else:
            ax.set_xlim(0, n)
        ax.set_ylim(n * 1.05, -n * 0.02)  # Inverted y-axis

    # First draw to update the figure and get the correct RDM axis position
    fig.canvas.draw()

    # Get the position of the last RDM plot in figure coordinates
    bbox = last_rdm_ax.get_position()

    # Create the colorbar axes with EXACTLY the same height and vertical position
    cbar_ax = fig.add_axes(
        [
            bbox.x1 + 0.02,  # Position right after the last RDM
            bbox.y0 + 0.037,  # Same bottom position
            0.02,  # Width of the colorbar
            bbox.height - 0.05,  # EXACT same height as the RDM
        ]
    )

    # Add the colorbar with only min and max ticks
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(size=0)  #####################hide ticks
    # Set only min and max ticks on the colorbar
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])

    cbar.set_label(
        colorbar_label,
        rotation=270,
        labelpad=-5,
    )
    if title is not None:
        plt.suptitle(title, fontsize=16)
    return fig


def plot_meg_rdms(
    rdms,
    names=None,
    items=None,
    n_rows=1,
    cmap="viridis",
    vmin=0,
    vmax=100,
    plot_size=2,
    colorbar_label=None,
    title=None,
):
    """Plot one or more RDMs.

    Parameters
    ----------
    rdms : ndarray | list of ndarray
        The RDM or list of RDMs to plot. The RDMs can either be two-dimensional (n_items
        x n_items) matrices or be in condensed form.
    names : str | list of str | None
        For each given RDM, a name to show above it. Defaults to no names.
    items : list of str | None
        The each item (row/col) in the RDM, a string description. This will be displayed
        along the axes. Defaults to None which means the items will be numbered.
    n_rows : int
        Number of rows to use when plotting multiple RDMs at once. Defaults to 1.
    cmap : str
        Matplotlib colormap to use. See
        https://matplotlib.org/gallery/color/colormap_reference.html
        for all possibilities. Defaults to 'viridis'.
    title : str | None
        Title for the entire figure. Defaults to no title.

    Returns
    -------
    fig : matplotlib figure
        The figure produced by matplotlib

    """
    if not isinstance(rdms, list):
        rdms = [rdms]

    if isinstance(names, str):
        names = [names]
    if names is not None and len(names) != len(rdms):
        raise ValueError(
            f"Number of given names ({len(names)}) does not "
            f"match the number of RDMs ({len(rdms)})"
        )

    # Define the condition labels
    condition_labels = ["RW", "RL1", "RL2", "RL3"]

    n_cols = int(np.ceil(len(rdms) / n_rows))
    fig = plt.figure(figsize=(plot_size * n_cols, plot_size * n_rows))

    ax = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * n_cols + col % n_cols
            if i < len(rdms):
                rdm = rdms[i]
                if rdm.ndim == 1:
                    rdm = distance.squareform(rdm)
                elif rdm.ndim > 2:
                    raise ValueError(f"Invalid shape {rdm.shape} for RDM")
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                im = ax[row, col].imshow(rdm, cmap=cmap, norm=norm)

                if names is not None:
                    name = names[i]
                    ax[row, col].set_title(name)

                # Set x and y axis labels to condition labels
                ax[row, col].set_xticks(np.arange(len(condition_labels)))
                ax[row, col].set_xticklabels(condition_labels)
                ax[row, col].set_yticks(np.arange(len(condition_labels)))
                ax[row, col].set_yticklabels(condition_labels)
                ax[row, col].tick_params(size=0)

                # Turn off grid if it's on
                ax[row, col].grid(False)

                # Override with custom items if provided
                if items is not None:
                    ax[row, col].set_xticks(np.arange(len(items)))
                    ax[row, col].set_xticklabels(items)
                    ax[row, col].set_yticks(np.arange(len(items)))
                    ax[row, col].set_yticklabels(items)
            else:
                ax[row, col].set_visible(False)

    # Add colorbar with custom label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        colorbar_label,
        rotation=270,
        labelpad=10,
    )
    if title is not None:
        plt.suptitle(title, x=0.12, fontweight="bold", fontsize=12)
    return fig


def plot_rdms_model(
    rdms,
    condition_labels,
    names=None,
    n_rows=1,
    cmap="viridis",
    vmin=0,
    vmax=100,
    plot_size=2,
    main_titles=None,
    colorbar_label=None,
    title=None,
):
    """Plot one or more RDMs.

    Parameters
    ----------
    rdms : ndarray | list of ndarray
        The RDM or list of RDMs to plot. The RDMs can either be two-dimensional (n_items
        x n_items) matrices or be in condensed form.
    condition_labels : list of str
    names : str | list of str | None
        For each given RDM, a name to show above it. Defaults to no names.
    n_rows : int
        Number of rows to use when plotting multiple RDMs at once. Defaults to 1.
    cmap : str
        Matplotlib colormap to use. See
        https://matplotlib.org/gallery/color/colormap_reference.html
        for all possibilities. Defaults to 'viridis'.
    title : str | None
        Title for the entire figure. Defaults to no title.

    Returns
    -------
    fig : matplotlib figure
        The figure produced by matplotlib

    """
    if not isinstance(rdms, list):
        rdms = [rdms]

    if isinstance(names, str):
        names = [names]

    n_cols = int(np.ceil(len(rdms) * len(names)))
    fig = plt.figure(
        figsize=(plot_size * n_cols, plot_size * n_rows + 0.1),
    )

    # Create main grid: [subplots] + [colorbar]
    main_gs = GridSpec(1, 2, figure=fig, width_ratios=[n_cols, 0.05], wspace=0.05)

    subplot_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, main_gs[0], wspace=0.15, hspace=0.1
    )

    # Convert to hierarchical dict format
    data_dict = {}
    for i, (group_data, group_name) in enumerate(
        zip(rdms, ["PCoder 1", "PCoder 2", "PCoder 3"])
    ):
        data_dict[group_name] = {f"Pair 1": group_data}

    # Plot subplots
    axes = []
    subplot_idx = 0
    main_group_positions = {}  # Track positions for main titles
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for main_group_idx, (main_group, pairs) in enumerate(data_dict.items()):
        group_start_idx = subplot_idx

        for pair_idx, (pair_name, pair_data) in enumerate(pairs.items()):
            pair_axes = []

            # Plot each subplot in the pair
            for data_idx, data in enumerate(pair_data):
                ax = fig.add_subplot(subplot_gs[0, subplot_idx])
                if data.ndim == 1:
                    data = distance.squareform(data)
                elif data.ndim > 2:
                    raise ValueError(f"Invalid shape {data.shape} for RDM")
                im = ax.imshow(data, cmap=cmap, norm=norm)
                # Set x and y axis labels to condition labels
                ax.set_xticks(np.arange(len(condition_labels)))
                ax.set_xticklabels(condition_labels)
                ax.set_yticks(np.arange(len(condition_labels)))
                if subplot_idx == 0:
                    ax.set_yticklabels(condition_labels)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(size=0)
                ax.set_title(names[data_idx])
                ax.grid(False)
                pair_axes.append(ax)
                axes.append(ax)
                subplot_idx += 1
            # Add pair title (Level 1: pair level)
            if len(pairs) > 1:  # Only show pair titles if multiple pairs
                pair_center_x = (
                    pair_axes[0].get_position().x0 + pair_axes[1].get_position().x1
                ) / 2
                fig.text(
                    pair_center_x,
                    0.88,
                    pair_name,
                    ha="center",
                    va="bottom",
                    # fontsize=11, style='italic',
                    transform=fig.transFigure,
                )
        # Store main group position info
        group_end_idx = subplot_idx - 1
        main_group_positions[main_group] = (group_start_idx, group_end_idx)

    # Add main group titles (Level 2: main level)
    for main_group, (start_idx, end_idx) in main_group_positions.items():
        if start_idx == end_idx:
            center_x = (
                axes[start_idx].get_position().x0
                + axes[start_idx].get_position().width / 2
            )
        else:
            center_x = (
                axes[start_idx].get_position().x0 + axes[end_idx].get_position().x1
            ) / 2

        title = (
            main_titles[list(main_group_positions.keys()).index(main_group)]
            if main_titles
            else main_group
        )
        fig.text(
            center_x,
            0.95,
            title,
            ha="center",
            va="bottom",
            # fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

    # Add shared colorbar
    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, main_gs[1])
    cbar_ax = fig.add_subplot(cbar_gs[0])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(colorbar_label, rotation=270, labelpad=10)
    return fig


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def cluster_vertices(x):

    x0 = x.copy()  # (68,135)->(vertices, n_epochs)
    x1 = x.copy()
    model = KMeans(n_init="auto", random_state=0)
    visualizer = KElbowVisualizer(
        model,
        k=(2, 13),
    )
    visualizer.fit(x0)
    n_clusters = visualizer.elbow_value_
    plt.close()
    # print('n_clusters:',n_clusters)
    if visualizer.elbow_value_ is None:
        n_clusters = 10
        # print('visualizer.elbow_value_ is None')
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
    # print('model = KMeans(n_clusters=n_clusters,n_init=auto  DONE')
    model.fit(x1)
    # print('model.fit(x1)  DONE')
    yhat = model.predict(x1)  # (68,)
    # print('yhat = model.predict(x1)  DONE')
    clusters = np.unique(yhat)  # [0,1,2,3,4,5]
    x_clusters = np.zeros([x.shape[1], clusters.shape[0]])  # n_trials*vertex
    for n, cluster in enumerate(clusters):
        row_ix = np.where(yhat == cluster)
        data = x[row_ix, :].transpose()[:, :, 0]
        # data=x[row_ix, :].reshape([x[row_ix, :].shape[2],x[row_ix, :].shape[1]])

        cluster_var = data.var(0)
        idx = np.where(cluster_var == cluster_var.max())
        x_clusters[:, n] = x[idx, :]

    return x_clusters


def scale_patterns(x):
    from sklearn.preprocessing import StandardScaler

    trial, v_x = x.shape
    x_vector = x.reshape(trial * v_x, 1)
    scaler = StandardScaler()
    x_vec_scaled = scaler.fit_transform(x_vector)
    x_scaled = x_vec_scaled.reshape(trial, v_x)

    return x_scaled


def stc_baseline_correction(X, stc, tmin, tmax):
    time_dim = len(stc.times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    # baseline_timepoints = X.times[np.where(X.times==tmin):np.where(X.times==tmax)]
    # Convert tmin/tmax to sample indices
    tmin, tmax = np.searchsorted(stc.times, [tmin, tmax])

    baseline_timepoints = stc.times[tmin:tmax]

    baseline_mean = X[:, tmin:tmax].mean(1)

    baseline_mean_mat = np.repeat(
        baseline_mean.reshape([len(baseline_mean), 1]), time_dim, axis=1
    )
    corrected_stc = X - baseline_mean_mat
    return corrected_stc


def xval_score(X, y, splits=10, avg_scores=True):
    """
    Cross-validated ridge regression between X and y.
    """

    not_nan_rows = ~np.isnan(X.mean(1))
    X = X[not_nan_rows]
    y = y[not_nan_rows]

    # train model on full data to save coefs
    alphas = np.logspace(-3, 3, 15)
    mdl = RidgeCV(alphas=alphas)
    mdl.fit(X, y)
    coefs_to_save = mdl.coef_

    pipe = Pipeline([("regress", RidgeCV(alphas=alphas))])

    mdl = MultiOutputRegressor(pipe)

    scores_corr = []
    scores_mse = []
    kFold = KFold(n_splits=splits, random_state=42, shuffle=True)
    for train_index, test_index in kFold.split(X):
        X_train, X_test, y_train, y_test = (
            X[train_index],
            X[test_index],
            y[train_index],
            y[test_index],
        )
        mdl.fit(X_train, y_train)
        predictions = [x.predict(X_test) for x in mdl.estimators_]
        scores_corr.append(
            [
                np.corrcoef(y_test[:, i], pred)[0, 1]
                for i, pred in enumerate(predictions)
            ]
        )
        scores_mse.append(
            [
                mean_squared_error(y_test[:, i], pred)
                for i, pred in enumerate(predictions)
            ]
        )

    if avg_scores:
        return (
            np.asarray(scores_corr).mean(0),
            np.asarray(scores_mse).mean(0),
            coefs_to_save,
        )  # mean over splits so shape is just (y.shape[1],), i.e. (num_targets,)
    else:
        return (
            np.asarray(scores_corr).T,
            np.asarray(scores_mse).T,
            coefs_to_save,
        )  # shape (num_targets, num_splits)


def create_labels_adjacency_matrix(labels, src_to):
    adjacency = mne.spatial_src_adjacency(src_to)
    n_labels = len(labels)
    # Initialize an empty adjacency matrix for labels
    label_adjacency_matrix = np.zeros((n_labels, n_labels))
    labels1 = [
        label.restrict(src_to, name=None) for label in labels
    ]  # Restrict a label to a source space.

    # Loop through each label and find its vertices
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels1):
            if i != j:
                # Check if any vertices of label1 are adjacent to vertices of label2
                # (you need to adapt this depending on how you define adjacency)

                label1_vertices = np.in1d(adjacency.row, label1.vertices)
                label2_vertices = np.in1d(adjacency.col, label2.vertices)
                label1_vertices0 = np.in1d(adjacency.row, label2.vertices)
                label2_vertices0 = np.in1d(adjacency.col, label1.vertices)
                if np.any(label1_vertices & label2_vertices) or np.any(
                    label1_vertices0 & label2_vertices0
                ):
                    label_adjacency_matrix[i, j] = 1
            else:
                label_adjacency_matrix[i, j] = 1
    label_adjacency_matrix = sparse.coo_matrix(label_adjacency_matrix)
    return label_adjacency_matrix


def plot_cluster_label(
    cluster, rois, brain, time_index=None, color="black", width=1, alpha=1
):

    cluster_time_index, cluster_vertex_index = cluster

    # A cluster is defined both in space and time. If we want to plot the boundaries of
    # the cluster in space, we must choose a specific time for which to show the
    # boundaries (as they change over time).
    if time_index is None:
        time_index, n_vertices = np.unique(
            cluster_time_index,
            return_counts=True,
        )
        time_index = time_index[np.argmax(n_vertices)]

    # Select only the vertex indices at the chosen time
    draw_vertex_index = [
        v for v, t in zip(cluster_vertex_index, cluster_time_index) if t == time_index
    ]

    for index in draw_vertex_index:
        roi = rois[index]
        brain.add_label(roi, borders=width, color=color, alpha=alpha)
