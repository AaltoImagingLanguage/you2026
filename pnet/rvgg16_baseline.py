# pnet/rvgg16_baseline.py: parameter-matched RNN/GRU control model (S6 Table).
# Imported by 07_train_rnn_gru_controls.py (training), 08_eval_rnn_gru_controls.py
# (accuracy, S3 Fig) and 09_rnn_gru_representations.py (representations, S12 Fig).
# Reported configuration: use_topdown=False, input_adapter=False, residual_mix=True,
# init_mix=0.10, bottleneck=128 (the defaults of 07_train_rnn_gru_controls.py).
# Running this file builds a model on an untrained VGG16 and prints parameter counts.

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


CellType = Literal["rnn", "gru"]


def count_trainable_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    """Number of all parameters."""
    return sum(p.numel() for p in model.parameters())


def pcoder_feedback_param_count() -> int:
    """
    Trainable feedback/generative parameters in the current 3-PCoder VGG model:

    PCoder1: Linear(4096 -> 25088)
    PCoder2: Linear(4096 -> 4096)
    PCoder3: Linear(1000 -> 4096)

    Used to report the parameter matching (S6 Table).
    """
    pc1 = 4096 * 25088 + 25088
    pc2 = 4096 * 4096 + 4096
    pc3 = 1000 * 4096 + 4096
    return pc1 + pc2 + pc3


def _make_cell(cell_type: CellType, dim: int) -> nn.Module:
    if cell_type == "gru":
        return nn.GRUCell(input_size=dim, hidden_size=dim)
    if cell_type == "rnn":
        return nn.RNNCell(input_size=dim, hidden_size=dim, nonlinearity="tanh")
    raise ValueError(f"Unknown cell_type: {cell_type}")


def _build_cell(cell_type: CellType, dim: int, bottleneck: Optional[int]):
    """Return (cell, down, up). If bottleneck is set, the cell runs in the
    bottleneck space (down-project stage->b, cell, up-project b->stage), so the
    added recurrent capacity is small and the total added parameters match the
    PC feedback pathway (~123.67 M). Otherwise a full-width cell (down=up=None)."""
    if bottleneck is None:
        return _make_cell(cell_type, dim), None, None
    return (_make_cell(cell_type, bottleneck),
            nn.Linear(dim, bottleneck), nn.Linear(bottleneck, dim))


def _init_identity_if_square(linear: nn.Linear) -> None:
    """Initialize a square Linear layer close to identity."""
    with torch.no_grad():
        linear.weight.zero_()
        n = min(linear.weight.shape)
        linear.weight[:n, :n] = torch.eye(n, device=linear.weight.device)
        if linear.bias is not None:
            linear.bias.zero_()


def _init_gru_copy_bias(cell: nn.GRUCell) -> None:
    """
    Initialize GRU update gate bias so the first recurrent steps preserve the
    feedforward state reasonably well.

    PyTorch GRU gate order is reset, update, new.
    A positive update-gate bias makes h_t closer to h_{t-1} initially.
    """
    hidden_size = cell.hidden_size
    with torch.no_grad():
        # update gate slice: [hidden_size : 2 * hidden_size]
        cell.bias_ih[hidden_size : 2 * hidden_size].fill_(2.0)
        cell.bias_hh[hidden_size : 2 * hidden_size].fill_(2.0)


class ClassifierRecurrentVGG16Baseline(nn.Module):
    """
    Non-predictive-coding recurrent baseline for the VGG16 word classifier.

    It inserts RNN/GRU dynamics at the same three classifier stages as PCoders:

        stage 1: classifier[0], dim 4096
        stage 2: classifier[3], dim 4096
        stage 3: classifier[6], dim 1000

    Trained with the same unsupervised objective as the PC model: generative
    heads (gen1/gen2/gen3, matched to the PCoder feedback shapes) reconstruct the
    previous layer's representation, and the loss is the summed reconstruction MSE
    over recurrent timesteps (see reconstruction_loss()).

    Differences from the PC model:
        - the recurrent update is a learned RNN/GRU cell, not the PC
          error-minimizing update (no prediction-error-gradient correction term)
        - representations are anchored to the feed-forward state by a residual
          mixing gate, so the recurrence refines rather than overwrites

    Interface:
        t = 0:
            out = model(images)

        t > 0:
            out = model(None)

    This matches the calling convention of the PC model:
        outputs = net(inputs)  # t=0
        outputs = net(None)    # t=1,2,...
    """

    def __init__(
        self,
        backbone: nn.Module,
        cell_type: CellType = "rnn",
        use_topdown: bool = True,
        input_adapter: bool = True,
        residual_mix: bool = True,
        init_mix: float = 0.10,
        bottleneck: Optional[int] = None,
        freeze_backbone: bool = True,
        keep_backbone_eval: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.cell_type = cell_type
        self.use_topdown = use_topdown
        self.input_adapter = input_adapter
        self.residual_mix = residual_mix
        self.bottleneck = bottleneck
        self.freeze_backbone = freeze_backbone
        self.keep_backbone_eval = keep_backbone_eval

        # Compatibility with code that expects PCoder-like models.
        self.number_of_pcoders = 3

        # VGG16 classifier:
        # classifier[0]: Linear(25088 -> 4096)
        # classifier[1]: ReLU
        # classifier[2]: Dropout
        # classifier[3]: Linear(4096 -> 4096)
        # classifier[4]: ReLU
        # classifier[5]: Dropout
        # classifier[6]: Linear(4096 -> 1000)
        self.fc1 = self.backbone.classifier[0]
        self.relu1 = self.backbone.classifier[1]
        self.drop1 = self.backbone.classifier[2]

        self.fc2 = self.backbone.classifier[3]
        self.relu2 = self.backbone.classifier[4]
        self.drop2 = self.backbone.classifier[5]

        self.fc3 = self.backbone.classifier[6]

        self.dims = [4096, 4096, 1000]

        # Recurrent cells (optionally bottlenecked so the added recurrence is
        # small and the total added params match PC's feedback pathway).
        self.cell1, self.down1, self.up1 = _build_cell(cell_type, 4096, bottleneck)
        self.cell2, self.down2, self.up2 = _build_cell(cell_type, 4096, bottleneck)
        self.cell3, self.down3, self.up3 = _build_cell(cell_type, 1000, bottleneck)

        if cell_type == "gru":
            _init_gru_copy_bias(self.cell1)
            _init_gru_copy_bias(self.cell2)
            _init_gru_copy_bias(self.cell3)

        # Generative heads for the unsupervised reconstruction objective.
        # Each head predicts the previous layer's representation, matching
        # the PCoder feedback module shapes (parameter-matched, ~123.67 M):
        #   gen1: h1 (4096) -> flattened Conv5 features (25088)
        #   gen2: h2 (4096) -> stage-1 representation (4096)
        #   gen3: h3 (1000) -> stage-2 representation (4096)
        self.gen1 = nn.Linear(4096, 25088)
        self.gen2 = nn.Linear(4096, 4096)
        self.gen3 = nn.Linear(1000, 4096)

        # Optional input adapters.
        # With RNN cells, these help parameter-match the PCoder feedback modules.
        if input_adapter:
            self.adapter1 = nn.Linear(4096, 4096)
            self.adapter2 = nn.Linear(4096, 4096)
            self.adapter3 = nn.Linear(1000, 1000)

            _init_identity_if_square(self.adapter1)
            _init_identity_if_square(self.adapter2)
            _init_identity_if_square(self.adapter3)
        else:
            self.adapter1 = nn.Identity()
            self.adapter2 = nn.Identity()
            self.adapter3 = nn.Identity()

        # Optional top-down recurrent projections.
        # These are not predictive-coding feedback modules:
        # they are not trained to reconstruct lower-layer representations,
        # and they do not compute prediction errors.
        if use_topdown:
            self.td2_to_1 = nn.Linear(4096, 4096, bias=False)
            self.td3_to_2 = nn.Linear(1000, 4096, bias=False)

            # Start with no top-down influence; allow it to learn.
            nn.init.zeros_(self.td2_to_1.weight)
            nn.init.zeros_(self.td3_to_2.weight)
        else:
            self.td2_to_1 = None
            self.td3_to_2 = None

        if residual_mix:
            if not (0.0 < init_mix < 1.0):
                raise ValueError("init_mix must be between 0 and 1.")
            init_logit = torch.logit(torch.tensor(float(init_mix)))
            self.mix_logits = nn.Parameter(torch.full((3,), init_logit))
        else:
            self.mix_logits = None

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.reset()

    def train(self, mode: bool = True):
        """
        Keep the frozen VGG backbone in eval mode during recurrent-baseline
        training. This avoids stochastic dropout changes in the frozen backbone.
        """
        super().train(mode)
        if self.freeze_backbone and self.keep_backbone_eval:
            self.backbone.eval()
        return self

    def reset(self) -> None:
        """Clear cached feedforward features and recurrent states."""
        self._flat: Optional[Tensor] = None

        self.h1: Optional[Tensor] = None
        self.h2: Optional[Tensor] = None
        self.h3: Optional[Tensor] = None

        # recurrent cell hidden states (== h_i when no bottleneck, else b-dim)
        self.hs1: Optional[Tensor] = None
        self.hs2: Optional[Tensor] = None
        self.hs3: Optional[Tensor] = None

        # PCoder-like names for downstream RSA/ridge scripts.
        self.rep1: Optional[Tensor] = None
        self.rep2: Optional[Tensor] = None
        self.rep3: Optional[Tensor] = None

        # per-stage reconstruction MSE for the current timestep
        self.rec_err: Optional[List[Tensor]] = None

    def _features_to_flat(self, x: Tensor) -> Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _feedforward_init(self, x: Tensor) -> Tensor:
        """
        Run normal VGG feedforward pass and cache the three classifier states.

        This is t=0 and should be interpreted as the purely feedforward baseline.
        """
        flat = self._features_to_flat(x)

        if self.freeze_backbone:
            flat = flat.detach()

        z1 = self.fc1(flat)
        z2 = self.fc2(self.drop1(self.relu1(z1)))
        z3 = self.fc3(self.drop2(self.relu2(z2)))

        self._flat = flat
        self.h1 = z1
        self.h2 = z2
        self.h3 = z3

        # seed the recurrent cell hidden states from the feed-forward reps
        if self.bottleneck is None:
            self.hs1, self.hs2, self.hs3 = self.h1, self.h2, self.h3
        else:
            self.hs1 = self.down1(self.h1)
            self.hs2 = self.down2(self.h2)
            self.hs3 = self.down3(self.h3)

        self.rep1 = self.h1
        self.rep2 = self.h2
        self.rep3 = self.h3

        self._compute_recon()
        return self.h3

    def _cell_step(self, cell, down, up, z, old_hs):
        """One cell update. Returns (new_h_raw in stage space, new_hs)."""
        if down is None:
            new_hs = cell(z, old_hs)
            return new_hs, new_hs
        new_hs = cell(down(z), old_hs)
        return up(new_hs), new_hs

    def _mix(self, old: Tensor, new: Tensor, layer_idx: int) -> Tensor:
        if not self.residual_mix:
            return new
        mix = torch.sigmoid(self.mix_logits[layer_idx])
        return (1.0 - mix) * old + mix * new

    def _compute_recon(self) -> None:
        """Per-stage reconstruction MSE: each generative head predicts the
        previous layer's (detached) representation, exactly as the PC model's
        feedback modules reconstruct the previous layer."""
        self.rec_err = [
            F.mse_loss(self.gen1(self.h1), self._flat.detach()),
            F.mse_loss(self.gen2(self.h2), self.h1.detach()),
            F.mse_loss(self.gen3(self.h3), self.h2.detach()),
        ]

    def reconstruction_loss(self) -> Tensor:
        """Summed reconstruction MSE across the three stages for the current
        timestep (call after each forward step; sum/average over timesteps in
        the training loop)."""
        return sum(self.rec_err)

    def _recurrent_step(self) -> Tensor:
        """
        One recurrent update step.

        Bottom-up drive is computed through the same frozen classifier layers.
        Optional top-down projections provide a strong non-PC recurrent-feedback
        control, but without reconstruction or prediction-error minimization.
        """
        if self._flat is None or self.h1 is None or self.h2 is None or self.h3 is None:
            raise RuntimeError(
                "Call model(images) once before calling model(None). "
                "The first call initializes the recurrent states."
            )

        old_h1, old_h2, old_h3 = self.h1, self.h2, self.h3
        bn = self.bottleneck is None

        # Stage 1: closest to flattened Conv5-derived visual features.
        z1 = self.fc1(self._flat)
        if self.use_topdown:
            z1 = z1 + self.td2_to_1(old_h2)
        z1 = self.adapter1(z1)

        new_h1_raw, new_hs1 = self._cell_step(self.cell1, self.down1, self.up1, z1, self.hs1)
        new_h1 = self._mix(old_h1, new_h1_raw, layer_idx=0)

        # Stage 2: use updated lower stage, old higher stage for top-down input.
        z2 = self.fc2(self.drop1(self.relu1(new_h1)))
        if self.use_topdown:
            z2 = z2 + self.td3_to_2(old_h3)
        z2 = self.adapter2(z2)

        new_h2_raw, new_hs2 = self._cell_step(self.cell2, self.down2, self.up2, z2, self.hs2)
        new_h2 = self._mix(old_h2, new_h2_raw, layer_idx=1)

        # Stage 3: output-stage recurrence.
        z3 = self.fc3(self.drop2(self.relu2(new_h2)))
        z3 = self.adapter3(z3)

        new_h3_raw, new_hs3 = self._cell_step(self.cell3, self.down3, self.up3, z3, self.hs3)
        new_h3 = self._mix(old_h3, new_h3_raw, layer_idx=2)

        self.h1, self.h2, self.h3 = new_h1, new_h2, new_h3
        # carry hidden state: the mixed rep when no bottleneck, else the b-dim state
        self.hs1 = new_h1 if bn else new_hs1
        self.hs2 = new_h2 if bn else new_hs2
        self.hs3 = new_h3 if bn else new_hs3

        self.rep1 = self.h1
        self.rep2 = self.h2
        self.rep3 = self.h3

        self._compute_recon()
        return self.h3

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """
        Same calling convention as the PC model.

        t=0:
            logits = model(images)

        t>0:
            logits = model(None)
        """
        if x is not None:
            self.reset()
            return self._feedforward_init(x)

        return self._recurrent_step()

    @torch.no_grad()
    def forward_sequence(self, x: Tensor, max_timestep: int) -> List[Tensor]:
        """
        Convenience method for evaluation.

        Returns:
            outputs[0] = feedforward output, t=0
            outputs[t] = recurrent output
        """
        outputs = [self.forward(x)]
        for _ in range(1, max_timestep):
            outputs.append(self.forward(None))
        return outputs

    def get_representations(self, detach: bool = True) -> Dict[str, Tensor]:
        """
        Return current recurrent states for RSA/ridge analyses.
        """
        reps = {
            "rnn1": self.rep1,
            "rnn2": self.rep2,
            "rnn3": self.rep3,
        }

        if detach:
            reps = {k: v.detach() for k, v in reps.items() if v is not None}

        return reps
    
if __name__ == "__main__":
    import torch
    from torchvision.models import vgg16

    backbone =  vgg16(weights=None)
    model = ClassifierRecurrentVGG16Baseline(
        backbone=backbone,
        cell_type="rnn",          # use "gru" for GRU baseline
        use_topdown=True,         # strong non-PC recurrent-feedback control
        input_adapter=True,       # helps parameter matching
        residual_mix=True,
        freeze_backbone=True,
    )

    print("Baseline trainable params:", count_trainable_params(model))
    print("PCoder feedback params:", pcoder_feedback_param_count())

    x = torch.randn(2, 3, 224, 224)
    out0 = model(x)       # t = 0, feedforward
    out1 = model(None)    # t = 1, recurrent
    out2 = model(None)    # t = 2, recurrent

    print(out0.shape, out1.shape, out2.shape)