"""S3 Table: top-1 accuracy (%) of the feedforward backbone trained with soft labels
versus one-hot labels on the experimental stimuli, per condition. The values are
entered by hand from the output of 03_test_backbone_on_stimuli.py (soft-label backbone:
default; one-hot backbone: --backbone .../vgg16_checkpoint_best_onehot.pth.tar,
trained by 02_train_backbone.py --target onehot).

Output: figures/behav/soft_vs_onehot_backbone_accuracy.{pdf,png}

    python figures_tables/S3Tab_soft_vs_onehot.py
"""

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("figures/behav")
OUT_STEM = OUT_DIR / "soft_vs_onehot_backbone_accuracy"

COLUMNS = ["Training target", "RW", "RL1", "RL2", "RL3"]
ROWS = [
    ["Soft label", "100.0", "84.4", "56.3", "14.1"],
    ["One-hot label", "98.5", "54.1", "20.0", "3.7"],
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 1.75))
    ax.axis("off")

    table = ax.table(
        cellText=ROWS,
        colLabels=COLUMNS,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.165, 0.165, 0.165, 0.165],
        bbox=[0.01, 0.08, 0.98, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)

    n_rows = len(ROWS) + 1
    n_cols = len(COLUMNS)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0)
        cell.PAD = 0.12
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.1)
        elif row == n_rows - 1:
            cell.visible_edges = "B"
            cell.set_linewidth(1.1)
        else:
            cell.visible_edges = ""

        if col == 0:
            cell.get_text().set_ha("left")

    # lower rule across every column
    for col in range(n_cols):
        table[(n_rows - 1, col)].visible_edges = "B"

    fig.savefig(
        OUT_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=False,
    )
    fig.savefig(
        OUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)

    print(f"saved -> {OUT_STEM}.[pdf|png]")


if __name__ == "__main__":
    main()
