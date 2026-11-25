"""Supporting-information files for submission.

figures/supporting_information/ holds only the files submitted to the journal:
S<N>_Tab.pdf, S<N>_Fig.tif and S1_text.pdf. LaTeX sources of the tables and the
PDF versions of the figures go to figures/supporting_information/src/.

Figures: TIFF, RGB, LZW compression, 300 dpi, 2250 px wide (PLOS maximum width,
7.5 in at 300 dpi). Tables: PDF without caption (the legend is in the manuscript).

Command line (render a figure PDF made by another script):
    python si_utils.py figures/behav/Fig_S1_term_contributions.pdf S1_Fig
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from config import fname

SI = Path(fname.figures_dir) / "supporting_information"
SRC = SI / "src"
PLOS_MAX_WIDTH_PX = 2250


def build_table(tex, name):
    """Write SRC/<name>.tex, compile it there and copy the PDF to SI/<name>.pdf."""
    SRC.mkdir(parents=True, exist_ok=True)
    (SRC / f"{name}.tex").write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=SRC, check=True,
                   stdout=subprocess.DEVNULL)
    for ext in ("aux", "log"):
        (SRC / f"{name}.{ext}").unlink(missing_ok=True)
    shutil.copy(SRC / f"{name}.pdf", SI / f"{name}.pdf")
    print("saved ->", SI / f"{name}.pdf")


def pdf_to_tif(pdf, name):
    """Render a figure PDF to SI/<name>.tif (RGB, LZW, 300 dpi, 2250 px wide) and keep a
    copy of the PDF in SRC."""
    SRC.mkdir(parents=True, exist_ok=True)
    if Path(pdf).resolve() != (SRC / f"{name}.pdf").resolve():
        shutil.copy(pdf, SRC / f"{name}.pdf")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", str(PLOS_MAX_WIDTH_PX),
                        "-scale-to-y", "-1", str(pdf), f"{tmp}/x"], check=True)
        Image.open(f"{tmp}/x.png").convert("RGB").save(SI / f"{name}.tif", compression="tiff_lzw",
                                                       dpi=(300, 300))
    print("saved ->", SI / f"{name}.tif")


def fig_to_tif(fig, name, **savefig_kwargs):
    """Save a matplotlib figure as SRC/<name>.pdf and render SI/<name>.tif."""
    SRC.mkdir(parents=True, exist_ok=True)
    fig.savefig(SRC / f"{name}.pdf", **savefig_kwargs)
    pdf_to_tif(SRC / f"{name}.pdf", name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pdf_to_tif(sys.argv[1], sys.argv[2])
