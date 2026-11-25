#########################################################################
# Fig 3 composite of three panels (assembled with LaTeX; PDFs embedded unchanged):
#   A  figures/behav/iterate_accs_mean_heldout.pdf  (12_eval_runs_behavior.py ->
#      Fig3A_heldout_accuracy.py; nested CV, held-out RL2/RL3)
#   B  figures/behav/iterate_accs_fold5.pdf         (Fig3BC_model_behavior.py --fold 5;
#      median hyperparameters)
#   C  figures/behav/pair_out_activations.pdf       (Fig3BC_model_behavior.py; base-word ranks)
# Outputs: figures/Figure3.{pdf,tif}, figures/Figure3_preview.png
#
#   python figures_tables/Fig3_compose.py
#########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import fname

FIG = Path(fname.figures_dir).resolve()
PANELS = {"A": FIG / "behav/iterate_accs_mean_heldout.pdf",
          "B": FIG / "behav/iterate_accs_fold5.pdf",
          "C": FIG / "behav/pair_out_activations.pdf"}
assert all(p.exists() for p in PANELS.values())
lab = r"{\fontsize{11}{11}\selectfont\bfseries %s}"
tex = r"""\documentclass{article}
\usepackage[paperwidth=17.5cm,paperheight=18.6cm,margin=0.2cm]{geometry}
\usepackage{graphicx}\pagestyle{empty}\setlength{\parindent}{0pt}
\begin{document}
\begin{minipage}[t]{0.49\linewidth}%(A)s\par\includegraphics[width=\linewidth]{%(pA)s}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}%(B)s\par\includegraphics[width=\linewidth]{%(pB)s}\end{minipage}
\par\medskip
%(C)s\par\includegraphics[width=\linewidth]{%(pC)s}
\end{document}
""" % {"A": lab % "A", "B": lab % "B", "C": lab % "C",
       "pA": PANELS["A"], "pB": PANELS["B"], "pC": PANELS["C"]}
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "fig3.tex").write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig3.tex"], cwd=tmp, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copy(Path(tmp) / "fig3.pdf", FIG / "Figure3.pdf")
subprocess.run(["pdftoppm", "-r", "300", "-singlefile", "-tiff", "-tiffcompression", "lzw",
                str(FIG / "Figure3.pdf"), str(FIG / "Figure3")], check=True)
subprocess.run(["pdftoppm", "-r", "100", "-singlefile", "-png", str(FIG / "Figure3.pdf"),
                str(FIG / "Figure3_preview")], check=True)
print("saved ->", FIG / "Figure3.pdf", FIG / "Figure3.tif")
