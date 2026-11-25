"""Supporting-information files that are not written by their own scripts.

Run last, after the scripts that make the figures (see README). It
  - composes S5 Fig from the MEG RDMs of S5Fig_meg_rdms.py (A, B) and the model
    RDMs of 16_model_rdms.py (C), with pdflatex;
  - renders the figure PDFs below to figures/supporting_information/S<N>_Fig.tif
    (RGB, LZW, 300 dpi, 2250 px wide; si_utils.pdf_to_tif);
  - typesets S6 Table from tables/S6_Tab.tex and copies the S3 Table PDF.
S1, S2, S4 and S5 Tables and S4, S6, S8, S9 and S11 Figs are written by their own
scripts (S<N>Tab_*.py, S<N>Fig_*.py). Missing inputs are reported and skipped.
Needs pdflatex (TeX Live, with the standalone package) and pdftoppm (poppler).

    python figures_tables/make_supporting_information.py
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import shutil
import subprocess
from pathlib import Path

from config import fname
from si_utils import SI, build_table, pdf_to_tif

F = Path(fname.figures_dir)
FIGS = {  # S<N>_Fig: figure PDF (script)
    "S1_Fig": F / "behav/Fig_S1_term_contributions.pdf",             # 15_update_terms_dynamics.py
    "S2_Fig": F / "behav/seed_accuracy.pdf",                         # 13_runs_behavior_stats.py
    "S3_Fig": F / "behav/controls_acc_RL23.pdf",                     # 14_controls_ablations_behavior.py
    "S5_Fig": F / "rsa/S5_Fig_compose.pdf",                          # composed below
    "S7_Fig": F / "rsa/rsa_whole_brain_pcoderall_RL23_pcoder12.pdf",  # 19_rsa_wholecortex.py
    "S10_Fig": F / "dynamics/dynamics_rsa.pdf",                      # S10Fig_rsa_timesteps.py
    "S12_Fig": F / "rsa/rsa_controls.pdf",                           # 21_rsa_controls.py
    "S13_Fig": F / "rsa/stim_rsa_ST_vOT_RL23.pdf",                   # 27_stimulus_rsa_partial.py --combine_rois
    "S14_Fig": F / "rsa/stim_item_rdms_combined_RL23.pdf",           # S14Fig_stimulus_rdms.py
    "S15_Fig": F / "ridge/ridge_subjinf_08style_vOT_ST_RL23.pdf",    # S15Fig_ridge_participant_level.py
}

S5_TEX = r"""\documentclass[10pt,varwidth=5.25in,border=1pt]{standalone}
\usepackage{graphicx}
\newcommand{\panel}[2]{\noindent\textbf{#1}\par\vspace{1pt}\noindent\includegraphics[width=5.25in]{#2}\par\medskip}
\begin{document}
\panel{A}{meg_vOT_rdms.pdf}
\panel{B}{meg_ST_rdms.pdf}
\panel{C}{model_pcoders_rdms.pdf}
\end{document}
"""


def compose_s5():
    rsa = F / "rsa"
    parts = [rsa / p for p in ("meg_vOT_rdms.pdf", "meg_ST_rdms.pdf", "model_pcoders_rdms.pdf")]
    if not all(p.exists() for p in parts):
        print("S5 Fig: missing", [str(p) for p in parts if not p.exists()])
        return
    (rsa / "S5_Fig_compose.tex").write_text(S5_TEX)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "S5_Fig_compose.tex"], cwd=rsa, check=True,
                   stdout=subprocess.DEVNULL)


compose_s5()
for name, pdf in FIGS.items():
    if pdf.exists():
        pdf_to_tif(pdf, name)
    else:
        print(f"{name}: missing {pdf}")

build_table(Path("tables/S6_Tab.tex").read_text(), "S6_Tab")
s3 = F / "behav/soft_vs_onehot_backbone_accuracy.pdf"                # S3Tab_soft_vs_onehot.py
if s3.exists():
    shutil.copy(s3, SI / "S3_Tab.pdf")
    print("saved ->", SI / "S3_Tab.pdf")
else:
    print("S3_Tab: missing", s3)
