"""
Offline requirement audit for ENG2440 Assignment 2.

Run it from a terminal after executing the notebook:

    python -m helpers.checklist 25011189_ENG2440_A2.ipynb [report/main.tex]

It re-reads the saved .ipynb file and checks that
* the notebook was executed top to bottom in one run, without errors,
* every requirement of the brief has its evidence in the notebook: a printed output, a piece of code, a heading
  or a written answer,
* every expected figure was saved and passed the automatic figure audit, and
* decimal numbers quoted in the notebook markdown, and (if a report .tex is given) in the report, also appear in
  some code output (a guard against stale or invented numbers).
"""

import json
import re
import sys
from pathlib import Path

# (id, marks, requirement, [evidence]); evidence items are ("md" | "code" | "out" | "fig", text that must appear)
RUBRIC = [
    ("A1", 2, "Input, output, unit of analysis, intended use; voxel/slice/volume/case",
     [("md", "**Input.**"), ("md", "**Output.**"), ("md", "**Unit of analysis.**"), ("md", "**Intended technical use.**")]),
    ("A2", 3, ">= 3 cases: shape, channel order, affine, orientation, spacing, label values",
     [("out", "channel order (dataset.json"), ("out", "orientation"), ("out", "voxel spacing (mm)"), ("out", "label values"),
      ("out", "affine of BRATS_001")]),
    ("A3", 2, "One case: 4 channels at one physical location + reference", [("out", "tumour centre at voxel"), ("fig", "A_views")]),
    ("A4", 3, "Three planes + why index order is insufficient", [("md", "### A.4 Why array index order is not enough")]),
    ("B1", 2, "Supplied split used exactly; no slice/patch split", [("code", 'pd.read_csv(CONFIG["split_file"])'), ("md", "### B.1")]),
    ("B2", 3, "Files exist; case ids disjoint", [("out", "the three splits share no case")]),
    ("B3", 3, "Case counts + tumour-volume distribution per split", [("out", "Reference whole-tumour volume per split"), ("fig", "B_volumes")]),
    ("C1", 3, "Geometry of 4 channels + label verified; target grid", [("out", "image affine == label affine"), ("md", "**No resampling is needed.**")]),
    ("C2", 2, "Linear for MRI, nearest for labels; labels stay valid", [("out", "only valid labels {0,1,2,3}")]),
    ("C3", 3, "Non-zero-voxel normalisation; resulting mean / SD", [("out", "Brain-voxel statistics before and after z-scoring")]),
    ("C4", 4, "Crop/pad; training-only augmentation; before/after; justified",
     [("md", "**Crop and pad.**"), ("md", "**Augmentation (training patches only).**"), ("fig", "C_augmentation")]),
    ("D1", 3, "Dataset + DataLoader shapes printed and verified", [("out", "[batch, C, D, H, W]"), ("code", "assert x.shape == (B, 4, P, P, P)")]),
    ("D2", 5, "3D U-Net: in/out channels, parameter count, patch size", [("out", "trainable parameters"), ("out", "input channels: 4")]),
    ("D3", 4, "Uniform training: loss, optimiser, LR, budget, seed", [("md", "### D.3 Training with uniform random patches"), ("md", "**loss**"), ("md", "**seed**")]),
    ("D4", 5, "Tumour-aware training, controlled; tumour-patch fractions", [("out", "every setting except the sampler"), ("out", "patches containing tumour")]),
    ("D5", 3, "Loss + Dice curves; under/overfitting discussion", [("fig", "D_curves"), ("md", "**Reading the curves.**")]),
    ("E1", 3, "Same architecture, split, loss, budget", [("out", "shared by both samplers")]),
    ("E2", 4, "Val Dice, sensitivity, FP volume; mechanism", [("out", "Validation, 73 cases"), ("md", "**Mechanism.**")]),
    ("E3", 3, "Final configuration from validation; oversampling disadvantage", [("out", "final configuration:"), ("md", "**Too much oversampling.**")]),
    ("F1", 4, "Sliding window: patch, stride, overlap, blending", [("out", "stride 32, overlap 50%")]),
    ("F2", 4, "Test Dice, IoU, sensitivity, HD95 mm; mean/SD/median; empty", [("out", "empty predictions"), ("out", "Test split, 73 cases")]),
    ("F3", 3, "Volume in mL; absolute and % error per case", [("out", "|error| (%)"), ("fig", "F_volumes")]),
    ("F4", 4, "Connected-component rule chosen on validation; before/after", [("out", "chosen on validation"), ("out", "postprocessing changed")]),
    ("G1", 3, "Best/median/worst, three planes, overlays", [("fig", "G_errors")]),
    ("G2", 3, "TP/FP/FN coloured; error related to boundary/small/context", [("out", "within 2 mm of the reference surface"), ("md", "**What the errors are.**")]),
    ("G3", 2, "Dice vs HD95 vs volume error; one metric is not enough", [("out", "Best, median and worst test cases by Dice"), ("md", "**One number per case is not enough.**")]),
    ("G4", 2, ">= 3 limitations incl. budget and external-site validation", [("md", "### G.4 Limitations"), ("md", "external")]),
    ("X1", 0, "BONUS 1: missing-modality stress test", [("out", "one channel replaced by zeros"), ("fig", "X1_missing_modality"), ("md", "**Why zeroing is an imperfect simulation.**")]),
    ("X2", 0, "BONUS 2: sliding-window seam test", [("out", "Window overlap: run time and metrics"), ("fig", "X2_seams"), ("md", "**Why choosing the overlap on test labels would bias the result.**")]),
]

EXPECTED_FIGURES = ["A_views", "B_volumes", "C_augmentation", "D_curves", "E_comparison", "F_volumes", "G_errors",
                    "X1_missing_modality", "X2_seams"]
NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d{2,})(?![\w.])")      # numbers with 2+ decimals, e.g. 0.85


def _text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _outputs(cell):
    chunks = []
    for out in cell.get("outputs", []):
        if "text" in out:
            chunks.append("".join(out["text"]) if isinstance(out["text"], list) else out["text"])
        for key in ("text/plain", "text/html", "text/markdown"):
            data = out.get("data", {}).get(key)
            if data:
                chunks.append("".join(data) if isinstance(data, list) else data)
    return "\n".join(chunks)


def _found(num, outputs):
    if num in outputs:
        return True
    # rounded from a longer printed number, e.g. 0.861 from 0.8613261
    d = len(num.split(".")[1])
    target = float(num)
    for m in re.finditer(r"-?\d+\.\d+", outputs):
        v = m.group(0)
        if len(v.split(".")[1]) > d and abs(round(float(v), d) - target) < 1e-9:
            return True
        if abs(round(abs(float(v)) * 100, d) - target) < 1e-9 and len(v.split(".")[1]) >= d + 2:
            return True                                                       # a fraction quoted as a percentage
    return False


def audit(nb_path, report=None, fig_dir="figures"):
    cells = json.loads(Path(nb_path).read_text())["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    md = "\n\n".join(_text(c) for c in cells if c["cell_type"] == "markdown")
    src = "\n\n".join(_text(c) for c in code)
    out = "\n\n".join(_outputs(c) for c in code)
    figs = {n for n in EXPECTED_FIGURES if (Path(fig_dir) / f"{n}.pdf").exists()}
    problems, warnings = [], []

    counts = [c.get("execution_count") for c in code if _text(c).strip()]
    if any(n is None for n in counts):
        problems.append("some code cells were never executed")
    elif counts != list(range(1, len(counts) + 1)):
        problems.append("execution counts are not 1..N: the notebook was not run top to bottom in one go")
    for c in code:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                problems.append(f"error output: {o.get('ename')}: {o.get('evalue', '')[:120]}")

    where = {"md": md, "code": src, "out": out}
    met = 0
    for rid, marks, desc, evidence in RUBRIC:
        missing = [text for kind, text in evidence
                   if (kind == "fig" and text not in figs) or (kind != "fig" and text not in where[kind])]
        if missing:
            problems.append(f"[{rid}] {desc}: missing {missing}")
        else:
            met += 1

    for name in EXPECTED_FIGURES:
        if name not in figs:
            problems.append(f"missing figure {fig_dir}/{name}.pdf")
        log = Path(fig_dir) / "_audit" / f"{name}.json"
        if log.exists():
            errors = [i for i in json.loads(log.read_text())["issues"] if i["severity"] == "error"]
            if errors:
                problems.append(f"figure {name}: {len(errors)} figure-audit error(s)")

    texts = [("notebook markdown", md)]
    if report:
        tex = Path(report).read_text()
        tex = tex[:tex.find(r"\begin{thebibliography}")] if r"\begin{thebibliography}" in tex else tex
        texts.append((Path(report).name, tex))
    for name, text in texts:
        text = re.sub(r"doi:\S+|arXiv:\S+|\d+\.\d+\\,?(in|cm|pt)\b", "", text)
        for num in sorted(set(NUMBER_RE.findall(text))):
            if not _found(num, out):
                warnings.append(f"{name}: number {num} not found in any code output")
        for placeholder in ("TODO", "TBD", "XXX", "???"):
            if placeholder in text:
                problems.append(f"{name} still contains placeholder {placeholder!r}")

    print(f"Audit of {nb_path}" + (f" and {report}" if report else ""))
    print(f"  code cells executed : {len(counts)}")
    print(f"  requirements with evidence : {met}/{len(RUBRIC)}")
    print(f"  figures present     : {len(figs)}/{len(EXPECTED_FIGURES)}")
    for w in warnings:
        print("  WARN ", w)
    for p in problems:
        print("  FAIL ", p)
    print("RESULT:", "PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return not problems


if __name__ == "__main__":
    ok = audit(sys.argv[1] if len(sys.argv) > 1 else "25011189_ENG2440_A2.ipynb",
               sys.argv[2] if len(sys.argv) > 2 else None)
    sys.exit(0 if ok else 1)
