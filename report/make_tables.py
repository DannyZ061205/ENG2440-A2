"""Write the report's LaTeX tables from the tables the executed notebook saved in outputs/tables/ (no hand-copied numbers).

    python report/make_tables.py        ->  report/tables/*.tex
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def m(x):
    """Typeset a number with a true minus sign."""
    return x.replace("-", "$-$") if isinstance(x, str) else x


def f(v, d):
    return m(f"{v:.{d}f}")


def ci(lo, hi, d):
    return m(f"[{lo:+.{d}f}, {hi:+.{d}f}]")


def sf(v, d):
    """Signed number, for differences."""
    return m(f"{v:+.{d}f}")


# ---------------------------------------------------------------- Table 1: data
vol = pd.read_csv(T / "volumes_per_split.csv", index_col=0)
rows = [rf"{s} & {int(r['cases'])} & {f(r['median (mL)'], 1)} [{f(r['IQR low'], 1)}, {f(r['IQR high'], 1)}] & "
        rf"{f(r['mean (mL)'], 1)} $\pm$ {f(r['SD (mL)'], 1)} & {f(r['min (mL)'], 1)}--{f(r['max (mL)'], 1)} \\"
        for s, r in vol.iterrows()]
(OUT / "data.tex").write_text("\n".join([
    r"\begin{tabular}{lrccc}", r"\toprule",
    r"split & cases & median [IQR] (mL) & mean $\pm$ SD (mL) & range (mL) \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Table 2: sampler comparison (validation)
samp = pd.read_csv(T / "sampling.csv", index_col=[0, 1])
per_run = pd.read_csv(T / "validation_per_run.csv", index_col=[0, 1])
diff = pd.read_csv(T / "validation_paired_difference.csv", index_col=0)
seeds = sorted(per_run.index.get_level_values(1).unique())
n_seeds = len(seeds)


def seed_cell(sampler, col, d, scale=1.0):
    v = per_run.loc[sampler][col] * scale
    return f(v.mean(), d) + (rf" $\pm$ {f(v.std(), d)}" if n_seeds > 1 else "")


def samp_cell(sampler, col, d):
    v = samp.loc[sampler][col] * 100
    return f(v.mean(), d) + (rf" $\pm$ {f(v.std(), d)}" if n_seeds > 1 else "")


lines = [
    rf"patches containing tumour (\%) & {samp_cell('uniform', 'patches containing tumour', 1)} & {samp_cell('tumour', 'patches containing tumour', 1)} & \\",
    rf"tumour share of patch voxels (\%) & {samp_cell('uniform', 'mean tumour fraction of patch voxels', 1)} & {samp_cell('tumour', 'mean tumour fraction of patch voxels', 1)} & \\",
    r"\midrule",
]
for label, col_run, key, d in [(r"Dice $\uparrow$", "Dice mean", "Dice", 3), (r"sensitivity $\uparrow$", "sensitivity mean", "sensitivity", 3),
                               (r"precision $\uparrow$", "precision mean", "precision", 3),
                               (r"false-positive volume (mL) $\downarrow$", "FP volume mean (mL)", "FP volume (mL)", 1),
                               (r"false-negative volume (mL) $\downarrow$", "FN volume mean (mL)", "FN volume (mL)", 1)]:
    r = diff.loc[key]
    lines.append(rf"{label} & {seed_cell('uniform', col_run, d)} & {seed_cell('tumour', col_run, d)} & "
                 rf"{sf(r['tumour-aware − uniform'], d)} {ci(r['95% CI low'], r['95% CI high'], d)} \\")
(OUT / "samplers.tex").write_text("\n".join([
    r"\begin{tabular}{lccc}", r"\toprule",
    r" & uniform & tumour-aware & difference [95\% CI] \\", r"\midrule", *lines, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Table 3: test results
summ = pd.read_csv(T / "test_summary.csv", header=[0, 1], index_col=0)
cis = pd.read_csv(T / "test_ci.csv", header=[0, 1], index_col=0)
vols = pd.read_csv(T / "volume_summary.csv", index_col=0)
lines = []
for metric, label, d in [("dice", r"Dice $\uparrow$", 3), ("iou", r"IoU $\uparrow$", 3), ("sensitivity", r"sensitivity $\uparrow$", 3),
                         ("hd95_mm", r"HD95 (mm) $\downarrow$", 1)]:
    cells = []
    for stage in ["before postprocessing", "after postprocessing"]:
        s = summ[stage][metric]
        cells.append(rf"{f(s['mean'], d)} $\pm$ {f(s['SD'], d)} & {f(s['median'], d)}")
    lo, hi = [float(x) for x in cis[("after postprocessing", metric)].iloc[0].strip("[]").split(",")]
    lines.append(rf"{label} & {cells[0]} & {cells[1]} & {m(f'[{lo:.{d}f}, {hi:.{d}f}]')} \\")
empty = [int(summ[s]["dice"]["empty predictions"]) for s in ["before postprocessing", "after postprocessing"]]
lines.append(rf"empty predictions & \multicolumn{{2}}{{c}}{{{empty[0]}}} & \multicolumn{{2}}{{c}}{{{empty[1]}}} & \\")
lines.append(r"\midrule")
lines.append(rf"$|$volume error$|$ (mL) $\downarrow$ & & & {f(vols.loc['mean', '|error| (mL)'], 1)} $\pm$ {f(vols.loc['SD', '|error| (mL)'], 1)} & "
             rf"{f(vols.loc['median', '|error| (mL)'], 1)} & \\")
lines.append(rf"$|$volume error$|$ (\%) $\downarrow$ & & & {f(vols.loc['mean', '|error| (%)'], 1)} $\pm$ {f(vols.loc['SD', '|error| (%)'], 1)} & "
             rf"{f(vols.loc['median', '|error| (%)'], 1)} & \\")
(OUT / "test.tex").write_text("\n".join([
    r"\begin{tabular}{lccccc}", r"\toprule",
    r" & \multicolumn{2}{c}{before postprocessing} & \multicolumn{3}{c}{after postprocessing} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
    r" & mean $\pm$ SD & median & mean $\pm$ SD & median & 95\% CI of mean \\", r"\midrule", *lines,
    r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Table 4: three cases
three = pd.read_csv(T / "three_cases.csv", index_col=0)
rows = [rf"{r['case']} & {cid.replace('_', chr(92) + '_')} & {f(r['dice'], 3)} & {f(r['hd95_mm'], 1)} & {f(r['ref_ml'], 1)} & "
        rf"{f(r['pred_ml'], 1)} & {f(r['vol_err_pct'], 1)} & {f(r['fp_ml'], 1)} & {f(r['fn_ml'], 1)} \\" for cid, r in three.iterrows()]
(OUT / "three.tex").write_text("\n".join([
    r"\begin{tabular}{llccccccc}", r"\toprule",
    r"case & id & Dice & HD95 (mm) & ref.\ (mL) & pred.\ (mL) & vol.\ error (\%) & FP (mL) & FN (mL) \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Table 5: bonus tests
mm = pd.read_csv(T / "missing_modality.csv", index_col=0)
rows = []
for name, r in mm.iterrows():
    delta = r.get("ΔDice vs all four [95% CI]")
    delta = "--" if not isinstance(delta, str) else m(delta.replace(" [", " [").replace("+", "+"))
    rows.append(rf"{name} & {f(r['Dice mean'], 3)} & {f(r['HD95 median (mm)'], 1)} & {f(r['|volume error| median (%)'], 1)} & "
                rf"{int(r['empty predictions'])} & {delta} \\")
(OUT / "missing.tex").write_text("\n".join([
    r"\begin{tabular}{lccccc}", r"\toprule",
    r"input & Dice (mean) & HD95 median (mm) & $|$vol.\ error$|$ median (\%) & empty & $\Delta$Dice [95\% CI] \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))
seam = pd.read_csv(T / "seams.csv", index_col=[0, 1])
rows = []
for (split, ov), r in seam.iterrows():
    rows.append(rf"{split} & {ov.replace('%', chr(92) + '%')} & {f(r['windows per case'], 1)} & {f(r['seconds per case'], 2)} & "
                rf"{f(r['Dice mean'], 3)} & {f(r['HD95 median (mm)'], 1)} & {f(r['|volume error| median (%)'], 1)} \\")
(OUT / "seams.tex").write_text("\n".join([
    r"\begin{tabular}{llccccc}", r"\toprule",
    r"split & overlap & windows/case & s/case & Dice (mean) & HD95 median (mm) & $|$vol.\ error$|$ median (\%) \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
