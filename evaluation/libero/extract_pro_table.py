#!/usr/bin/env python
"""
extract_pro_table.py
=====================
Parse LIBERO / LIBERO-PRO evaluation .txt logs (produced by run_eval_all.sh /
run_eval_pro.sh) and build a summary table: rows = task suite x method,
columns = Ori / Obj / Pos / Sem / Task -- same layout as the LIBERO-PRO paper.

File naming convention this script expects:
  Ori (no perturbation):   <prefix>_<suite>.txt
  Perturbed (LIBERO-PRO):  <prefix>_<dim>_<suite>.txt
    dim   in {object, position, semantic, task}
    suite in {spatial, object, goal, 10}

Each .txt must contain a line such as:
  Total success rate: 97/200 (97.0%)

Usage (one row):
  python extract_pro_table.py --dir . --run tds40k:"SimVLA + TDS @40k"

Usage (several rows, e.g. two budgets stacked in one table):
  python extract_pro_table.py --dir . \
      --run tds40k:"SimVLA + TDS @40k" \
      --run tds100k:"SimVLA + TDS @100k" \
      --latex
"""

import argparse
import os
import re

SUITE_ORDER = [("spatial", "Spatial"), ("object", "Object"),
               ("goal", "Goal"), ("10", "Long")]
DIM_ORDER = [(None, "Ori"), ("object", "Obj"), ("position", "Pos"),
             ("semantic", "Sem"), ("task", "Task")]

PCT_RE = re.compile(r"Total success rate:.*?\(([\d.]+)\s*%\)", re.IGNORECASE)


def read_pct(path):
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        text = f.read()
    m = PCT_RE.search(text)
    return (float(m.group(1)) if m else None), path


def file_for(directory, prefix, suite_key, dim_key):
    fname = f"{prefix}_{suite_key}.txt" if dim_key is None \
        else f"{prefix}_{dim_key}_{suite_key}.txt"
    return read_pct(os.path.join(directory, fname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory with the eval .txt files")
    ap.add_argument("--run", action="append", required=True,
                     help="prefix:Label pair, e.g. tds40k:'SimVLA+TDS @40k'. Repeatable.")
    ap.add_argument("--missing", default="--", help="placeholder for missing/unparsed values")
    ap.add_argument("--latex", action="store_true", help="also print a LaTeX table body")
    args = ap.parse_args()

    runs = []
    for r in args.run:
        if ":" not in r:
            raise SystemExit(f"--run must be prefix:Label, got: {r!r}")
        prefix, label = r.split(":", 1)
        runs.append((prefix, label))

    data = {}          # suite_key -> [(label, {dim_label: value_or_None}), ...]
    missing_files = []
    for suite_key, _ in SUITE_ORDER:
        rows = []
        for prefix, label in runs:
            row = {}
            for dim_key, dim_label in DIM_ORDER:
                val, path = file_for(args.dir, prefix, suite_key, dim_key)
                row[dim_label] = val
                if val is None:
                    missing_files.append(path)
            rows.append((label, row))
        data[suite_key] = rows

    col_w = 8
    header = f"{'Suite':<10}{'Method':<22}" + "".join(f"{d:>{col_w}}" for _, d in DIM_ORDER)
    print(header)
    print("-" * len(header))
    for suite_key, suite_name in SUITE_ORDER:
        for label, row in data[suite_key]:
            vals = "".join(
                f"{(f'{row[d]:.1f}' if row[d] is not None else args.missing):>{col_w}}"
                for _, d in DIM_ORDER)
            print(f"{suite_name:<10}{label:<22}{vals}")

    if missing_files:
        uniq = sorted(set(missing_files))
        print(f"\n[note] {len(uniq)} file(s) not found or unparsed (shown as "
              f"'{args.missing}'):")
        for p in uniq:
            print(f"   {p}")

    if args.latex:
        print("\n% ---- LaTeX table body ----")
        print(r"\begin{tabular}{ll" + "c" * len(DIM_ORDER) + "}")
        print(r"\toprule")
        print("Task Suite & Method & " + " & ".join(d for _, d in DIM_ORDER) + r" \\")
        print(r"\midrule")
        for si, (suite_key, suite_name) in enumerate(SUITE_ORDER):
            rows = data[suite_key]
            for i, (label, row) in enumerate(rows):
                lead = f"\\multirow{{{len(rows)}}}{{*}}{{{suite_name}}}" if i == 0 else ""
                vals = " & ".join(
                    (f"{row[d]:.1f}" if row[d] is not None else args.missing)
                    for _, d in DIM_ORDER)
                print(f"{lead} & {label} & {vals} \\\\")
            if si < len(SUITE_ORDER) - 1:
                print(r"\midrule")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
