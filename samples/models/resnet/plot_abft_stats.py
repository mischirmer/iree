#!/usr/bin/env python3
"""
plot_abft_stats.py

Parse ABFT stats/log file(s) and plot distributions for:
 - FIC values
 - Row differences
 - Column differences

Usage:
  python plot_abft_stats.py --stats-file /path/to/abft_stats.log --out-dir plots

The parser is intentionally permissive: it looks for common labels like
`fic`, `FIC`, `row`, `row_diff`, `row-diff`, `col`, `col_diff`, `col-diff` and
will extract numeric values following those labels. It also attempts to
extract `row` and `col` when both appear on the same line.

Outputs (in OUT_DIR):
 - fic_hist.png
 - row_hist.png
 - col_hist.png
 - row_vs_col_scatter.png
 - row_col_box.png

"""

import argparse
import os
import re
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def parse_stats_file(path):
    """Parse the stats file and return lists: fic_vals, row_diffs, col_diffs.

    The function uses case-insensitive key matching and several label
    variants. It's forgiving about whitespace and separators.
    """
    fic_vals = []
    row_vals = []
    col_vals = []

    # candidate keys (lowercase) to search for
    # include variants such as "Dot -OC" used in some logs (normalize later)
    # Do NOT treat plain 'dot' or 'oc' as the FIC value — FIC is the difference (OC-Dot).
    # Include explicit variants for OC-Dot naming so labeled "OC-Dot" or "Dot-OC" are matched.
    fic_keys = ['fic', 'fuc', 'fic_value', 'fic_val', 'dot_oc', 'dot-oc', 'dot oc', 'dot -oc', 'oc_dot', 'oc-dot', 'oc dot']
    row_keys = ['row', 'row_diff', 'row-diff', 'row difference', 'row_difference']
    col_keys = ['col', 'col_diff', 'col-diff', 'col difference', 'col_difference']

    # common label-value regex: label [:=] number
    # allow parentheses in labels (e.g. "FIC (Dot -OC): 0.123") and spaces/hyphens
    label_value_re = re.compile(r"(?i)\b([a-z0-9_\- \(\)]+?)\b\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    # fallback: just extract all floats on a line
    float_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    with open(path, 'r', errors='replace') as fh:
        mode = None
        # mode: None | 'row' | 'col'
        for line in fh:
            line = line.strip()
            if not line:
                mode = None
                continue
            # mark whether this line was explicitly handled (e.g. OC/Dot header)
            line_handled = False

            # first try to find explicit labeled values
            found_labels = { 'fic': [], 'row': [], 'col': [] }
            for m in label_value_re.finditer(line):
                label = m.group(1).strip().lower()
                # normalize label: remove surrounding parentheses and excess spaces
                label = label.replace('(', ' ').replace(')', ' ').strip()
                val = m.group(2)
                # normalize label spacing/hyphens
                label_n = label.replace('-', '_')
                # check membership
                if any(k in label_n for k in fic_keys):
                    try:
                        found_labels['fic'].append(float(val))
                    except Exception:
                        pass
                if any(k in label_n for k in row_keys):
                    try:
                        found_labels['row'].append(float(val))
                    except Exception:
                        pass
                if any(k in label_n for k in col_keys):
                    try:
                        found_labels['col'].append(float(val))
                    except Exception:
                        pass

            # if we captured values by label, consume them
            if found_labels['fic']:
                fic_vals.extend(found_labels['fic'])
            if found_labels['row']:
                row_vals.extend(found_labels['row'])
            if found_labels['col']:
                col_vals.extend(found_labels['col'])

            # specialized parsing for hwacc logs: OC/Dot line and following row/col diffs
            # Example header: "OC: -5753.337402, Dot: -5753.339844, OC-Dot: 0.002441"
            m = re.search(r"\bOC\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*Dot\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*,\s*OC-?Dot\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?", line, flags=re.IGNORECASE)
            if m:
                try:
                    oc = float(m.group(1))
                    dot = float(m.group(2))
                    # If the log explicitly provides OC-Dot (group(3)), prefer it.
                    # Otherwise fall back to computing Dot - OC.
                    oc_dot_str = m.group(3)
                    if oc_dot_str is not None:
                        try:
                            oc_dot = float(oc_dot_str)
                        except Exception:
                            oc_dot = dot - oc
                    else:
                        oc_dot = dot - oc
                    fic_vals.append(oc_dot)
                    line_handled = True
                except Exception:
                    pass

            # Detect start of row/col diffs sections
            if re.search(r"^row diffs\b", line, flags=re.IGNORECASE):
                mode = 'row'
                continue
            if re.search(r"^col(?:umn)? diffs\b", line, flags=re.IGNORECASE):
                mode = 'col'
                continue

            # lines like: index,value
            if mode in ('row', 'col'):
                m2 = re.match(r"\s*(\d+)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m2:
                    try:
                        val = float(m2.group(2))
                        if mode == 'row':
                            row_vals.append(val)
                        else:
                            col_vals.append(val)
                        continue
                    except Exception:
                        pass

            # if no labeled values present, try to detect patterns like "row=.. col=.." was handled above
            # Next, try heuristics: lines like "row X col Y" or simply a trio (fic,row,col)
            if not (found_labels['row'] or found_labels['col'] or found_labels['fic']) and not line_handled:
                # attempt to find tokens and numbers
                toks = re.findall(r"([A-Za-z]+)\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if toks:
                    # map tokens
                    for t, v in toks:
                        tl = t.strip().lower().replace('-', '_')
                        if any(k in tl for k in fic_keys):
                            try:
                                fic_vals.append(float(v))
                            except Exception:
                                pass
                        elif any(k in tl for k in row_keys):
                            try:
                                row_vals.append(float(v))
                            except Exception:
                                pass
                        elif any(k in tl for k in col_keys):
                            try:
                                col_vals.append(float(v))
                            except Exception:
                                pass
                else:
                    # as a last resort, extract all floats and if a line contains exactly 3 floats,
                    # assume fic,row,col or if 2 floats then row,col
                    floats = float_re.findall(line)
                    if len(floats) == 3:
                        try:
                            a, b, c = map(float, floats)
                            fic_vals.append(a)
                            row_vals.append(b)
                            col_vals.append(c)
                        except Exception:
                            pass
                    elif len(floats) == 2:
                        try:
                            a, b = map(float, floats)
                            row_vals.append(a)
                            col_vals.append(b)
                        except Exception:
                            pass

    return fic_vals, row_vals, col_vals


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_hist(values, title, xlabel, out_path, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color='C0', alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_box(values_list, labels, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.boxplot(values_list, labels=labels, showmeans=True)
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(x, y, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.6, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot ABFT statistics (FIC, row diff, col diff)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stats-file', '-s', help='Path to ABFT stats/log file to parse')
    group.add_argument('--stats-dir', '-d', help='Directory containing ABFT stats/log files to aggregate (processes all *.log)')
    parser.add_argument('--out-dir', '-o', default='plots', help='Directory to write plots into')
    parser.add_argument('--show', action='store_true', help='Show plots interactively (requires display)')
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)

    fic_vals = []
    row_vals = []
    col_vals = []

    printed_any = False

    if args.stats_file:
        stats_file = args.stats_file
        if not os.path.isfile(stats_file):
            print(f"Stats file not found: {stats_file}")
            return
        f, r, c = parse_stats_file(stats_file)
        fic_vals.extend(f)
        row_vals.extend(r)
        col_vals.extend(c)

    else:
        stats_dir = args.stats_dir
        if not os.path.isdir(stats_dir):
            print(f"Stats directory not found: {stats_dir}")
            return
        files = sorted([os.path.join(stats_dir, p) for p in os.listdir(stats_dir) if p.endswith('.log') and os.path.isfile(os.path.join(stats_dir, p))])
        if not files:
            print(f"No .log files found in directory: {stats_dir}")
            return
        print(f"Parsing {len(files)} log files from {stats_dir}")
        for fn in files:
            f, r, c = parse_stats_file(fn)
            fic_vals.extend(f)
            row_vals.extend(r)
            col_vals.extend(c)
    if fic_vals:
        printed_any = True
        fic_path = os.path.join(out_dir, 'fic_hist.png')
        plot_hist(fic_vals, 'FIC Distribution', 'FIC', fic_path, bins=50)
        print(f'Wrote FIC histogram: {fic_path}')

    if row_vals:
        printed_any = True
        row_path = os.path.join(out_dir, 'row_hist.png')
        plot_hist(row_vals, 'Row Differences Distribution', 'Row difference', row_path, bins=50)
        print(f'Wrote row histogram: {row_path}')

    if col_vals:
        printed_any = True
        col_path = os.path.join(out_dir, 'col_hist.png')
        plot_hist(col_vals, 'Column Differences Distribution', 'Column difference', col_path, bins=50)
        print(f'Wrote col histogram: {col_path}')

    if row_vals and col_vals:
        scatter_path = os.path.join(out_dir, 'row_vs_col_scatter.png')
        plot_scatter(row_vals[:len(col_vals)], col_vals[:len(row_vals)], 'Row vs Column differences', 'Row diff', 'Col diff', scatter_path)
        print(f'Wrote row vs col scatter: {scatter_path}')

    if (row_vals or col_vals):
        # produce boxplot for whichever exist
        lists = []
        labels = []
        if row_vals:
            lists.append(row_vals)
            labels.append('row')
        if col_vals:
            lists.append(col_vals)
            labels.append('col')
        box_path = os.path.join(out_dir, 'row_col_box.png')
        plot_box(lists, labels, 'Row/Col Differences (boxplot)', box_path)
        print(f'Wrote boxplot: {box_path}')

    if not printed_any:
        print('No FIC/row/col values found in the stats file. Check file format.')

    if args.show:
        # If show requested, open each produced file with matplotlib display
        # (replot to screen) -- simple approach: show hist for each
        if fic_vals:
            plt.hist(fic_vals, bins=50, color='C0', alpha=0.8)
            plt.title('FIC Distribution')
            plt.show()
        if row_vals:
            plt.hist(row_vals, bins=50, color='C1', alpha=0.8)
            plt.title('Row Differences Distribution')
            plt.show()
        if col_vals:
            plt.hist(col_vals, bins=50, color='C2', alpha=0.8)
            plt.title('Column Differences Distribution')
            plt.show()


if __name__ == '__main__':
    main()
