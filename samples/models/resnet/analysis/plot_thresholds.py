#!/usr/bin/env python3
"""
plot_abft_stats.py

Original functionality:
 - Parse a single stats file (--stats-file) or one directory of stats files
   (--stats-dir) and produce 5 plots (FIC hist, row hist, col hist,
   row_vs_col_scatter, row_col_box).

Extended functionality (new):
 - With --multi-node, scan a set of node directories (default:
   /tmp/logfiles/node_00??/). For each node, parse its .log files,
   then create 5 big figures where each node has its own subplot.
   Subplot titles use M/N/K from a provided lookup for indices 0..50.

"""

import argparse
import os
import re
import math
from collections import defaultdict
import glob

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Node -> (M, N, K) mapping for indices 0..50
# ---------------------------------------------------------------------------

NODE_MNK = {
    0:  (12544, 64, 147),
    1:  (3136, 256, 64),
    2:  (3136, 64, 64),
    3:  (3136, 64, 576),
    4:  (3136, 256, 64),
    5:  (3136, 64, 256),
    6:  (3136, 64, 576),
    7:  (3136, 256, 64),
    8:  (3136, 64, 256),
    9:  (3136, 64, 576),
    10: (3136, 256, 64),
    11: (784, 512, 256),
    12: (784, 128, 256),
    13: (784, 128, 1152),
    14: (784, 512, 128),
    15: (784, 128, 512),
    16: (784, 128, 1152),
    17: (784, 512, 128),
    18: (784, 128, 512),
    19: (784, 128, 1152),
    20: (784, 512, 128),
    21: (784, 128, 512),
    22: (784, 128, 1152),
    23: (784, 512, 128),
    24: (196, 1024, 512),
    25: (196, 256, 512),
    26: (196, 256, 2304),
    27: (196, 1024, 256),
    28: (196, 256, 1024),
    29: (196, 256, 2304),
    30: (196, 1024, 256),
    31: (196, 256, 1024),
    32: (196, 256, 2304),
    33: (196, 1024, 256),
    34: (196, 256, 1024),
    35: (196, 256, 2304),
    36: (196, 1024, 256),
    37: (196, 256, 1024),
    38: (196, 256, 2304),
    39: (196, 1024, 256),
    40: (196, 256, 1024),
    41: (196, 256, 2304),
    42: (196, 1024, 256),
    43: (49, 2048, 1024),
    44: (49, 512, 1024),
    45: (49, 512, 4608),
    46: (49, 2048, 512),
    47: (49, 512, 2048),
    48: (49, 512, 4608),
    49: (49, 2048, 512),
    50: (49, 512, 2048),
    51: (49, 512, 4608),
    52: (49, 2048, 512),
    53: (1, 1000, 2048),
}


def node_title_with_mnk(node_name: str) -> str:
    """Generate a title 'node_000X\nM=.. N=.. K=..' if we have MNK info."""
    try:
        idx_str = node_name.split("_")[-1]
        idx = int(idx_str)
    except Exception:
        return node_name

    mnk = NODE_MNK.get(idx)
    if not mnk:
        return node_name
    M, N, K = mnk
    return f"{node_name}\nM={M} N={N} K={K}"


# ---------------------------------------------------------------------------
# Original parsing + single-plot helpers
# ---------------------------------------------------------------------------

def parse_stats_file(path):
    """Parse the stats file and return lists: fic_vals, row_diffs, col_diffs.

    The function uses case-insensitive key matching and several label
    variants. It's forgiving about whitespace and separators.
    """
    fic_vals = []
    row_vals = []
    col_vals = []

    fic_keys = ['fic', 'fuc', 'fic_value', 'fic_val',
                'dot_oc', 'dot-oc', 'dot oc', 'dot -oc',
                'oc_dot', 'oc-dot', 'oc dot']
    row_keys = ['row', 'row_diff', 'row-diff', 'row difference', 'row_difference']
    col_keys = ['col', 'col_diff', 'col-diff', 'col difference', 'col_difference']

    label_value_re = re.compile(
        r"(?i)\b([a-z0-9_\- \(\)]+?)\b\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    float_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    with open(path, 'r', errors='replace') as fh:
        mode = None
        for line in fh:
            line = line.strip()
            if not line:
                mode = None
                continue

            line_handled = False
            found_labels = {'fic': [], 'row': [], 'col': []}

            for m in label_value_re.finditer(line):
                label = m.group(1).strip().lower()
                label = label.replace('(', ' ').replace(')', ' ').strip()
                val = m.group(2)
                label_n = label.replace('-', '_')
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

            if found_labels['fic']:
                fic_vals.extend(found_labels['fic'])
            if found_labels['row']:
                row_vals.extend(found_labels['row'])
            if found_labels['col']:
                col_vals.extend(found_labels['col'])

            m = re.search(
                r"\bOC\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*Dot\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*,\s*OC-?Dot\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?",
                line, flags=re.IGNORECASE)
            if m:
                try:
                    oc = float(m.group(1))
                    dot = float(m.group(2))
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

            if re.search(r"^row diffs\b", line, flags=re.IGNORECASE):
                mode = 'row'
                continue
            if re.search(r"^col(?:umn)? diffs\b", line, flags=re.IGNORECASE):
                mode = 'col'
                continue

            if mode in ('row', 'col'):
                m2 = re.match(
                    r"\s*(\d+)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
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

            if not (found_labels['row'] or found_labels['col'] or found_labels['fic']) and not line_handled:
                toks = re.findall(
                    r"([A-Za-z]+)\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if toks:
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


# ---------------------------------------------------------------------------
# New: multi-node plotting helpers
# ---------------------------------------------------------------------------

def plot_multi_hist(data_by_node, title, xlabel, out_path, bins=50):
    """data_by_node: dict {node_name: [values]} -> grid of histograms."""
    if not data_by_node:
        return

    node_names = sorted(data_by_node.keys())
    n = len(node_names)
    cols = min(8, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, node in enumerate(node_names):
        vals = data_by_node[node]
        ax = axes.ravel()[i]
        ax.set_visible(True)
        ax.hist(vals, bins=bins, color='C0', alpha=0.8)
        ax.set_title(node_title_with_mnk(node), fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def plot_multi_scatter(row_by_node, col_by_node, title, xlabel, ylabel, out_path):
    """Grid of scatter plots; only nodes with both row & col are shown."""
    common_nodes = sorted(set(row_by_node.keys()) & set(col_by_node.keys()))
    if not common_nodes:
        return

    n = len(common_nodes)
    cols = min(8, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, node in enumerate(common_nodes):
        rvals = row_by_node[node]
        cvals = col_by_node[node]
        length = min(len(rvals), len(cvals))
        if length == 0:
            continue
        ax = axes.ravel()[i]
        ax.set_visible(True)
        ax.scatter(rvals[:length], cvals[:length], alpha=0.6, s=5)
        ax.set_title(node_title_with_mnk(node), fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def plot_multi_box(row_by_node, col_by_node, title, out_path):
    """Grid: each subplot shows row/col boxplot for one node."""
    nodes = sorted(set(row_by_node.keys()) | set(col_by_node.keys()))
    if not nodes:
        return

    n = len(nodes)
    cols = min(8, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, node in enumerate(nodes):
        rvals = row_by_node.get(node, [])
        cvals = col_by_node.get(node, [])
        if not (rvals or cvals):
            continue
        data = []
        labels = []
        if rvals:
            data.append(rvals)
            labels.append("row")
        if cvals:
            data.append(cvals)
            labels.append("col")

        ax = axes.ravel()[i]
        ax.set_visible(True)
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(node_title_with_mnk(node), fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def plot_multi_hist_with_fit(data_by_node, title, xlabel, out_path, bins=50):
    """data_by_node: dict {node_name: [values]}
       Draw per-node histograms (normalized) and overlay N(μ, σ²) fit.
       Also annotates μ and σ² in each subplot.
    """
    if not data_by_node:
        return

    node_names = sorted(data_by_node.keys())
    n = len(node_names)
    cols = min(8, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)

    # hide all axes first
    for ax in axes.ravel():
        ax.set_visible(False)

    for i, node in enumerate(node_names):
        vals = np.asarray(data_by_node[node], dtype=float)
        if vals.size == 0:
            continue

        ax = axes.ravel()[i]
        ax.set_visible(True)

        # Histogram as a density
        ax.hist(vals, bins=bins, density=True, color='C0', alpha=0.5)

        # Fit parameters
        mu = float(vals.mean())
        # unbiased estimator (ddof=1) if at least 2 points
        sigma = float(vals.std(ddof=1)) if vals.size > 1 else 0.0

        # Overlay Gaussian curve if sigma > 0
        if sigma > 0:
            x_min = vals.min()
            x_max = vals.max()
            # if all values identical, just make a small window
            if x_min == x_max:
                x_min -= 1e-9
                x_max += 1e-9
            xs = np.linspace(x_min, x_max, 200)
            pdf = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(
                -0.5 * ((xs - mu) / sigma) ** 2
            )
            ax.plot(xs, pdf, linewidth=1.0)

        ax.set_title(node_title_with_mnk(node), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("density", fontsize=7)

        # Annotate μ and σ² inside the subplot (top-left)
        ax.text(
            0.05, 0.95,
            f"μ = {mu:.2e}\nσ² = {sigma**2:.2e}",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=6,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def make_multi_node_plots(node_pattern, out_dir):
    """Scan node_00?? dirs, parse logs per node, and make grid plots.
       Also produce normalized versions (value * 1/(M*N*K)) per node.
    """
    node_dirs = sorted(glob.glob(node_pattern))
    if not node_dirs:
        print(f"No node directories match pattern: {node_pattern}")
        return

    print(f"Found {len(node_dirs)} node directories")

    fic_by_node = {}
    row_by_node = {}
    col_by_node = {}

    # normalized dicts
    fic_by_node_norm = {}
    row_by_node_norm = {}
    col_by_node_norm = {}

    for node_dir in node_dirs:
        node_name = os.path.basename(node_dir)
        log_paths = sorted(
            p for p in glob.glob(os.path.join(node_dir, "*.log"))
            if os.path.isfile(p)
        )
        if not log_paths:
            print(f"  [WARN] no .log files in {node_dir}")
            continue

        fic_vals = []
        row_vals = []
        col_vals = []

        for log_path in log_paths:
            f, r, c = parse_stats_file(log_path)
            fic_vals.extend(f)
            row_vals.extend(r)
            col_vals.extend(c)

        if fic_vals:
            fic_by_node[node_name] = fic_vals
        if row_vals:
            row_by_node[node_name] = row_vals
        if col_vals:
            col_by_node[node_name] = col_vals

        # --- compute normalization factor 1/(M*N*K) if we know MNK for this node ---
        norm_scale = None
        try:
            idx = int(node_name.split("_")[-1])
            if idx in NODE_MNK:
                M, N, K = NODE_MNK[idx]
                norm_scale = 1.0 / (M * N * K)
        except Exception:
            norm_scale = None

        if norm_scale is not None:
            if fic_vals:
                fic_by_node_norm[node_name] = [v * norm_scale for v in fic_vals]
            if row_vals:
                row_by_node_norm[node_name] = [v * norm_scale for v in row_vals]
            if col_vals:
                col_by_node_norm[node_name] = [v * norm_scale for v in col_vals]

        print(f"  {node_name}: FIC={len(fic_vals)} Row={len(row_vals)} Col={len(col_vals)}"
              + (f" (normalized with 1/(M*N*K) where M,N,K={NODE_MNK[idx]})" if norm_scale is not None else ""))

    if not (fic_by_node or row_by_node or col_by_node):
        print("No FIC/row/col values found in any node directories.")
        return

    # ---------- raw plots ----------
    if fic_by_node:
        fic_path = os.path.join(out_dir, "fic_hist.png")
        print(f"Writing multi-node FIC hist grid: {fic_path}")
        plot_multi_hist(fic_by_node, "FIC Distribution per Node", "FIC", fic_path, bins=50)

    if row_by_node:
        row_path = os.path.join(out_dir, "row_hist.png")
        print(f"Writing multi-node row hist grid: {row_path}")
        plot_multi_hist(row_by_node, "Row Differences per Node", "Row diff", row_path, bins=50)

    if col_by_node:
        col_path = os.path.join(out_dir, "col_hist.png")
        print(f"Writing multi-node col hist grid: {col_path}")
        plot_multi_hist(col_by_node, "Column Differences per Node", "Col diff", col_path, bins=50)

    if row_by_node and col_by_node:
        scatter_path = os.path.join(out_dir, "row_vs_col_scatter.png")
        print(f"Writing multi-node row vs col scatter grid: {scatter_path}")
        plot_multi_scatter(row_by_node, col_by_node,
                           "Row vs Column Differences per Node",
                           "Row diff", "Col diff",
                           scatter_path)

    if row_by_node or col_by_node:
        box_path = os.path.join(out_dir, "row_col_box.png")
        print(f"Writing multi-node row/col boxplot grid: {box_path}")
        plot_multi_box(row_by_node, col_by_node,
                       "Row/Col Differences per Node (boxplots)",
                       box_path)

        # ---------- normalized plots (value * 1/(M*N*K)) ----------
    if fic_by_node_norm:
        fic_norm_path = os.path.join(out_dir, "fic_hist_normalized.png")
        print(f"Writing normalized multi-node FIC hist grid: {fic_norm_path}")
        plot_multi_hist(fic_by_node_norm,
                        "Normalized FIC/(M*N*K) per Node",
                        "FIC / (M*N*K)",
                        fic_norm_path,
                        bins=50)

        # With Gaussian fit
        fic_norm_fit_path = os.path.join(out_dir, "fic_hist_normalized_fit.png")
        print(f"Writing normalized multi-node FIC hist grid with Gaussian fit: {fic_norm_fit_path}")
        plot_multi_hist_with_fit(
            fic_by_node_norm,
            "Normalized FIC/(M*N*K) per Node (Gaussian fit)",
            "FIC / (M*N*K)",
            fic_norm_fit_path,
            bins=50
        )

    if row_by_node_norm:
        row_norm_path = os.path.join(out_dir, "row_hist_normalized.png")
        print(f"Writing normalized multi-node row hist grid: {row_norm_path}")
        plot_multi_hist(row_by_node_norm,
                        "Normalized Row Differences per Node",
                        "Row diff / (M*N*K)",
                        row_norm_path,
                        bins=50)

        # With Gaussian fit
        row_norm_fit_path = os.path.join(out_dir, "row_hist_normalized_fit.png")
        print(f"Writing normalized multi-node row hist grid with Gaussian fit: {row_norm_fit_path}")
        plot_multi_hist_with_fit(
            row_by_node_norm,
            "Normalized Row Differences per Node (Gaussian fit)",
            "Row diff / (M*N*K)",
            row_norm_fit_path,
            bins=50
        )

    if col_by_node_norm:
        col_norm_path = os.path.join(out_dir, "col_hist_normalized.png")
        print(f"Writing normalized multi-node col hist grid: {col_norm_path}")
        plot_multi_hist(col_by_node_norm,
                        "Normalized Column Differences per Node",
                        "Col diff / (M*N*K)",
                        col_norm_path,
                        bins=50)

        # With Gaussian fit
        col_norm_fit_path = os.path.join(out_dir, "col_hist_normalized_fit.png")
        print(f"Writing normalized multi-node col hist grid with Gaussian fit: {col_norm_fit_path}")
        plot_multi_hist_with_fit(
            col_by_node_norm,
            "Normalized Column Differences per Node (Gaussian fit)",
            "Col diff / (M*N*K)",
            col_norm_fit_path,
            bins=50
        )
      
        
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot ABFT statistics (FIC, row diff, col diff)')
    parser.add_argument('--stats-file', '-s',
                        help='Path to ABFT stats/log file to parse')
    parser.add_argument('--stats-dir', '-d',
                        help='Directory containing ABFT stats/log files to aggregate (processes all *.log)')
    parser.add_argument('--out-dir', '-o', default='plots',
                        help='Directory to write plots into')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively (single-file/dir mode only)')
    parser.add_argument('--multi-node', action='store_true',
                        help='Scan multiple node_00?? directories and create grid plots')
    parser.add_argument('--node-pattern', default='/tmp/logfiles/node_00??',
                        help='Glob pattern for node dirs when using --multi-node '
                             '(default: /tmp/logfiles/node_00??)')
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)

    # ---------------- Multi-node mode ----------------
    if args.multi_node:
        make_multi_node_plots(args.node_pattern, out_dir)
        return

    # ---------------- Original single file/dir mode ----------------
    if not args.stats_file and not args.stats_dir:
        parser.error("Must specify either --stats-file/--stats-dir or --multi-node")

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
        files = sorted(
            os.path.join(stats_dir, p)
            for p in os.listdir(stats_dir)
            if p.endswith('.log') and os.path.isfile(os.path.join(stats_dir, p))
        )
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
        plot_scatter(row_vals[:len(col_vals)], col_vals[:len(row_vals)],
                     'Row vs Column differences', 'Row diff', 'Col diff', scatter_path)
        print(f'Wrote row vs col scatter: {scatter_path}')

    if (row_vals or col_vals):
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