from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def plot_performance(times_ms: np.ndarray, labels: list[str], out: str | None = None) -> None:
	"""Plot times (ms) as a bar chart and annotate overhead vs baseline.

	Args:
	  times_ms: array of shape (4,) with times in milliseconds.
	  labels: list of 4 labels for the bars.
	  out: optional path to save the figure. If None, the figure is shown.
	"""
	if len(times_ms) != 4 or len(labels) != 4:
		raise ValueError("Expected four columns/times and four labels")

	baseline = float(times_ms[0])
	overhead_pct = (times_ms - baseline) / baseline * 100.0

	x = np.arange(len(labels))
	fig, ax = plt.subplots(figsize=(8, 4.5))
	bars = ax.bar(x, times_ms, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], edgecolor="k")

	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=20, ha="right")
	ax.set_ylabel("Time (ms)")
	ax.set_title("ResNet sample: baseline vs ABFT configurations")
	ax.grid(axis="y", linestyle="--", alpha=0.4)

	# Annotate each bar with time and overhead (for ABFT bars)
	for i, bar in enumerate(bars):
		h = bar.get_height()
		ax.annotate(f"{h:.0f} ms", xy=(bar.get_x() + bar.get_width() / 2, h),
					xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)

		if i != 0:
			pct = overhead_pct[i]
			sign = "+" if pct >= 0 else ""
			ax.annotate(f"{sign}{pct:.1f}%",
						xy=(bar.get_x() + bar.get_width() / 2, h),
						xytext=(0, 20), textcoords="offset points", ha="center", va="bottom", fontsize=9,
						color="#444444")

	top = float(np.max(times_ms))
	y_padding = max(top * 0.18, 10.0)
	ax.set_ylim(0.0, top + y_padding)

	plt.tight_layout()

	if out:
		# Use bbox_inches='tight' to avoid clipping annotations when saving.
		fig.savefig(out, dpi=200, bbox_inches="tight")
		print(f"Saved plot to {out}")
	else:
		plt.show()


def main(argv: list[str] | None = None) -> int:
	argv = argv if argv is not None else sys.argv[1:]
	parser = argparse.ArgumentParser(description="Plot ResNet performance and ABFT overheads")
	parser.add_argument("--out", "-o", help="Output image path (PNG). If omitted, show interactively.")
	args = parser.parse_args(argv)

	# Values taken from local benchmark comments (Debug build).
	# Represent the data as a mapping from label -> time (ms).
	# Use an insertion-ordered dict so labels/times keep the intended order.
	data = {
		"Baseline": 118.0,
		"ABFT (FIC)": 130.0,
		"ABFT (FuC)": 133.0,
		"ABFT (FuC + Scaling)": 135.0,
	}

	# Unpack into the list/array form expected by the plotting function.
	labels = list(data.keys())
	times_ms = np.array(list(data.values()))

	# Print overheads to stdout as well
	baseline = times_ms[0]
	overheads = (times_ms - baseline) / baseline * 100.0
	print("Times (ms):")
	for lbl, t, ov in zip(labels, times_ms, overheads):
		if lbl == "Baseline":
			print(f"  {lbl}: {t:.0f} ms (baseline)")
		else:
			print(f"  {lbl}: {t:.0f} ms ({ov:+.1f}% vs baseline)")

	plot_performance(times_ms, labels, out=args.out)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())



