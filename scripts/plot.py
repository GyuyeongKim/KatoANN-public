#!/usr/bin/env python3
"""Draw recall@10 against throughput for every measured system and dataset, and
print the highest throughput of each system at recall 0.99, 0.995, and 0.999
or more.
usage: plot.py RESULTS_DIR
Reads RESULTS_DIR/<dataset>/<system>[.<part>].jsonl (one JSON line per operating
point, with "recall" and "qps"; parts such as <system>.fine.jsonl join their
system), draws each system's best throughput at each recall from 0.99 on (its
Pareto front), and writes RESULTS_DIR/recall_throughput.svg/.png and RESULTS_DIR/summary.csv.
"""
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = [("openai3072-1m", "OpenAI-3072-1M"), ("bioasq-10m", "BioASQ-1024-10M")]
STYLE = {
    "katoann": ("KatoANN", "#0062CC", "o", "-"),
    "svs-leanvec": ("SVS (Vamana, LeanVec)", "#C05A8A", "s", "--"),
    "scann": ("ScaNN", "#D9822B", "P", "--"),
    "cabana": ("CABANA (bfloat16 on AMX)", "#3E8E7E", "^", "--"),
    "faiss-ivf": ("Faiss IVF-Flat", "#7F7F7F", "D", ":"),
    "hnsw": ("HNSW (hnswlib)", "#C99A2E", "v", ":"),
}
ORDER = list(STYLE)


def load(paths):
    pts = []
    for path in paths:
        for line in open(path):
            d = json.loads(line)
            if "recall" in d and "qps" in d:
                pts.append((d["recall"], d["qps"], d))
    # Pareto front: keep a point only if no point with higher recall is faster
    front, best = [], 0.0
    for p in sorted(pts, key=lambda p: (-p[0], -p[1])):
        if p[1] > best:
            front.append(p)
            best = p[1]
    return sorted(front, key=lambda p: p[0])


def system_of(path):
    return os.path.basename(path)[:-len(".jsonl")].split(".")[0]


# The x axis counts nines of recall, -log10(1 - recall), so that 0.995, 0.998, and
# 0.999 get room; a linear axis squeezes the high-recall end into a corner. It starts
# at recall 0.99, the low end of the range this system is for.
XMIN = 0.99
TICKS = [0.99, 0.995, 0.998, 0.999, 0.9995]


def nines(r):
    return -np.log10(np.maximum(1e-6, 1.0 - np.asarray(r, dtype=float)))


def main():
    out = sys.argv[1]
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    family = [f for f in ("Helvetica", "Arial", "Nimbus Sans", "Liberation Sans") if f in have] or ["DejaVu Sans"]
    plt.rcParams.update({"font.family": family[:1], "font.size": 10, "svg.fonttype": "path"})
    present = [(k, t) for k, t in DATASETS if os.path.isdir(os.path.join(out, k))]
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 3.5), squeeze=False)
    rows = []
    for ax, (key, title) in zip(axes[0], present):
        groups = {}
        for path in glob.glob(os.path.join(out, key, "*.jsonl")):
            groups.setdefault(system_of(path), []).append(path)
        names = [n for n in ORDER if n in groups]  # files of other names (old runs) are ignored
        for name in names:
            pts = load(groups[name])
            if not pts:
                continue
            label, color, marker, ls = STYLE[name]
            # the last point below XMIN stays, so the curve enters at the left edge
            below = [i for i, p in enumerate(pts) if p[0] < XMIN]
            shown = pts[below[-1]:] if below else pts
            ax.plot(nines([p[0] for p in shown]), [p[1] for p in shown], color=color, marker=marker,
                    linestyle=ls, linewidth=1.8, markersize=5, label=label)
            for target in (0.99, 0.995, 0.999):
                ok = [p for p in pts if p[0] >= target]
                best = max(ok, key=lambda p: p[1]) if ok else None
                rows.append((key, label, f"{target}", f"{best[1]:.0f}" if best else "",
                             f"{best[0]:.4f}" if best else "", json.dumps(best[2]) if best else ""))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Recall@10")
        ax.set_xlim(nines(XMIN), nines(0.9998))
        ax.set_xticks(nines(TICKS))
        ax.set_xticklabels([f"{t:g}" for t in TICKS])
        ax.set_yscale("log")
        ax.grid(alpha=0.3, linewidth=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("Throughput (QPS)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "recall_throughput.svg"), metadata={"Date": None}, facecolor="white")
    fig.savefig(os.path.join(out, "recall_throughput.png"), dpi=150, facecolor="white")
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)  # system labels contain commas
        w.writerow(["dataset", "system", "recall_target", "qps", "recall", "point"])
        w.writerows(rows)
    for r in rows:
        print(f"{r[0]:15s} {r[1]:26s} recall >= {r[2]}: {r[3] or 'not reached':>8s} QPS {r[4]}")


if __name__ == "__main__":
    main()
