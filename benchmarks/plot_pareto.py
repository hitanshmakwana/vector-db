"""Turn benchmark CSVs into the committed plots.

Plots read CSV and never touch a live index (BENCHMARKS.md: "results go to CSV,
plots read from CSV. Reproducibility is a feature"). Re-plotting after a tweak to
the styling must never mean re-running a multi-hour index build.

    python -m benchmarks.plot_pareto                      # every CSV it finds
    python -m benchmarks.plot_pareto --csv results/sift_1m.csv

Produces, per BENCHMARKS.md:

* ``plots/hnsw_sift1m_pareto.png``  — recall@10 vs QPS, PyVec vs FAISS
* ``plots/ivf_sift1m_pareto.png``   — one curve per nlist
* ``plots/hybrid_msmarco.png``      — grouped bars per metric

matplotlib is a demo extra, not a core dependency. When it is missing this script
still prints an ASCII rendering of each chart — the same shape as the sketch in
BENCHMARKS.md — so the numbers stay readable in a terminal or a CI log.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from benchmarks.harness import PLOTS_DIR, RESULTS_DIR, try_import_matplotlib

K = 10

#: ASCII-only scatter markers, in assignment order.
_MARKERS = "o^sx+*"

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Two different colour *jobs* are in play here, and they get different treatment:
#
# * **Which system** (PyVec / FAISS / brute force) is categorical — identity, no
#   order — so it takes the first three categorical slots in fixed order. Three,
#   not more: on a scatter/line chart any pair of series can end up adjacent, and
#   under all-pairs comparison this palette only clears the colour-blind separation
#   floors for its first three slots. That cap is why the IVF curves live in their
#   own figure instead of being crammed in beside these.
# * **nlist** is *ordered* magnitude (64 -> 4096), so it takes a single-hue
#   light-to-dark ordinal ramp. Giving four ordered values four unrelated hues (the
#   first version of this file did) throws away the ordering and produced four
#   indistinguishable green lines.
#
# Both sets were checked with the palette validator rather than by eye:
# categorical trio passes all-pairs (worst CVD ΔE 9.2, normal-vision 24.0); the
# ordinal ramp passes monotone-lightness, step-gap and light-end contrast.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"

SERIES_STYLE = {
    "pyvec-hnsw": {"color": "#2a78d6", "marker": "o", "label": "PyVec HNSW"},
    "faiss-hnsw": {"color": "#eb6834", "marker": "^", "label": "FAISS HNSW"},
    "pyvec-flat": {"color": "#1baf7a", "marker": "s", "label": "Brute force (exact)"},
    "pyvec-ivf": {"color": "#2a78d6", "marker": "D", "label": "PyVec IVF-Flat"},
}

#: Ordinal ramp for nlist, light -> dark. Light end stays above 2:1 on the surface.
NLIST_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#104281"]

#: Mark specs from the design guidance: 2px strokes, markers >= 8px, a surface-
#: coloured ring so overlapping markers stay countable.
LINE_WIDTH = 2.0
MARKER_SIZE = 7.0
MARKER_EDGE = 1.4


def _pretty_dataset(raw: Any, fallback: str) -> str:
    """Turn an internal dataset label into something fit for a chart title.

    ``sift-1m-100000`` is a precise machine label and a terrible axis title; a
    reader needs to know it is SIFT and that it is a 100k subset, not the full 1M.
    """
    name = str(raw or fallback)
    if name.startswith("sift-1m-"):
        try:
            return f"SIFT-1M, first {int(name.rsplit('-', 1)[1]):,} vectors"
        except ValueError:
            pass
    if name == "sift-1m":
        return "SIFT-1M (all 1,000,000 vectors)"
    if name.startswith("glove-100-"):
        try:
            return f"GloVe-100, first {int(name.rsplit('-', 1)[1]):,} vectors"
        except ValueError:
            pass
    if name == "glove-100":
        return "GloVe-100 (all 1,183,514 vectors)"
    if "synthetic" in name:
        return f"{name}  [SYNTHETIC — not a real dataset]"
    return name


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", None):
                row[key] = None
                continue
            try:
                row[key] = int(value)
            except ValueError:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


# --------------------------------------------------------------------------- #
# ASCII rendering — always available
# --------------------------------------------------------------------------- #


def ascii_scatter(
    series: dict[str, list[tuple[float, float]]],
    title: str,
    x_label: str = "QPS (log scale)",
    y_label: str = f"recall@{K}",
    width: int = 62,
    height: int = 18,
) -> str:
    """A log-x scatter in text, in the style of the sketch in BENCHMARKS.md."""
    import math

    points = [p for pts in series.values() for p in pts]
    if not points:
        return f"{title}\n  (no data)"

    xs = [max(p[0], 1e-9) for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = math.log10(min(xs)), math.log10(max(xs))
    if x_max - x_min < 1e-9:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    y_min, y_max = min(ys), max(ys)
    y_pad = max((y_max - y_min) * 0.1, 0.01)
    y_min, y_max = max(0.0, y_min - y_pad), min(1.0, y_max + y_pad)
    if y_max - y_min < 1e-9:
        y_min, y_max = y_min - 0.05, y_max + 0.05

    grid = [[" "] * width for _ in range(height)]
    # Strictly ASCII markers. The Windows console defaults to cp1252, which cannot
    # encode box-drawing or geometric-shape characters, and a benchmark script
    # that raises UnicodeEncodeError while printing its own results is worse than
    # one with plain markers.
    markers = {name: _MARKERS[i % len(_MARKERS)] for i, name in enumerate(series)}
    for name, pts in series.items():
        for x, y in pts:
            col = int((math.log10(max(x, 1e-9)) - x_min) / (x_max - x_min) * (width - 1))
            rowpos = int((y - y_min) / (y_max - y_min) * (height - 1))
            grid[height - 1 - rowpos][max(0, min(width - 1, col))] = markers[name]

    lines = [title, ""]
    for i, row in enumerate(grid):
        value = y_max - (y_max - y_min) * i / (height - 1)
        lines.append(f"  {value:5.3f} |{''.join(row)}")
    lines.append("        +" + "-" * width)
    lines.append(
        f"         {10 ** x_min:<10,.0f}"
        + " " * max(0, width - 24)
        + f"{10 ** x_max:>10,.0f}"
    )
    lines.append(f"         {x_label}   (y: {y_label})")
    lines.append("")
    for name, marker in markers.items():
        label = SERIES_STYLE.get(name, {}).get("label", name)
        lines.append(f"   {marker} {label}  ({len(series[name])} points)")
    return "\n".join(lines)


def ascii_bars(
    groups: dict[str, dict[str, float]], title: str, width: int = 34
) -> str:
    """Grouped horizontal bars: ``{metric: {system: value}}``."""
    lines = [title, ""]
    for metric, systems in groups.items():
        lines.append(f"  {metric}")
        peak = max(systems.values()) if systems else 0.0
        for name, value in systems.items():
            filled = int(value / peak * width) if peak > 0 else 0
            lines.append(
                f"    {name:14} {'#' * filled}{'.' * (width - filled)} {value:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Pareto plots
# --------------------------------------------------------------------------- #


def _pareto_front(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only points not dominated on both axes (higher QPS *and* recall)."""
    ordered = sorted(points, key=lambda p: (-p[0], -p[1]))
    front: list[tuple[float, float]] = []
    best_recall = -1.0
    for qps, recall in ordered:
        if recall > best_recall:
            front.append((qps, recall))
            best_recall = recall
    return sorted(front)


def _style_axes(ax, title: str, subtitle: str | None = None) -> None:
    """Recessive grid and axes, ink-coloured text, no chartjunk."""
    ax.set_xscale("log")
    ax.set_xlabel(
        "Queries per second — single thread, log scale", fontsize=9,
        color=INK_SECONDARY,
    )
    ax.set_ylabel(f"recall@{K}", fontsize=9, color=INK_SECONDARY)
    # Title above subtitle, both left-aligned to the plot area. The pad has to
    # clear the subtitle's own line height or the two overlap.
    ax.set_title(title, fontsize=12.5, color=INK_PRIMARY,
                 pad=30 if subtitle else 10, loc="left")
    if subtitle:
        ax.annotate(
            subtitle, xy=(0, 1), xycoords="axes fraction",
            xytext=(0, 9), textcoords="offset points",
            fontsize=8.5, color=INK_SECONDARY, va="bottom", annotation_clip=False,
        )
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=3)


def _recall_reference(ax, level: float = 0.95) -> None:
    ax.axhline(level, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax.annotate(
        f"{level:.0%} recall target", xy=(1, level), xycoords=("axes fraction", "data"),
        xytext=(-4, 4), textcoords="offset points",
        fontsize=7.5, color=INK_SECONDARY, ha="right", va="bottom",
    )


def _plot_curve(ax, points, *, color, marker, label, annotations=None,
                annotate_every=1, ends_only=False, zorder=3):
    pts = sorted(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(
        xs, ys, color=color, marker=marker, label=label,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        markeredgecolor=SURFACE, markeredgewidth=MARKER_EDGE,
        zorder=zorder, solid_capstyle="round",
    )
    if annotations:
        ordered = [a for _, a in sorted(zip(points, annotations), key=lambda t: t[0])]
        # `ends_only` exists for the multi-curve figure: four curves times five
        # points is twenty labels. Only the *fast* end gets one — at the slow end
        # every curve has saturated at recall 1.0 in the same small region, so those
        # labels stack on top of each other and say nothing the reader cannot see.
        # The parameter range goes in the subtitle instead.
        keep = {len(pts) - 1} if ends_only else None
        for i, ((x, y), note) in enumerate(zip(pts, ordered)):
            # Annotate sparsely and alternate the offset.
            if keep is not None:
                if i not in keep:
                    continue
            elif i % annotate_every:
                continue
            above = i % 2 == 0
            ax.annotate(
                note, (x, y), fontsize=7, color=INK_SECONDARY,
                textcoords="offset points",
                xytext=(0, 9) if above else (0, -14),
                ha="center", va="bottom" if above else "top", zorder=5,
            )
    return pts


def _finish(fig, ax, out: Path, plt, note: str = "up and to the right is better"):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.text(0.995, 0.005, note, ha="right", fontsize=7, color=INK_MUTED)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out}", file=sys.stderr)


def plot_systems_pareto(rows: list[dict], out: Path, dataset: str) -> None:
    """Benchmark 1: PyVec HNSW against FAISS and brute force.

    Three series maximum — see the palette note. This is the headline chart.
    """
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    annotations: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        system = row.get("system")
        recall, qps = row.get(f"recall@{K}"), row.get("qps")
        if system not in ("pyvec-hnsw", "faiss-hnsw", "pyvec-flat"):
            continue
        if recall is None or qps is None:
            continue
        series[system].append((float(qps), float(recall)))
        annotations[system].append(
            f"ef={row.get('ef_search')}" if row.get("ef_search") else "exact"
        )

    if not series:
        return
    print()
    print(ascii_scatter(series, f"HNSW recall-QPS ({dataset})  (ASCII)"))

    plt = try_import_matplotlib()
    if plt is None:
        print("  matplotlib not installed, PNG skipped.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
    for system in ("pyvec-hnsw", "faiss-hnsw", "pyvec-flat"):
        if system not in series:
            continue
        style = SERIES_STYLE[system]
        single_point = len(series[system]) == 1
        _plot_curve(
            ax, series[system], color=style["color"], marker=style["marker"],
            label=style["label"],
            annotations=None if single_point else annotations[system],
        )
        # Direct label at the fast end of each curve, in ink rather than the series
        # colour — the marker beside it already carries identity, and one of these
        # hues sits below 3:1 on this surface so it must not be load-bearing text.
        # Skipped for a single-point series (brute force), where there is no curve
        # to disambiguate and the label just collides with the sweep annotations.
        if single_point:
            continue
        pts = sorted(series[system])
        x, y = pts[-1]
        ax.annotate(
            style["label"], (x, y), fontsize=8.5, color=INK_PRIMARY,
            textcoords="offset points", xytext=(9, 0), va="center", zorder=6,
        )

    _recall_reference(ax)
    _style_axes(
        ax,
        f"PyVec HNSW vs FAISS — recall-QPS Pareto, {dataset}",
        "Same build parameters (M=16, ef_construction=200). Each point is one "
        "ef_search value.",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK_SECONDARY)
    ax.margins(x=0.16)
    _finish(fig, ax, out, plt)


def plot_ivf_pareto(rows: list[dict], out: Path, dataset: str) -> None:
    """Benchmark 2: IVF-Flat, one curve per nlist, on an ordinal colour ramp."""
    by_nlist: dict[int, list[tuple[float, float]]] = defaultdict(list)
    notes: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("system") != "pyvec-ivf":
            continue
        recall, qps, nlist = row.get(f"recall@{K}"), row.get("qps"), row.get("nlist")
        if recall is None or qps is None or nlist is None:
            continue
        by_nlist[int(nlist)].append((float(qps), float(recall)))
        notes[int(nlist)].append(f"nprobe={row.get('nprobe')}")
    if not by_nlist:
        return

    print()
    print(ascii_scatter(
        {f"nlist={n}": p for n, p in sorted(by_nlist.items())},
        f"IVF-Flat recall-QPS ({dataset})  (ASCII)",
    ))

    plt = try_import_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
    ordered = sorted(by_nlist)
    for position, nlist in enumerate(ordered):
        # Ordinal ramp: darker = more centroids. Reading order is the data order.
        color = NLIST_RAMP[
            min(
                len(NLIST_RAMP) - 1,
                round(position * (len(NLIST_RAMP) - 1) / max(len(ordered) - 1, 1)),
            )
        ]
        _plot_curve(
            ax, by_nlist[nlist], color=color, marker="o",
            label=f"nlist = {nlist:,}",
            annotations=notes[nlist], ends_only=True,
            zorder=3 + position,
        )
        pts = sorted(by_nlist[nlist])
        ax.annotate(
            f"{nlist:,}", pts[-1], fontsize=8, color=INK_PRIMARY,
            textcoords="offset points", xytext=(8, 0), va="center", zorder=6,
        )

    probes = sorted(
        {int(r["nprobe"]) for r in rows
         if r.get("system") == "pyvec-ivf" and r.get("nprobe") is not None}
    )
    _recall_reference(ax)
    _style_axes(
        ax,
        f"PyVec IVF-Flat — recall-QPS Pareto, {dataset}",
        f"One curve per nlist (darker = more centroids). Each point is one nprobe, "
        f"rising from {probes[0]} to {probes[-1]} right to left.",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK_SECONDARY,
              title="centroids", title_fontsize=8)
    ax.margins(x=0.16)
    _finish(fig, ax, out, plt)


def plot_hnsw_vs_ivf(rows: list[dict], out: Path, dataset: str) -> None:
    """The comparison BENCHMARKS.md asks to surface: which index wins, and by how
    much. Only the best frontier per family, so it is two lines and readable."""
    frontier: dict[str, list[tuple[float, float]]] = {}
    for system, key in (("pyvec-hnsw", None), ("pyvec-ivf", "nlist")):
        points = [
            (float(r["qps"]), float(r[f"recall@{K}"]))
            for r in rows
            if r.get("system") == system
            and r.get("qps") is not None
            and r.get(f"recall@{K}") is not None
        ]
        if points:
            frontier[system] = _pareto_front(points)
    if len(frontier) < 2:
        return

    plt = try_import_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=150)
    for system in ("pyvec-hnsw", "pyvec-ivf"):
        if system not in frontier:
            continue
        style = SERIES_STYLE[system]
        color = "#2a78d6" if system == "pyvec-hnsw" else "#eb6834"
        _plot_curve(
            ax, frontier[system], color=color, marker=style["marker"],
            label=style["label"],
        )
        pts = sorted(frontier[system])
        ax.annotate(
            style["label"], pts[-1], fontsize=8.5, color=INK_PRIMARY,
            textcoords="offset points", xytext=(8, 0), va="center", zorder=6,
        )

    _recall_reference(ax)
    _style_axes(
        ax,
        f"HNSW vs IVF-Flat — best achievable frontier, {dataset}",
        "Pareto-optimal points only, across every parameter setting tried.",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK_SECONDARY)
    ax.margins(x=0.18)
    _finish(fig, ax, out, plt)


def plot_hybrid(rows: list[dict], out: Path) -> None:
    """Benchmark 3: grouped bars, three systems per metric."""
    metrics = [f"ndcg@{K}", f"mrr@{K}", "recall@100"]
    groups: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = {
            str(r["system"]): float(r[metric])
            for r in rows
            if r.get(metric) is not None and r.get("system")
        }
        if values:
            groups[metric] = values
    if not groups:
        print("  no plottable hybrid rows", file=sys.stderr)
        return

    print()
    print(ascii_bars(groups, "Hybrid vs dense vs BM25  (ASCII)"))

    plt = try_import_matplotlib()
    if plt is None:
        print("  matplotlib not installed, PNG skipped.", file=sys.stderr)
        return

    import numpy as np

    systems = list(dict.fromkeys(s for g in groups.values() for s in g))
    colors = {"dense": "#0072B2", "bm25": "#E69F00", "hybrid-rrf": "#009E73"}
    x = np.arange(len(groups))
    bar_width = 0.8 / max(len(systems), 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    for i, system in enumerate(systems):
        heights = [groups[m].get(system, 0.0) for m in groups]
        offset = (i - (len(systems) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, heights, bar_width,
                      label=system, color=colors.get(system), alpha=0.9)
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1)

    ax.set_xticks(x, list(groups))
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Hybrid retrieval (RRF) vs dense-only vs BM25-only")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", nargs="*", default=None,
                        help="specific CSVs; defaults to everything in results/")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = (
        [Path(p) for p in args.csv]
        if args.csv
        else sorted(results_dir.glob("*.csv"))
    )
    if not paths:
        print(
            f"no CSVs in {results_dir}. Run a benchmark first, e.g.\n"
            f"    python -m benchmarks.sift_1m --synthetic --n 20000",
            file=sys.stderr,
        )
        return 1

    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        print(f"\n{'=' * 70}\n{path.name}  ({len(rows)} rows)\n{'=' * 70}",
              file=sys.stderr)
        stem = path.stem

        if any(r.get("system") in ("dense", "bm25", "hybrid-rrf") for r in rows):
            plot_hybrid(rows, plots_dir / f"{stem}.png")
            continue

        label = _pretty_dataset(rows[0].get("dataset"), stem)
        plot_systems_pareto(rows, plots_dir / f"hnsw_{stem}_pareto.png", label)
        plot_ivf_pareto(rows, plots_dir / f"ivf_{stem}_pareto.png", label)
        plot_hnsw_vs_ivf(rows, plots_dir / f"hnsw_vs_ivf_{stem}.png", label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
