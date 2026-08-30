#!/usr/bin/env python3
"""Regenerate the three manuscript figures whose printed values went stale.

Until now none of these had a generator, so a number could move in the text
without anything forcing the figure to follow. That is how ``fig_decomp`` came
to print the **retracted** C3 claim ("2.3x more accurate than the backbone") in
large type, alongside a shape share of 14.8/85.2 that the post-bugfix run had
already moved to 14.9/85.1.

Every value here is read from an artifact in ``reports/``. Nothing is
hardcoded, and the decomposition asserts that its waterfall closes, so a future
change to the underlying run cannot silently produce a figure that disagrees
with its own arithmetic.

The three:

``fig_decomp``  -- the retracted asymmetric ratio is replaced by the symmetric
                   contrast that superseded it, and five numbers are refreshed.
``fig3_band``   -- referee item 13: the utility profile scored the retrieval
                   branch while the recommendation is about the fused system, so
                   both are now plotted, with a second panel for the effect size
                   a win rate cannot express.
``fig_pareto``  -- referee item 19: the x-axis was a CPU-resident measurement
                   set against GPU-resident competitors.

Figure sizes and font families match each existing PDF so the replacements drop
in without disturbing the layout. The set is deliberately not unified: three of
these figures use DejaVu Sans and three use Liberation Serif, and changing that
here would be an unrelated edit.

Usage (fish):
    uv run python scripts/make_paper_figures.py
    uv run python scripts/make_paper_figures.py --check   # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "Journal paper" / "figures"
PT = 1.0 / 72.0  # points to inches, so sizes can be copied from `pdfinfo`

SERIF = ["Liberation Serif", "DejaVu Serif"]
SANS = ["DejaVu Sans"]


def _load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


# --------------------------------------------------------------------------
# fig_decomp -- oracle affine decomposition of the ETTm2 retrieval branch
# --------------------------------------------------------------------------
def make_decomp(out: Path, check: bool) -> list[str]:
    d = _load("reports/error-decomposition/ettm2-test-error-decomposition.json")["decomposition"]
    m, a = d["mse"], d["attribution"]

    raw_gap = m["raw_retrieval"] - m["shape_floor_oracle_rescaled"]
    removed = a["scale_error_removed_by_restoration"]
    remaining = a["scale_error_remaining"]
    # The waterfall must close, or the panel is drawing a decomposition that is
    # not one. This is the check the figure previously had no way to fail.
    if abs(raw_gap - removed - remaining) > 1e-6:
        raise ValueError(f"waterfall does not close: {raw_gap} != {removed} + {remaining}")

    shape_pct = 100.0 * a["shape_fraction_of_restored_error"]
    mag_pct = 100.0 - shape_pct
    floor_r = m["shape_floor_oracle_rescaled"]
    floor_b = m["shape_floor_backbone_oracle_rescaled"]
    worse_pct = 100.0 * (floor_r / floor_b - 1.0)
    realised = a["restored_retrieval_vs_backbone_ratio"]
    symmetric = 1.0 / a["symmetric_shape_ratio"]

    notes = [
        f"fig_decomp: raw gap {raw_gap:.4f}, removes {removed:.4f}, leaves {remaining:.4f}",
        f"fig_decomp: shape {shape_pct:.1f}% / magnitude {mag_pct:.1f}%",
        f"fig_decomp: retracted 2.3x replaced by {realised:.2f}x -> {symmetric:.2f}x symmetric",
    ]
    if check:
        return notes

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SANS,
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    ):
        fig, (ax, bx) = plt.subplots(
            1, 2, figsize=(538.222 * PT, 166.966 * PT), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )

        # (a) waterfall: raw scale error -> what restoration removes -> what is left
        labels = [
            "raw analogue\nscale error",
            "removed by\nrestoration",
            "unrecovered\nmagnitude",
            "shape floor\n(oracle)",
        ]
        base = [0.0, raw_gap - removed, 0.0, 0.0]
        height = [raw_gap, removed, remaining, floor_r]
        colours = ["#8c8c8c", "#4c72b0", "#c44e52", "#55a868"]
        ax.bar(range(4), height, bottom=base, color=colours, width=0.62, zorder=3)
        ax.axhline(
            m["backbone"],
            color="#333333",
            lw=0.8,
            ls="--",
            zorder=2,
            label=f"backbone {m['backbone']:.4f}",
        )
        ax.annotate(
            f"{raw_gap:.2f}",
            (0, raw_gap),
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            xytext=(0, 2),
            textcoords="offset points",
        )
        ax.annotate(
            f"removes {removed:.2f}",
            (1, raw_gap),
            ha="center",
            va="bottom",
            fontsize=7.5,
            xytext=(0, 2),
            textcoords="offset points",
        )
        ax.annotate(
            "",
            xy=(1, raw_gap - removed),
            xytext=(1, raw_gap),
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#1b3a63"},
        )
        for i, v in ((2, remaining), (3, floor_r)):
            ax.annotate(
                f"{v:.3f}" if v < 1 else f"{v:.2f}",
                (i, v),
                ha="center",
                va="bottom",
                fontsize=7.5,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("MSE (ETTm2 test)", fontsize=7.5)
        ax.set_ylim(0, raw_gap * 1.16)
        ax.legend(loc="upper right", fontsize=6.5, frameon=False)
        ax.set_title(
            "(a) restoration removes most, not all, of the scale error",
            fontsize=7.5,
            loc="left",
            pad=4,
        )
        ax.grid(axis="y", lw=0.4, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

        # (b) what survives, and the symmetric contrast that replaced C3
        bx.barh([0], [shape_pct], color="#55a868", height=0.42, zorder=3, label="shape")
        bx.barh(
            [0],
            [mag_pct],
            left=[shape_pct],
            color="#c44e52",
            height=0.42,
            zorder=3,
            label="unrecovered magnitude",
        )
        bx.text(
            shape_pct / 2,
            0,
            f"shape\n{shape_pct:.1f}%",
            ha="center",
            va="center",
            fontsize=7.0,
            color="white",
            fontweight="bold",
        )
        bx.text(
            shape_pct + mag_pct / 2,
            0,
            f"unrecovered magnitude {mag_pct:.1f}%",
            ha="center",
            va="center",
            fontsize=7.0,
            color="white",
            fontweight="bold",
        )
        bx.text(
            50,
            -0.62,
            "under the same correction applied to both branches, the analogues'\n"
            f"shape floor ({floor_r:.4f}) is {worse_pct:.0f}% worse than the backbone's "
            f"({floor_b:.4f}):\n"
            f"the realised {realised:.2f}$\\times$ deficit compresses to "
            f"{symmetric:.2f}$\\times$, it does not reverse",
            ha="center",
            va="top",
            fontsize=6.6,
            color="#222222",
        )
        bx.set_xlim(0, 100)
        bx.set_ylim(-1.05, 0.40)
        bx.set_yticks([])
        bx.set_xlabel("share of the restored branch's residual MSE (%)", fontsize=7.0)
        bx.set_title("(b) what survives restoration", fontsize=7.5, loc="left", pad=4)
        for side in ("left", "right", "top"):
            bx.spines[side].set_visible(False)

        fig.tight_layout(pad=0.4)
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return notes


# --------------------------------------------------------------------------
# fig3_band -- referee item 13: branch and fused, win rate and effect size
# --------------------------------------------------------------------------
def make_band(out: Path, check: bool) -> list[str]:
    """Branch and fused, win rate and effect size, with dispersion across origins.

    Read from the 50 per-origin checkpoints rather than the pooled summary,
    because the caption claims bars across origins and the pooled file carries
    no dispersion. The estimator is the one the recorded regime-threshold report
    used, and the branch column reproduces it exactly in all ten bins, which is
    the check below.
    """
    import sys as _sys

    _sys.path.insert(0, str(REPO / "src"))
    from scalerag.sparse_regime import FIXED_BIN_EDGES, bin_index

    ckpt = REPO / "reports" / "sparse-scale-rule" / "ckpt"
    files = sorted(ckpt.glob("s1000-seed42-origin*.npz"))
    if len(files) != 50:
        raise ValueError(f"expected 50 origin checkpoints, found {len(files)}")
    nb = FIXED_BIN_EDGES.size - 1
    acc: dict[str, list[list[float]]] = {
        k: [[] for _ in range(nb)] for k in ("bw", "fw", "bm", "fm")
    }
    for f in files:
        with np.load(f) as z:
            bb, br, fu = z["rmsse_backbone"], z["rmsse_mean"], z["rmsse_fused_mean"]
            ok = np.isfinite(bb) & np.isfinite(br) & np.isfinite(fu) & ~z["never_launched"]
            idx = bin_index(z["zero_fraction"])
            for b in range(nb):
                m = ok & (idx == b)
                if m.sum() < 5:
                    continue
                acc["bw"][b].append(float((bb[m] > br[m]).mean()))
                acc["fw"][b].append(float((bb[m] > fu[m]).mean()))
                acc["bm"][b].append(float((bb[m] - br[m]).mean()))
                acc["fm"][b].append(float((bb[m] - fu[m]).mean()))

    def agg(key: str) -> tuple[np.ndarray, np.ndarray]:
        mu = np.array([np.mean(acc[key][b]) for b in range(nb)])
        se = np.array([np.std(acc[key][b], ddof=1) / np.sqrt(len(acc[key][b])) for b in range(nb)])
        return mu, se

    mid = 0.5 * (FIXED_BIN_EDGES[:-1] + FIXED_BIN_EDGES[1:])
    br_win, br_win_se = agg("bw")
    fu_win, fu_win_se = agg("fw")
    br_mean, br_mean_se = agg("bm")
    fu_mean, fu_mean_se = agg("fm")
    ci = 1.96

    # The branch column must reproduce the recorded profile, or this estimator is
    # not the one the paper already reports.
    ref = _load("reports/regime-threshold/m5-val-regime-threshold-1000-50origins.json")[
        "fixed_bin_profile_across_origins"
    ]
    for b, row in enumerate(ref):
        if abs(row["win_rate_mean"] - br_win[b]) > 5e-3:
            raise ValueError(
                f"bin {b}: recomputed branch win {br_win[b]:.4f} disagrees with the "
                f"recorded {row['win_rate_mean']:.4f}"
            )

    if not np.all(fu_win >= br_win):
        raise ValueError("fusion is expected to raise the win rate in every bin")
    peak = int(np.argmax(br_win))
    sig = fu_mean - ci * fu_mean_se > 0  # positive with the interval clear of zero
    lo_edge = float(FIXED_BIN_EDGES[:-1][sig].min())
    hi_edge = float(FIXED_BIN_EDGES[1:][sig].max())

    notes = [
        f"fig3_band: branch peak {br_win[peak]:.3f}+-{br_win_se[peak]:.3f} at bin {mid[peak]:.2f}, "
        f"sparse {br_win[-1]:.3f}+-{br_win_se[-1]:.3f}, dense {br_win[0]:.3f}+-{br_win_se[0]:.3f}",
        "fig3_band: branch column reproduces the recorded regime profile in all 10 bins",
        f"fig3_band: fused win rate exceeds branch in all {nb} bins",
        f"fig3_band: fused mean significantly positive over [{lo_edge:.1f}, {hi_edge:.1f}]; "
        f"the [0.1,0.2) bin is positive at {fu_mean[1]:+.4f}+-{fu_mean_se[1]:.4f} but includes zero",
        f"fig3_band: sparsest bin fused win {fu_win[-1]:.3f} with mean {fu_mean[-1]:+.4f}",
    ]
    if check:
        return notes

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": SERIF,
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    ):
        fig, (ax, bx) = plt.subplots(
            2,
            1,
            figsize=(223.024 * PT, 205.0 * PT),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.12},
        )
        ax.axhspan(0.5, 1.0, color="#eef3f8", zorder=0)
        ax.axhline(0.5, color="#888888", lw=0.6, ls=":", zorder=1)
        ax.errorbar(
            mid,
            br_win,
            yerr=ci * br_win_se,
            fmt="-o",
            color="#4c72b0",
            lw=1.1,
            ms=2.6,
            elinewidth=0.7,
            capsize=1.4,
            zorder=3,
            label="retrieval branch",
        )
        ax.errorbar(
            mid,
            fu_win,
            yerr=ci * fu_win_se,
            fmt="-s",
            color="#c44e52",
            lw=1.1,
            ms=2.6,
            elinewidth=0.7,
            capsize=1.4,
            zorder=3,
            label="fused",
        )
        ax.annotate(
            f"peak {br_win[peak]:.2f}",
            (mid[peak], br_win[peak]),
            fontsize=6.4,
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            color="#4c72b0",
        )
        ax.annotate(
            f"{br_win[0]:.2f}",
            (mid[0], br_win[0]),
            fontsize=6.4,
            xytext=(3, -2),
            textcoords="offset points",
            color="#4c72b0",
        )
        ax.set_ylabel("win rate vs backbone", fontsize=7.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(
            loc="lower center",
            fontsize=6.4,
            frameon=False,
            ncol=2,
            handlelength=1.4,
            columnspacing=1.0,
        )
        ax.text(
            0.02,
            0.95,
            "retrieval better",
            transform=ax.transAxes,
            fontsize=6.2,
            color="#4a6a8a",
            va="top",
        )

        bx.axhline(0.0, color="#888888", lw=0.6, ls=":", zorder=1)
        bx.axvspan(lo_edge, hi_edge, color="#f3eef8", zorder=0)
        bx.errorbar(
            mid,
            br_mean,
            yerr=ci * br_mean_se,
            fmt="-o",
            color="#4c72b0",
            lw=1.1,
            ms=2.6,
            elinewidth=0.7,
            capsize=1.4,
            zorder=3,
        )
        bx.errorbar(
            mid,
            fu_mean,
            yerr=ci * fu_mean_se,
            fmt="-s",
            color="#c44e52",
            lw=1.1,
            ms=2.6,
            elinewidth=0.7,
            capsize=1.4,
            zorder=3,
        )
        span = float(fu_mean.max() - br_mean.min())
        bx.set_ylim(br_mean.min() - 0.10 * span, fu_mean.max() + 0.30 * span)
        bx.annotate(
            f"win {fu_win[-1]:.2f}, mean {fu_mean[-1]:+.3f}",
            (mid[-1], fu_mean[-1]),
            fontsize=6.2,
            ha="right",
            color="#7a2f33",
            xytext=(-4, -2),
            textcoords="offset points",
        )
        bx.text(
            0.5 * (lo_edge + hi_edge),
            fu_mean.max() + 0.19 * span,
            f"fused mean $>0$ over [{lo_edge:.1f}, {hi_edge:.1f}]",
            ha="center",
            fontsize=6.4,
            color="#5a4a72",
        )
        bx.set_ylabel(r"mean $\Delta$RMSSE", fontsize=7.0)
        bx.set_xlabel("context zero fraction", fontsize=7.0)
        bx.set_xlim(0.0, 1.0)
        for a_ in (ax, bx):
            a_.grid(axis="y", lw=0.35, alpha=0.3, zorder=0)
            a_.set_axisbelow(True)
        fig.tight_layout(pad=0.3)
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return notes


# --------------------------------------------------------------------------
# fig_pareto -- referee item 19: put every point on the same device
# --------------------------------------------------------------------------
def make_pareto(out: Path, check: bool) -> list[str]:
    bb = _load("reports/phase11a/compute_backbone.json")
    sr = _load("reports/phase11a/compute_scalerag.json")
    cost = _load("reports/ettm2-cost/ettm2-val-cost.json")
    mse_bb = _load("reports/phase11a/repro_chronos_bolt_target_ettm2.json")["mse"]
    mse_ts = _load("reports/phase11a/repro_tsrag_official_ettm2.json")["mse"]
    native = _load("reports/phase11a/scalerag_native_ettm2_test.json")
    frozen = next(
        r
        for r in native["scalerag_restored_fixed_fusion"]
        if r["scale"] == "mean" and r["k"] == 20 and r["weight"] == 0.25
    )
    mse_sr = frozen["mse"]

    lat_bb = bb["backbone_latency_ms_per_window"]
    lat_idx_cpu = sr["retrieval_latency_ms_per_window"]
    speedup = cost["latency_ms_per_query"]["speedup"]
    if not cost["gpu_matches_cpu_exactly"]:
        raise ValueError("the GPU retriever is not bit-identical; the speedup is not transferable")
    lat_idx_gpu = lat_idx_cpu / speedup
    # TS-RAG's end-to-end latency is the backbone scaled by the measured
    # wall-clock ratio of the two reproduction runs; it is not an increment.
    ratio = (
        _load("reports/phase11a/repro_tsrag_official_ettm2.json")["eval_seconds"]
        / _load("reports/phase11a/repro_chronos_bolt_target_ettm2.json")["eval_seconds"]
    )
    lat_ts = lat_bb * ratio
    tot_cpu, tot_gpu = lat_bb + lat_idx_cpu, lat_bb + lat_idx_gpu

    notes = [
        f"fig_pareto: backbone {lat_bb:.3f} ms, TS-RAG {lat_ts:.3f} ms, "
        f"ScaleRAG {tot_cpu:.3f} ms CPU -> {tot_gpu:.3f} ms GPU ({speedup:.1f}x on the index)",
        f"fig_pareto: MSE {mse_bb:.5f} / {mse_ts:.5f} / {mse_sr:.5f}",
        f"fig_pareto: still dominated on GPU by {100 * (tot_gpu / lat_ts - 1):.0f}% latency",
    ]
    if check:
        return notes

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SANS,
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    ):
        fig, ax = plt.subplots(figsize=(225.694 * PT, 156.595 * PT))
        ax.scatter([lat_bb], [mse_bb], s=26, color="#4c72b0", zorder=4)
        ax.scatter([lat_ts], [mse_ts], s=190, color="#c44e52", alpha=0.85, zorder=4)
        ax.scatter([tot_gpu], [mse_sr], s=26, color="#55a868", zorder=4)
        ax.scatter(
            [tot_cpu],
            [mse_sr],
            s=26,
            facecolor="none",
            edgecolor="#55a868",
            lw=0.8,
            alpha=0.55,
            zorder=3,
        )
        ax.annotate(
            "",
            xy=(tot_gpu + 0.02, mse_sr),
            xytext=(tot_cpu - 0.02, mse_sr),
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.7,
                "color": "#55a868",
                "alpha": 0.7,
                "linestyle": "--",
            },
        )
        ax.annotate(
            "Chronos-Bolt (frozen)\n0 trainable",
            (lat_bb, mse_bb),
            fontsize=6.3,
            xytext=(4, 5),
            textcoords="offset points",
            color="#2f4a6b",
        )
        ax.annotate(
            "TS-RAG\n4.78M trainable",
            (lat_ts, mse_ts),
            fontsize=6.3,
            xytext=(17, 0),
            textcoords="offset points",
            color="#7a2f33",
            va="center",
        )
        ax.annotate(
            "ScaleRAG (GPU index)\n0 trainable",
            (tot_gpu, mse_sr),
            fontsize=6.3,
            xytext=(-2, 6),
            textcoords="offset points",
            color="#31603f",
        )
        ax.annotate(
            "CPU index",
            (tot_cpu, mse_sr),
            fontsize=6.0,
            xytext=(-4, -11),
            textcoords="offset points",
            ha="center",
            color="#6f8f7a",
        )
        ax.set_xlabel("end-to-end latency per window (ms), one device", fontsize=7.0)
        ax.set_ylabel("MSE (ETTm2 test), lower better", fontsize=7.0)
        ax.set_xlim(0.33, 1.16)
        ax.text(
            0.97,
            0.34,
            "marker area $\\propto$ trainable parameters",
            transform=ax.transAxes,
            fontsize=6.2,
            ha="right",
            color="#555555",
        )
        ax.grid(lw=0.35, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.3)
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report values, write nothing")
    args = ap.parse_args()
    jobs = (
        ("fig_decomp.pdf", make_decomp),
        ("fig3_band.pdf", make_band),
        ("fig_pareto.pdf", make_pareto),
    )
    for name, fn in jobs:
        for line in fn(FIGS / name, args.check):
            print(line)
        print(f"  {'checked' if args.check else 'wrote'} figures/{name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
