"""Render the oracle affine decomposition figure from the recorded run.

Every value is read from ``docs/error-decomposition-ettm2-test.json``.
Nothing is hardcoded, so the figure cannot drift from the run that produced it.

Panel (a) is a waterfall over the retrieval branch's MSE on ETTm2 test:

    raw analogue scale error  =  raw retrieval MSE - shape floor
    removed by restoration    =  the decrement the restoration step achieves
    unrecovered magnitude     =  what the decrement leaves behind
    shape floor               =  the oracle affine correction, a lower bound

Panel (b) splits the restored branch's residual into its shape and magnitude
shares.

Note on the decrement label. The y axis is an MSE, which is non-negative by
definition, so the middle bar is annotated with a downward arrow and a positive
magnitude rather than a negative number. An earlier revision printed "-2.56" on
this axis; the geometry was right but a negative tick on a non-negative
quantity reads as an error.

Usage
-----
    uv run python scripts/make_decomposition_figure.py \
        --results docs/error-decomposition-ettm2-test.json \
        --outdir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ORANGE_EDGE = "#D9541E"
ORANGE_FILL = "#FBE3D5"
BLUE_EDGE = "#2E7EBB"
BLUE_FILL = "#DCE9F7"
BLUE_SOLID = "#2E7EBB"
PURPLE = "#7B3FA0"
GREY = "#666666"

RC = {
    "font.size": 7,
    "axes.titlesize": 7.6,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#999999",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def load_decomposition(path: Path) -> dict[str, float]:
    """Pull the six quantities the figure needs out of the run record.

    Fails loudly if a key is absent: a silently defaulted value here would
    produce a plausible-looking but wrong figure, which is worse than a crash.
    """
    payload = json.loads(path.read_text())
    mse = payload["decomposition"]["mse"]
    attribution = payload["decomposition"]["attribution"]

    raw = float(mse["raw_retrieval"])
    shape_floor = float(mse["shape_floor_oracle_rescaled"])
    backbone = float(mse["backbone"])
    removed = float(attribution["scale_error_removed_by_restoration"])
    remaining = float(attribution["scale_error_remaining"])
    shape_fraction = float(attribution["shape_fraction_of_restored_error"])

    scale_error = raw - shape_floor
    closure = abs(scale_error - (removed + remaining))
    if closure > 1e-6:
        raise ValueError(
            f"waterfall does not close: {scale_error:.8f} != "
            f"{removed:.8f} + {remaining:.8f} (gap {closure:.2e})"
        )

    return {
        "scale_error": scale_error,
        "removed": removed,
        "remaining": remaining,
        "shape_floor": shape_floor,
        "backbone": backbone,
        "shape_fraction": shape_fraction,
        "n_windows": int(payload["decomposition"]["n_windows"]),
    }


def _waterfall(ax: plt.Axes, d: dict[str, float]) -> None:
    ax.set_title("(a) restoration removes most, not all, of the scale error", loc="left")

    labels = [
        "raw analogue\nscale error",
        "removed by\nrestoration",
        "unrecovered\nmagnitude",
        "shape floor\n(oracle)",
    ]
    bottoms = [0.0, d["remaining"], 0.0, 0.0]
    heights = [d["scale_error"], d["removed"], d["remaining"], d["shape_floor"]]
    faces = [ORANGE_FILL, BLUE_FILL, BLUE_SOLID, PURPLE]
    edges = [ORANGE_EDGE, BLUE_EDGE, BLUE_SOLID, PURPLE]

    ax.bar(
        range(4), heights, bottom=bottoms, width=0.62, color=faces, edgecolor=edges, linewidth=1.3
    )

    top = d["scale_error"]
    ax.set_ylim(0, top * 1.16)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("MSE (ETTm2 test)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)

    # Bar 1, 3, 4: plain magnitude above the bar.
    for i in (0, 2, 3):
        y = bottoms[i] + heights[i]
        ax.text(
            i,
            y + top * 0.022,
            f"{heights[i]:.3g}".rstrip("0").rstrip(".") if i != 0 else f"{heights[i]:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=7.6,
        )

    # Bar 2 is a decrement. The axis is an MSE and cannot go negative, so the
    # direction is carried by an arrow and the word "removes", not by a minus.
    mid_top = bottoms[1] + heights[1]
    ax.text(
        1,
        mid_top + top * 0.022,
        f"removes {d['removed']:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=7.6,
        color=BLUE_EDGE,
    )
    ax.add_patch(
        FancyArrowPatch(
            (1, mid_top - top * 0.07),
            (1, bottoms[1] + top * 0.05),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
            color=BLUE_EDGE,
            shrinkA=0,
            shrinkB=0,
        )
    )

    ax.axhline(d["backbone"], linestyle="--", linewidth=1.0, color="#222222")
    # Sits just above the dashed line over the palest bar, the only region of
    # the panel with no ink of its own.
    ax.text(
        -0.28,
        d["backbone"] + top * 0.028,
        f"backbone {d['backbone']:.4f}",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#333333",
    )


def _shares(ax: plt.Axes, d: dict[str, float]) -> None:
    ax.set_title("(b) what survives restoration", loc="left")

    shape_pct = d["shape_fraction"] * 100.0
    mag_pct = 100.0 - shape_pct

    ax.barh([0], [shape_pct], height=0.42, color=PURPLE)
    ax.barh([0], [mag_pct], left=[shape_pct], height=0.42, color="#E8722C")

    ax.text(
        shape_pct / 2,
        0.30,
        f"shape\n{shape_pct:.1f}%",
        ha="center",
        va="bottom",
        color=PURPLE,
        fontweight="bold",
        fontsize=7.4,
    )
    ax.text(
        shape_pct + mag_pct / 2,
        0,
        f"unrecovered magnitude {mag_pct:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=7.4,
    )

    ratio = d["backbone"] / d["shape_floor"]
    ax.text(
        0,
        -0.52,
        f"optimally rescaled, the analogues would be {ratio:.1f}\N{MULTIPLICATION SIGN} "
        "more accurate than the backbone",
        ha="left",
        va="center",
        fontsize=6.6,
        color=GREY,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("share of the restored branch's residual MSE (%)")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("docs/error-decomposition-ettm2-test.json"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--stem", default="error-decomposition")
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(
            f"{args.results} not found. Regenerate it with "
            "`uv run python scripts/error_decomposition_run.py` first."
        )

    d = load_decomposition(args.results)
    args.outdir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.3), gridspec_kw={"wspace": 0.30})
        _waterfall(axes[0], d)
        _shares(axes[1], d)
        for ext, dpi in (("pdf", None), ("png", 220)):
            fig.savefig(args.outdir / f"{args.stem}.{ext}", dpi=dpi)
        plt.close(fig)

    print(f"n_windows            {d['n_windows']:,}")
    print(f"raw scale error      {d['scale_error']:.4f}")
    print(f"removed              {d['removed']:.4f}")
    print(f"remaining            {d['remaining']:.4f}")
    print(f"shape floor          {d['shape_floor']:.4f}")
    print(f"backbone             {d['backbone']:.4f}")
    print(f"shape share          {d['shape_fraction'] * 100:.1f}%")
    print(f"wrote                {args.outdir / args.stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
