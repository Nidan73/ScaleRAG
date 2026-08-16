#!/usr/bin/env python3
"""Publication figures for the ScaleRAG-TS native TS-RAG study (Phase 11A).

Every value is sourced from a real artifact — no fabricated numbers:

* ETTm2 dev/test grid + per-window preds  -> ``reports/phase11a/*``
* ETTm2 mechanism CIs + per-series wins    -> ``docs/scalerag-native-dev-results.json``
* compute / latency                        -> ``reports/phase11a/compute_*.json``
* M5 ablation (restoration, top-k)         -> ``docs/ablation-report.md`` (cited constants)
* M5 regime slices + cross-dataset utility -> ``reports/scalerag-heldout-val-30490.json``,
                                              ``reports/scalerag-favorita.json``
* fig1 / fig3 exemplar windows             -> reconstructed with the frozen retriever
                                              (:mod:`scalerag.native`)

Outputs vector PDFs to ``paper/figures/`` in journal-style serif.

Design rules applied (see the figure-design notes in ``docs/project-status.md``):

* **One y-axis per panel.** Two measures of different scale become small multiples,
  never a twin axis — a dual-scale plot invents a correlation the data does not have.
* **Direct end-labels** on lines instead of legend boxes parked over the data; a legend
  is used only where marks cannot be labelled in place.
* Solid hairline gridlines on the value axis only; top/right spines dropped.
* Categorical palette below is validated colour-blind-safe on an all-pairs basis
  (worst deutan ΔE 13.0, worst normal-vision ΔE 16.3, all ≥3:1 on white).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ettm2_data as E  # noqa: N812  (canonical helper; `E` matches sibling scripts)
from scalerag.native import NativeScaleRetriever

ROOT = Path(__file__).resolve().parents[1]
P11 = ROOT / "reports/phase11a"
FIG = ROOT / "paper/figures"

# ---- palette (validated all-pairs CVD-safe; see module docstring) --------------------
C_OURS = "#2a78d6"  # blue    — ScaleRAG / scale-restored (the proposed mechanism)
C_RAW = "#eb6834"  # orange  — raw retrieval / target-only backbone (the comparison)
C_TSRAG = "#4a3aa7"  # violet  — official TS-RAG neural adapter
C_BAND = "#c6dcf7"  # blue tint for the retrieved-candidate envelope

INK = "#0b0b0b"  # primary ink (ground truth, emphasis)
INK_2 = "#52514e"  # secondary ink (labels)
INK_MUTED = "#898781"  # muted ink (context, annotations)
GRID = "#e1e0d9"  # hairline grid
RULE = "#c3c2b7"  # baseline / axis


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.alpha": 1.0,
            "grid.linestyle": "-",  # solid hairline: dashes read as "threshold"
            "grid.linewidth": 0.6,
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed editable TrueType (journal-safe)
            "ps.fonttype": 42,
        }
    )


def _despine(ax: plt.Axes, keep_x: bool = True) -> None:
    """Drop top/right spines and the x gridlines (value axis carries the grid)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=not keep_x)
    ax.grid(axis="y", visible=True)


def _panel(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, pad=8)


def _end_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    dx: float = 4,
    dy: float = 0,
    ha: str = "left",
    va: str = "center",
    weight: str = "normal",
) -> None:
    """Direct-label a line at a chosen point, in ink-with-colour rather than a legend box."""
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va=va,
        fontsize=8.5,
        color=color,
        fontweight=weight,
        zorder=6,
    )


# ---- shared data --------------------------------------------------------------------
def load_ettm2_test() -> dict:
    d = np.load(P11 / "scalerag_native_ettm2_test_preds.npz")
    tsrag = np.load(P11 / "repro_tsrag_official_ettm2.npz")["preds"].astype(np.float64)
    grid = json.loads((P11 / "scalerag_native_ettm2_test.json").read_text())
    dev = json.loads((ROOT / "docs/scalerag-native-dev-results.json").read_text())
    return {
        "trues": d["trues"].astype(np.float64),
        "var_of": d["var_of"],
        "chronos": d["chronos"].astype(np.float64),
        "res_mean_20": d["res_mean_20"].astype(np.float64),
        "tsrag": tsrag,
        "grid": grid,
        "dev": dev,
        "names": grid["variables"],
    }


def _pf(x: object) -> float:
    """Float a polars scalar (its stub return type is a broad union)."""
    return float(x)  # type: ignore[arg-type]


def build_retriever(scale: str, var: int) -> NativeScaleRetriever:
    z, _ = E.load_normalized()
    return NativeScaleRetriever(z[:, var], E.TRAIN_END, scale, E.L, E.H)


# ---- fig1: motivation ---------------------------------------------------------------
def fig1_motivation() -> None:
    """Query vs a raw-Euclidean candidate (scale mismatch) vs the restored candidate.

    Across the smoother ETTm2 channels, pick the real (query, top-1 candidate) pair whose
    candidate continuation best matches the query's future *shape* while differing most in
    *scale* — the exact case scale restoration is designed for. Selection criteria are
    unchanged from the original figure; only the rendering differs.
    """
    z, names = E.load_normalized()
    scale = "rms"
    w = E.build_windows(z, "test")
    best: tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = (
        None
    )
    for var in (6, 2, 0, 1):  # OT, MUFL, HUFL, HULL (smoother channels)
        retr = build_retriever(scale, var)
        m = w["var_of"] == var
        ctx, tru, org = w["contexts"][m], w["trues"][m], w["origins"][m]
        idx = np.arange(0, ctx.shape[0], 37)
        tk = retr.retrieve(ctx[idx], org[idx], 1)
        cid = tk.ids[:, 0]
        cont = retr.db.continuations[cid]  # (S, H) raw candidate continuations
        cp, qp = retr.params[cid], tk.qparams  # (S, 2)
        ratio = qp[:, 1] / cp[:, 1]
        a = cont - cont.mean(1, keepdims=True)
        b = tru[idx] - tru[idx].mean(1, keepdims=True)
        corr = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        logr = np.abs(np.log2(np.clip(ratio, 1e-3, None)))
        fvar = tru[idx].var(1)
        elig = (logr > 0.7) & (fvar > np.median(fvar)) & np.isfinite(corr)
        score = np.where(elig, corr, -np.inf)
        j = int(np.argmax(score))
        if np.isfinite(score[j]) and (best is None or score[j] > best[0]):
            best = (float(score[j]), var, ctx[idx[j]], tru[idx[j]], cont[j], cp[j], qp[j])
    assert best is not None, "no eligible shape-match/scale-gap window found"
    _, var, q_ctx, q_true, cand_cont, cp1, qp1 = best
    restored = (cand_cont - cp1[0]) / cp1[1] * qp1[1] + qp1[0]

    tail = 96
    hx = np.arange(-tail, 0)
    fx = np.arange(0, E.H)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)

    for a_, cand, col, lab in (
        (ax[0], cand_cont, C_RAW, "raw retrieved\ncandidate"),
        (ax[1], restored, C_OURS, "scale-restored\ncandidate"),
    ):
        a_.axvspan(-tail, 0, color=GRID, alpha=0.45, lw=0, zorder=0)
        a_.plot(hx, q_ctx[-tail:], color=INK_MUTED, lw=1.0, zorder=2)
        a_.plot(fx, q_true, color=INK, lw=1.9, zorder=4)
        a_.plot(fx, cand, color=col, lw=1.7, zorder=3)
        # shade the residual gap the mechanism is meant to close
        a_.fill_between(fx, q_true, cand, color=col, alpha=0.13, lw=0, zorder=1)
        a_.axvline(0, color=RULE, lw=0.8, zorder=1)
        a_.set_xlabel("step relative to forecast origin")
        _despine(a_)
        _end_label(a_, fx[-1], q_true[-1], "query future\n(ground truth)", INK, dx=5)
        _end_label(a_, fx[-1], cand[-1], lab, col, dx=5)
        a_.set_xlim(-tail, E.H + 34)

    ax[0].set_ylabel("normalized value")
    rms_ratio = cp1[1] / qp1[1]
    _panel(ax[0], f"(a) Raw Euclidean match — off by rms $\\times${rms_ratio:.2f}")
    _panel(ax[1], "(b) After explicit scale restoration")
    ax[0].annotate(
        "context window\n(used for matching)",
        xy=(-tail / 2, ax[0].get_ylim()[0]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK_MUTED,
    )
    fig.text(
        0.5,
        -0.06,
        f"ETTm2 · channel {names[var]} · a single real (query, top-1 candidate) pair",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.savefig(FIG / "fig1_motivation.pdf")
    plt.close(fig)


# ---- fig2: ablation -----------------------------------------------------------------
# M5 ablation constants — source: docs/ablation-report.md (1,000-series val)
M5_NO_RESTORE = 2.7884
M5_RESTORE = 0.7425


def _dumbbell(
    ax: plt.Axes, ys: np.ndarray, raw: np.ndarray, res: np.ndarray, label_first: bool
) -> None:
    """One row per category: orange dot (raw) -> blue dot (restored), joined by a rule."""
    ax.hlines(ys, raw, res, color=RULE, lw=1.6, zorder=1)
    ax.scatter(raw, ys, s=46, color=C_RAW, zorder=3, edgecolor="white", lw=1.0)
    ax.scatter(res, ys, s=46, color=C_OURS, zorder=3, edgecolor="white", lw=1.0)
    if label_first:
        # anchor the two identity labels on the widest row so they cannot collide
        w = int(np.argmax(np.abs(np.log10(raw) - np.log10(res))))
        _end_label(ax, raw[w], ys[w], "raw", C_RAW, dx=0, dy=9, ha="center", va="bottom")
        _end_label(ax, res[w], ys[w], "restored", C_OURS, dx=0, dy=9, ha="center", va="bottom")


def fig2_ablation() -> None:
    d = load_ettm2_test()
    per = d["dev"]["test_per_series"]["A_restoration"]["per_series"]
    labels = [r["var"] for r in per]
    raw = np.array([r["baseline_mse"] for r in per])
    res = np.array([r["method_mse"] for r in per])

    order = np.argsort(-raw)  # worst raw at the top
    labels = [labels[i] for i in order]
    raw, res = raw[order], res[order]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.45}
    )

    ys = np.arange(len(labels))[::-1]
    _dumbbell(ax_l, ys, raw, res, label_first=True)
    ax_l.set_xscale("log")
    ax_l.set_yticks(ys)
    ax_l.set_yticklabels(labels)
    ax_l.set_xlabel("retrieval MSE (log scale, lower better)")
    ax_l.set_ylim(-0.9, len(labels) - 0.1)
    _panel(ax_l, "(a) ETTm2 test — all 7/7 channels improve")
    ax_l.grid(axis="x", visible=True)
    ax_l.grid(axis="y", visible=False)

    ym = np.array([0.0])
    _dumbbell(ax_r, ym, np.array([M5_NO_RESTORE]), np.array([M5_RESTORE]), label_first=False)
    ax_r.set_yticks(ym)
    ax_r.set_yticklabels(["M5 (1k, val)"])
    ax_r.set_xlabel("RMSSE (lower better)")
    ax_r.set_xlim(0, M5_NO_RESTORE * 1.28)
    ax_r.set_ylim(-0.9, 0.9)
    _panel(ax_r, "(b) M5 — 3.8$\\times$ lower")
    ax_r.grid(axis="x", visible=True)
    ax_r.grid(axis="y", visible=False)
    _end_label(ax_r, M5_NO_RESTORE, 0.0, f"{M5_NO_RESTORE:.3f}", C_RAW, dx=0, dy=10, ha="center")
    _end_label(ax_r, M5_RESTORE, 0.0, f"{M5_RESTORE:.3f}", C_OURS, dx=0, dy=-16, ha="center")

    fig.savefig(FIG / "fig2_ablation.pdf")
    plt.close(fig)


# ---- fig3: qualitative --------------------------------------------------------------
def fig3_qualitative() -> None:
    d = load_ettm2_test()
    z, names = E.load_normalized()
    scale, var, k = "mean", 6, 20  # OT: mid-scale, well-behaved channel
    retr = build_retriever(scale, var)
    w = E.build_windows(z, "test")
    m = w["var_of"] == var
    ctx, tru, org = w["contexts"][m], w["trues"][m], w["origins"][m]
    tot = int(w["tot"])
    # row offset of this channel in the canonical bundle (var outer, window inner)
    base_row = var * tot
    # Pick a *representative* window rather than a flattering one: among windows with a
    # non-trivial future (above-median variance, so the plot shows structure), take the one
    # whose fused ScaleRAG error is closest to that subset's median. Typical, not best-case.
    fut_var = tru.var(axis=1)
    cand_rows = np.flatnonzero(fut_var > np.median(fut_var))
    fused_all = (
        0.75 * d["chronos"][base_row + cand_rows] + 0.25 * d["res_mean_20"][base_row + cand_rows]
    )
    err = ((fused_all - tru[cand_rows]) ** 2).mean(axis=1)
    s = int(cand_rows[np.argmin(np.abs(err - np.median(err)))])
    q_ctx, q_true, q_org = ctx[s], tru[s], int(org[s])
    row = base_row + s

    tk = retr.retrieve(q_ctx[None, :], np.array([q_org]), k)
    ids = tk.ids[0]
    conts = retr.db.continuations[ids]
    cp, qp = retr.params[ids], tk.qparams[0]
    # all k restored candidates -> an envelope, rather than 3 unreadable spaghetti lines
    restored_k = (conts - cp[:, 0:1]) / cp[:, 1:2] * qp[1] + qp[0]
    lo, hi = restored_k.min(axis=0), restored_k.max(axis=0)
    res_mean = d["res_mean_20"][row]  # locked artifact: mean of the k restored candidates
    chronos = d["chronos"][row]
    fused = 0.75 * chronos + 0.25 * res_mean  # frozen weight 0.25
    tsrag = d["tsrag"][row]

    fx = np.arange(0, E.H)
    tail = 96
    hx = np.arange(-tail, 0)
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True)

    for a_ in (ax_t, ax_b):
        a_.axvspan(-tail, 0, color=GRID, alpha=0.45, lw=0, zorder=0)
        a_.plot(hx, q_ctx[-tail:], color=INK_MUTED, lw=1.0, zorder=2)
        a_.axvline(0, color=RULE, lw=0.8, zorder=1)
        a_.set_ylabel("normalized value")
        _despine(a_)
        a_.set_xlim(-tail, E.H + 30)

    ax_t.fill_between(fx, lo, hi, color=C_BAND, lw=0, zorder=1)
    ax_t.plot(fx, res_mean, color=C_OURS, lw=1.7, zorder=3)
    ax_t.plot(fx, q_true, color=INK, lw=1.9, zorder=4)
    _end_label(ax_t, fx[-1], q_true[-1], "ground truth", INK, dx=5)
    _end_label(ax_t, fx[-1], res_mean[-1], "mean of $k$ restored", C_OURS, dx=5)
    _end_label(ax_t, fx[len(fx) // 2], hi[len(fx) // 2], f"spread of all $k={k}$", C_OURS, dy=6)
    _panel(ax_t, f"(a) Scale-restored retrieval envelope (ETTm2 · {names[var]}, $k={k}$)")

    # the four forecasts converge by the horizon end, so end-labels would pile up:
    # use a compact legend parked in the empty upper-left instead
    ax_b.plot(
        fx, chronos, color=C_RAW, lw=1.5, ls="--", zorder=3, label="Chronos-Bolt (target-only)"
    )
    ax_b.plot(fx, tsrag, color=C_TSRAG, lw=1.5, ls=":", zorder=3, label="TS-RAG (official)")
    ax_b.plot(fx, fused, color=C_OURS, lw=1.7, zorder=3, label="ScaleRAG (restored fusion)")
    ax_b.plot(fx, q_true, color=INK, lw=1.9, zorder=4, label="ground truth")
    handles, labels_b = ax_b.get_legend_handles_labels()
    ordering = [3, 0, 1, 2]  # ground truth first
    ax_b.legend(
        [handles[i] for i in ordering],
        [labels_b[i] for i in ordering],
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.2,
    )
    # headroom so the legend block clears every curve instead of grazing the fusion line
    lo_b, hi_b = ax_b.get_ylim()
    ax_b.set_ylim(lo_b, hi_b + 0.42 * (hi_b - lo_b))
    ax_b.set_xlabel("step relative to forecast origin")
    _panel(ax_b, "(b) Fused forecasts against ground truth")

    fig.savefig(FIG / "fig3_qualitative.pdf")
    plt.close(fig)


# ---- fig4: regimes ------------------------------------------------------------------
def _zero_fraction(parquet: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(parquet)
        .select("id", "sales")
        .group_by("id")
        .agg((pl.col("sales") <= 0).mean().alias("zf"))
        .collect()
    )


def fig4_regimes() -> None:
    m5 = _zero_fraction(ROOT / "data/processed/dynamic.parquet")
    fav = _zero_fraction(ROOT / "data/processed/favorita/dynamic.parquet")
    m5_zf = m5["zf"]
    ettm2_zf = 0.0  # continuous channels, no zeros
    dev = json.loads((ROOT / "docs/scalerag-native-dev-results.json").read_text())
    ettm2_util = dev["test_mechanisms"]["C_vs_target"]["rel_mse_improvement"]

    # (x=mean zero-fraction, y=utility over target-only TSFM, label, is_slice, label offset)
    # utility sources: reports/scalerag-heldout-val-30490.json (M5 overall+slices),
    #                  reports/scalerag-favorita.json (Favorita), Phase-11A dev (ETTm2)
    pts = [
        (
            _pf(m5_zf.filter(m5_zf > 0.8).mean()),
            0.05630,
            "M5 intermittent ($z>0.8$)",
            True,
            (-8, 4),
        ),
        (_pf(m5_zf.mean()), 0.05076, "M5 overall", False, (-8, 6)),
        (_pf(m5_zf.filter(m5_zf < 0.3).mean()), 0.00130, "M5 dense ($z<0.3$)", True, (8, 6)),
        (_pf(fav["zf"].mean()), 0.008306, "Favorita", False, (8, 4)),
        (ettm2_zf, ettm2_util, "ETTm2", False, (8, -2)),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))

    ax.axhspan(-1.5, 0, color=C_RAW, alpha=0.06, lw=0, zorder=0)
    ax.axhline(0, color=RULE, lw=1.0, zorder=2)
    # NOTE: deliberately no trend line -- these are five different populations on two
    # different metrics; a connecting curve would imply a fit that was never estimated.
    for x, y, lab, is_slice, off in pts:
        ax.scatter(
            x,
            y * 100,
            s=64,
            facecolor="white" if is_slice else C_OURS,
            edgecolor=C_OURS,
            lw=1.6,
            zorder=4,
        )
        ax.annotate(
            lab,
            (x, y * 100),
            textcoords="offset points",
            xytext=off,
            ha="right" if off[0] < 0 else "left",
            fontsize=8.5,
            color=INK_2,
            zorder=5,
        )
    ax.set_xlabel("intermittency  (mean fraction of zero observations)")
    ax.set_ylabel("retrieval utility over\ntarget-only TSFM  (%)")
    ax.set_xlim(-0.06, 1.0)
    ax.set_ylim(-1.5, 6.6)
    _despine(ax)
    _end_label(ax, 0.97, -0.75, "retrieval hurts", C_RAW, dx=0, ha="right")
    ax.scatter([], [], s=64, facecolor=C_OURS, edgecolor=C_OURS, lw=1.6, label="whole dataset")
    ax.scatter([], [], s=64, facecolor="white", edgecolor=C_OURS, lw=1.6, label="M5 sub-slice")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, handletextpad=0.4)
    fig.text(
        0.5,
        -0.10,
        "utility measured in RMSSE for M5 / Favorita (counts) and MSE for ETTm2 (continuous)",
        ha="center",
        fontsize=8,
        color=INK_MUTED,
    )
    fig.savefig(FIG / "fig4_regimes.pdf")
    plt.close(fig)


# ---- fig5: pareto -------------------------------------------------------------------
def fig5_pareto() -> None:
    cb = json.loads((P11 / "compute_backbone.json").read_text())
    sc = json.loads((P11 / "compute_scalerag.json").read_text())
    grid = json.loads((P11 / "scalerag_native_ettm2_test.json").read_text())
    back_ms = cb["backbone_latency_ms_per_window"]
    retr_ms = sc["retrieval_latency_ms_per_window"]
    # wall-clock ratios from the reproduction (backbone 34.06s, TS-RAG 35.53s)
    tsrag_ms = back_ms * (35.53 / 34.06)
    chronos_mse = grid["chronos_bolt_target"]["mse"]
    tsrag_mse = json.loads((P11 / "repro_tsrag_official_ettm2.json").read_text())["mse"]
    fused_mse = next(
        r["mse"]
        for r in grid["scalerag_restored_fixed_fusion"]
        if r["scale"] == "mean" and r["k"] == 20 and r["weight"] == 0.25
    )
    ours = (back_ms + retr_ms, fused_mse)
    pts = [
        (back_ms, chronos_mse, "Chronos-Bolt\n(target-only)", C_RAW, (0, -20), "center"),
        (tsrag_ms, tsrag_mse, "TS-RAG\n(official ARM)", C_TSRAG, (10, 0), "left"),
        (ours[0], ours[1], "ScaleRAG\n(restored fusion)", C_OURS, (-10, 0), "right"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))

    # everything up-and-right of TS-RAG is both slower and less accurate than TS-RAG
    ax.axhspan(tsrag_mse, 1, xmin=0, xmax=1, color=RULE, alpha=0.0, lw=0)
    ax.add_patch(
        plt.Rectangle(
            (tsrag_ms, tsrag_mse),
            10,
            10,
            color=C_RAW,
            alpha=0.07,
            lw=0,
            zorder=0,
        )
    )
    for x, y, lab, col, off, ha in pts:
        ax.scatter(x, y, s=78, color=col, edgecolor="white", lw=1.2, zorder=4)
        ax.annotate(
            lab,
            (x, y),
            textcoords="offset points",
            xytext=off,
            ha=ha,
            va="center" if off[1] == 0 else "top",
            fontsize=8.5,
            color=col,
            zorder=5,
        )
    ax.set_xlabel("inference latency  (ms / window, lower better)")
    ax.set_ylabel("ETTm2 test MSE  (lower better)")
    ax.set_xlim(back_ms - 0.13, ours[0] + 0.13)
    ax.set_ylim(tsrag_mse - 0.0006, ours[1] + 0.0006)
    _despine(ax)
    ax.annotate(
        "dominated:\nslower and less accurate\nthan TS-RAG",
        xy=(ours[0] - 0.02, tsrag_mse + 0.00045),
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_MUTED,
        zorder=5,
    )
    ax.annotate(
        "better",
        xy=(0.035, 0.06),
        xytext=(0.16, 0.06),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=8,
        color=INK_MUTED,
        arrowprops={"arrowstyle": "->", "color": INK_MUTED, "lw": 0.9},
    )
    fig.savefig(FIG / "fig5_pareto.pdf")
    plt.close(fig)


# ---- fig6: sensitivity --------------------------------------------------------------
# M5 restored-retrieval RMSSE top-k sweep — source: docs/ablation-report.md
M5_TOPK = {5: 0.7708, 10: 0.7511, 20: 0.7425}


def fig6_sensitivity() -> None:
    grid = json.loads((P11 / "scalerag_native_ettm2_test.json").read_text())
    ks = [5, 10, 20]
    restored = {
        r["k"]: r["mse"] for r in grid["scalerag_restored_retrieval"] if r["scale"] == "mean"
    }
    fusion = {
        r["k"]: r["mse"]
        for r in grid["scalerag_restored_fixed_fusion"]
        if r["scale"] == "mean" and r["weight"] == 0.25
    }
    chronos = grid["chronos_bolt_target"]["mse"]

    # Three panels, one y-axis each. The three quantities differ by an order of
    # magnitude and two of them are different metrics entirely, so they are shown as
    # small multiples -- never as twin axes on one plot.
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax_a, ax_b, ax_c = axes

    ya = [restored[k] for k in ks]
    ax_a.plot(ks, ya, "-o", color=C_OURS, lw=1.7, ms=5, mec="white", mew=1.0)
    ax_a.set_ylabel("ETTm2 retrieval MSE")
    _panel(ax_a, "(a) Retrieval quality")
    _end_label(ax_a, ks[-1], ya[-1], f"{ya[-1]:.3f}", C_OURS, dx=-8, dy=0, ha="right")

    yb = [fusion[k] for k in ks]
    ax_b.plot(ks, yb, "-o", color=C_OURS, lw=1.7, ms=5, mec="white", mew=1.0)
    ax_b.axhline(chronos, color=C_RAW, lw=1.4, ls="--", zorder=2)
    ax_b.set_ylabel("ETTm2 fused MSE")
    _panel(ax_b, "(b) End-to-end fusion")
    _end_label(ax_b, ks[0], chronos, "target-only backbone", C_RAW, dx=0, dy=-12, ha="left")
    _end_label(ax_b, ks[0], yb[0], "ScaleRAG", C_OURS, dx=5, dy=8, ha="left")
    ax_b.set_ylim(chronos - 0.0018, max(yb) + 0.0022)

    yc = [M5_TOPK[k] for k in ks]
    ax_c.plot(ks, yc, "-o", color=C_OURS, lw=1.7, ms=5, mec="white", mew=1.0)
    ax_c.set_ylabel("M5 retrieval RMSSE")
    _panel(ax_c, "(c) M5 retrieval quality")
    _end_label(ax_c, ks[-1], yc[-1], f"{yc[-1]:.4f}", C_OURS, dx=-8, dy=0, ha="right")

    for a_ in axes:
        a_.set_xticks(ks)
        a_.set_xlabel("retrieved sequences $k$")
        _despine(a_)
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(FIG / "fig6_sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_motivation()
    fig2_ablation()
    fig3_qualitative()
    fig4_regimes()
    fig5_pareto()
    fig6_sensitivity()
    print(f"wrote 6 figures to {FIG}")
    for f in sorted(FIG.glob("fig*.pdf")):
        print("  ", f.name, f"{f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
