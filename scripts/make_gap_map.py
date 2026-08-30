#!/usr/bin/env python3
"""Generate the gap-map schematic (Fig. 2) as a drawio file.

`figures/aaa.drawio.pdf` printed a stale 85.2/14.8 split and could not be
patched: the digits appearing anywhere in that export are 0-5 and 8, so the
embedded font subset has no `9` glyph. It had no source either, which is why it
drifted in the first place. This is that source.

Rows are data, so correcting a number is a one-line edit here and never a
hand-edit of markup. The palette is read off the original export rather than
invented, so the figure stays consistent with the four other drawio schematics.

Writes both:
  docs/figure-sources/gap-map.drawio      -- open directly in drawio
  docs/figure-sources/gap-map.drawio.xml  -- paste into Extras > Edit Diagram

Usage (fish):
    uv run python scripts/make_gap_map.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figure-sources"

# Read off aaa.drawio.pdf's own content streams; not a new palette.
GAP_LINE, GAP_FILL, GAP_TEXT = "#A6303B", "#F6E4E6", "#3A2A2C"
CON_LINE, CON_FILL, CON_TEXT = "#1F5A73", "#E4EEF2", "#22323A"
NEUTRAL = "#555555"
FONT = "Liberation Sans"

# (gap, contribution). Wording mirrors sec:gaps and sec:contributions.
ROWS: list[tuple[str, str]] = [
    (
        "Retrieval utility is reported as a dataset-level average",
        "Per-series regime profile over 50 origins, binned by context zero fraction: an inverted U",
    ),
    (
        "Magnitude is handled implicitly, inside learned capacity",
        "Closed-form scale restoration; the scale tag is measured, not fitted",
    ),
    (
        "The error that survives retrieval is never attributed",
        "Oracle affine decomposition of the residual: 85.1% unrecovered magnitude, 14.9% shape",
    ),
    (
        "Invariance is asserted, never measured under a controlled shift",
        "Synthetic probe separating invariance from equivariance, under a known affine transform",
    ),
    (
        "Evaluation is reported on the favourable metric, without pre-registration",
        "Single-open held-out panel, criteria registered in advance: 0 of 3 met, and reported",
    ),
]

X_GAP, W_GAP = 40, 380
X_CON, W_CON = 496, 564
Y_TOP, H_ROW, PITCH = 78, 72, 84


def _box(colour: str, fill: str, text: str) -> str:
    return (
        f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={colour};"
        f"strokeWidth=1;fontFamily={FONT};fontSize=13;fontColor={text};align=left;"
        "verticalAlign=middle;spacingLeft=12;spacingRight=10;"
    )


ARROW = (
    "edgeStyle=none;html=1;endArrow=block;endFill=1;endSize=5;"
    f"strokeColor={NEUTRAL};strokeWidth=1;"
    "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
)


def _label(tag: str, colour: str, body: str) -> str:
    """Tag in the accent colour, prose in near-black.

    Single quotes inside the HTML deliberately: no ``&quot;`` survives into the
    attribute, which is one less thing for a clipboard to mangle. The colour
    carries the tag only, so the boxes stay readable in grayscale print.
    """
    spacer = "&nbsp;" * 3
    return f"<b><font color='{colour}'>{tag}</font></b>{spacer}{body}"


def _cell(root: ET.Element, **attrs: str) -> ET.Element:
    return ET.SubElement(root, "mxCell", {k: str(v) for k, v in attrs.items()})


def _geom(cell: ET.Element, **attrs: str) -> None:
    g = ET.SubElement(cell, "mxGeometry", {k: str(v) for k, v in attrs.items()})
    g.set("as", "geometry")


def build() -> ET.Element:
    model = ET.Element(
        "mxGraphModel",
        {
            "dx": "1422",
            "dy": "762",
            "grid": "0",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1100",
            "pageHeight": "540",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    _cell(root, id="0")
    _cell(root, id="1", parent="0")

    hdr = (
        f"text;html=1;strokeColor=none;fillColor=none;verticalAlign=middle;"
        f"fontFamily={FONT};fontSize=14;fontStyle=1;spacingLeft=2;align=left;"
    )
    h1 = _cell(
        root,
        id="hdrL",
        value="critical gap in prior work",
        style=hdr + f"fontColor={GAP_LINE};",
        vertex="1",
        parent="1",
    )
    _geom(h1, x=X_GAP, y=24, width=W_GAP, height=26)
    h2 = _cell(
        root,
        id="hdrM",
        value="addressed by",
        style=(
            f"text;html=1;strokeColor=none;fillColor=none;align=center;"
            f"verticalAlign=middle;fontFamily={FONT};fontSize=11;"
            f"fontStyle=2;fontColor={NEUTRAL};"
        ),
        vertex="1",
        parent="1",
    )
    _geom(h2, x=420, y=24, width=76, height=26)
    h3 = _cell(
        root,
        id="hdrR",
        value="contribution of this work",
        style=hdr + f"fontColor={CON_LINE};",
        vertex="1",
        parent="1",
    )
    _geom(h3, x=X_CON, y=24, width=W_CON, height=26)

    rule = _cell(
        root,
        id="rule",
        value="",
        style=f"endArrow=none;html=1;strokeColor={NEUTRAL};strokeWidth=1;",
        edge="1",
        parent="1",
    )
    rg = ET.SubElement(rule, "mxGeometry", {"relative": "1"})
    rg.set("as", "geometry")
    for name, x in (("sourcePoint", X_GAP), ("targetPoint", X_CON + W_CON)):
        pt = ET.SubElement(rg, "mxPoint", {"x": str(x), "y": "58"})
        pt.set("as", name)

    for i, (gap, con) in enumerate(ROWS):
        y = Y_TOP + PITCH * i
        n = i + 1
        g = _cell(
            root,
            id=f"g{n}",
            value=_label(f"G{n}", GAP_LINE, gap),
            style=_box(GAP_LINE, GAP_FILL, GAP_TEXT),
            vertex="1",
            parent="1",
        )
        _geom(g, x=X_GAP, y=y, width=W_GAP, height=H_ROW)
        c = _cell(
            root,
            id=f"c{n}",
            value=_label(f"C{n}", CON_LINE, con),
            style=_box(CON_LINE, CON_FILL, CON_TEXT),
            vertex="1",
            parent="1",
        )
        _geom(c, x=X_CON, y=y, width=W_CON, height=H_ROW)
        a = _cell(
            root, id=f"a{n}", style=ARROW, edge="1", parent="1", source=f"g{n}", target=f"c{n}"
        )
        ag = ET.SubElement(a, "mxGeometry", {"relative": "1"})
        ag.set("as", "geometry")
    return model


def main() -> int:
    model = build()
    ET.indent(model, space="  ")
    body = ET.tostring(model, encoding="unicode")

    # Guard the two things that have actually gone wrong with this figure.
    if "85.2" in body or "14.8" in body:
        raise ValueError("the stale decomposition split is back in the figure")
    if "&quot;" in body:
        raise ValueError("a double-quote entity survived into an attribute")
    ET.fromstring(body)  # must parse exactly as drawio's parser will see it

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gap-map.drawio.xml").write_text(body + "\n")

    # A standalone .drawio file opens directly, with no clipboard in the path.
    dia = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
    page = ET.SubElement(dia, "diagram", {"name": "gap-map", "id": "gapmap"})
    page.append(model)
    ET.indent(dia, space="  ")
    (OUT / "gap-map.drawio").write_text(ET.tostring(dia, encoding="unicode") + "\n")

    n = len(body.split("\n"))
    print(f"wrote {OUT.relative_to(REPO)}/gap-map.drawio      (open directly)")
    print(f"wrote {OUT.relative_to(REPO)}/gap-map.drawio.xml  ({n} lines, paste target)")
    widest = max(len(row) for row in body.split("\n"))
    print(f"rows: {len(ROWS)}   longest line: {widest} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
