"""figure_split.py — save every panel of a composite figure as its own file.

Which panels go into the paper, and how they recombine, is not settled and will
change more than once.  Splitting up front makes that a LaTeX-side decision
(`subfigure`/`minipage`) instead of a "regenerate the script" decision.

    python3 figure_split.py figure_antisym_explain.py compare_speed_dp_hncb.py
    python3 figure_split.py --all              # every *.py that calls savefig
    python3 figure_split.py --list             # show what --all would run

Output, one folder per composite figure:

    numerical_notes/figures_subfigures_sep/<stem>/
        <stem>_full.png     the composite, exactly as the script wrote it
        <stem>_full.pdf     vector copy when the source was vector
        <stem>_a.pdf/.png   panel (a) — reading order, row-major
        <stem>_b.pdf/.png   ...
        PANELS.md           what each panel is, so it can be picked without opening it

⚠ By default the scripts' own outputs in `numerical_notes/results/` are NOT
overwritten — none of them is tracked by git, so a regenerated composite that
came out different could not be recovered.  Pass `--clobber` for a real refresh.

⚠⚠ **The savefig guard only covers figures.**  Scripts that write CSV / NPZ / MD /
JSON do so through ordinary file I/O, which nothing here intercepts.  Measured on
the 2026-08-10 sweep: 0 of 2729 csv/npz changed, but **5 markdown result files were
silently rewritten** —

    results/compare_energy_compress/route_gap_tables.md
    results/compare_energy_compress/critical_point/critical_tables.md
    results/compare_energy_compress/critical_point/critical_line_s2lj_tables.md
    results/fisher_widom/fw_vs_chi_id.md
    results/hsy_incore_opt.md          <- resurrected a path git had staged as MOVED

and none of them is tracked by git, so none could be diffed or restored.

The trap is not "the file was overwritten", it is **overwritten with the script's
DEFAULT parameters**.  `optimize_hsy_incore.py` rewrote its table at rho*=0.139
while the copy another session had just produced was rho*=0.19658 — same filename,
contradictory numbers.  A rerun is not a no-op unless the last real run also used
the defaults, and there is no way to know that from the file.

So before `--all`:  snapshot every csv/npz/md/json under `numerical_notes/`, not
just the ones `frozen_manifest.md` pins, and diff afterwards.  Worst offenders:
`compare_energy_compress.py`, `fisher_widom_line.py`, `phase_equilibrium*.py`,
`critical_line_s2lj.py`, `gcmc_dcf_figures.py`, `optimize_hsy_incore.py`,
`oz_closure_comparison.py`.

⚠ Other sessions may be running these same scripts concurrently — the FWL project
wrote into `results/fisher_widom/` twice during that sweep.  Check mtimes before
attributing a change to your own run.

⚠ NOT a rewrite of the 38 figure scripts.  `Figure.savefig` is wrapped, the
script is run untouched via `runpy`, and every save it performs is split as a
side effect.  A script whose panels are wrong stays wrong — this changes
packaging, not content.

⚠ The composite's `suptitle` is deliberately NOT carried onto the panels: it is
prose, and prose belongs in `\\caption{}`.  Panel titles are kept, since they are
often the only label distinguishing (a) from (b); drop them at use site if the
caption says the same thing.

Panels are cropped by each axes' tight bbox, which includes its title, labels and
any legend parented to it.  A legend attached to the *figure* rather than an axes
lands in no panel — `PANELS.md` flags that case rather than silently losing it.
"""

from __future__ import annotations

import argparse
import os
import re
import runpy
import shlex
import shutil
import string
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

OUT_ROOT = "numerical_notes/figures_subfigures_sep"
PUB_ROOT = "numerical_notes/figures_pub"

# 2-up in a revtex4-2 figure*: \textwidth is 510 pt = 7.06 in across both columns,
# two panels at 0.49\textwidth each.  Panels are generated AT this width so the
# scale factor on the page is 1 and matplotlib pt equals printed pt -- a PDF cannot
# be rescaled after the fact without dragging its fonts along.
# 7.06 in of text width, divided by how many panels sit in a row.  2-up gives
# 3.46 in (0.49\textwidth each); 3-up gives 2.31.  A panel MUST be generated at
# the width it is printed at -- \includegraphics scaling drags the fonts with
# it, so three 3.46 in panels squeezed into 7.06 in print 9 pt text at 6 pt.
PUB_TEXT_IN  = 7.06
PUB_COLS     = 2
PUB_PANEL_IN = 3.46
PUB_FONT_PT  = 9.0
PAD = 1.03                      # bbox expansion so strokes are not shaved
CLOBBER = False                 # see --clobber
PUB = False                     # see --pub
_orig_savefig = matplotlib.figure.Figure.savefig
_report: list[tuple[str, int, str]] = []


def _panel_letters(n: int) -> list[str]:
    """a, b, ... z, aa, ab — enough for the 24-axes figures in this tree."""
    letters = string.ascii_lowercase
    if n <= len(letters):
        return list(letters[:n])
    out = list(letters)
    i = 0
    while len(out) < n:
        out.append(letters[i // len(letters)] + letters[i % len(letters)])
        i += 1
    return out[:n]


def _reading_order(fig):
    """Axes in reading order: top row first, left to right.

    Sorting on the *drawn* position, not creation order, because several scripts
    here build axes out of order (twinx, add_subplot after a loop) and creation
    order would mislabel the panels.
    """
    axes = [a for a in fig.get_axes() if a.get_visible()]
    # tolerance so a row whose axes differ by a pixel is still one row
    def key(a):
        p = a.get_position()
        return (-round(p.y1, 2), round(p.x0, 2))
    return sorted(axes, key=key)


def _clean_title(t: str) -> str:
    """Drop a leading panel label from the title text.

    Most scripts here start the title with its own marker -- "A.  both cover the
    core", "(b) ...", "3) ..." -- which duplicates the (a)/(b)/(c) the caption
    already supplies.  Strip it so the caption reads "(a) both cover the core"
    rather than "(a) A. both cover the core".
    """
    t = (t or "").replace("\n", " ").strip()
    t = re.sub(r"^\(?[A-Za-z0-9]{1,2}[).:]\s+", "", t)
    return t.strip()


def _rescale_for_print(fig, n_panels: int) -> None:
    r"""Shrink the whole figure so one panel lands at PUB_PANEL_IN, and force every
    text object to PUB_FONT_PT.

    Both are needed and they are independent.  Resizing alone leaves the fonts at
    their absolute point size, so a figure built 23 in wide at 18 pt becomes
    unreadable dense when squeezed to 7 in; setting the font alone leaves the panel
    the wrong physical width, and \includegraphics would then rescale it -- fonts
    and all -- which is the thing this mode exists to prevent.

    Measured across the 172 panels, source widths run 0.68--8.34 in.  Placing those
    at a common \includegraphics width is an 11-fold spread in apparent text size
    between panels sitting side by side; this is what removes it.
    """
    import matplotlib.text
    w, h = fig.get_size_inches()
    k = (PUB_PANEL_IN * n_panels) / w
    fig.set_size_inches(w * k, h * k)
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontsize(PUB_FONT_PT)
    for ln in fig.findobj(matplotlib.lines.Line2D):
        ln.set_linewidth(ln.get_linewidth() * k)
        ln.set_markersize(ln.get_markersize() * k)

    # Re-space AFTER the resize.  The script's own tight_layout ran at the original
    # size and is now stale: at ~1/2 the width with fonts held at 9 pt the labels
    # are relatively much larger, they grow into the neighbouring axes, and each
    # panel's tight bbox then captures a slice of the panel beside it.
    try:
        fig.tight_layout(pad=0.3)
    except Exception:
        pass                                   # manual/constrained layouts: leave alone


def _has_raster(ax) -> bool:
    """True if the axes draws something the PDF backend must embed as a bitmap."""
    from matplotlib.image import AxesImage
    from matplotlib.collections import QuadMesh
    if any(isinstance(c, (AxesImage, QuadMesh)) for c in ax.get_children()):
        return True
    return any(getattr(c, "get_rasterized", lambda: False)() for c in ax.get_children())


def _describe(ax) -> str:
    t = (ax.get_title() or "").replace("\n", " ").strip()
    xl = (ax.get_xlabel() or "").strip()
    yl = (ax.get_ylabel() or "").strip()
    bits = []
    if t:
        bits.append(f"**{t}**")
    if yl or xl:
        bits.append(f"`{yl}` vs `{xl}`")
    leg = ax.get_legend()
    if leg is not None:
        labs = [tx.get_text() for tx in leg.get_texts()]
        if labs:
            bits.append("legend: " + ", ".join(f"`{l}`" for l in labs[:6])
                        + (" …" if len(labs) > 6 else ""))
    return " — ".join(bits) if bits else "*(no title or axis labels)*"


def _split(fig, fname: str) -> None:
    stem = os.path.splitext(os.path.basename(fname))[0]
    ext = os.path.splitext(fname)[1].lower() or ".png"
    outdir = os.path.join(OUT_ROOT, stem)
    os.makedirs(outdir, exist_ok=True)

    # the composite, in BOTH the source format and PDF.  Every script here writes
    # PNG, but the paper wants vector, and the PDF comes free from the same live
    # figure object — re-rendering later would need the script to still run.
    full = os.path.join(outdir, f"{stem}_full{ext}")
    if CLOBBER and os.path.exists(fname):
        shutil.copyfile(fname, full)
    else:
        _orig_savefig(fig, full, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    if ext != ".pdf":
        _orig_savefig(fig, os.path.join(outdir, f"{stem}_full.pdf"), bbox_inches="tight")

    axes = _reading_order(fig)
    letters = _panel_letters(len(axes))

    if PUB:
        # columns in the top row, not total axes: that is what sets panel width
        ncol = max(1, sum(1 for a in axes if round(a.get_position().y1, 2)
                          >= max(round(x.get_position().y1, 2) for x in axes) - 0.01))
        _rescale_for_print(fig, ncol)

        # Measured correction, ITERATED.  The first pass scales the AXES to the
        # target, but a panel is cropped to its tight bbox -- axes plus y-label,
        # legend, anything hanging outside -- so a decorated panel lands wider.
        #
        # One correction is not enough: shrinking the figure shrinks the axes but
        # NOT the 9 pt text, so crop width is affine in the scale factor, not
        # proportional, and a single divide overshoots.  Three passes converge.
        # Measured before this: 3.02-4.38 in against a 3.46 target, and 4.38+4.24
        # will not fit a 7.06 in row.
        for _ in range(3):
            fig.canvas.draw()
            r0 = fig.canvas.get_renderer()
            i0 = fig.dpi_scale_trans.inverted()
            got = []
            for a in axes:
                try:
                    got.append(a.get_tightbbox(r0).transformed(i0).width)
                except Exception:
                    pass
            if not got:
                break
            got.sort()
            # MAX, not median.  Every panel has to fit its own slot in the n-up
            # row, so the binding constraint is the widest one -- and the widest
            # is systematically the left column, which carries the y-label.
            # Targeting the median left that panel 4.14 in against a 3.53 in slot
            # and put a 2-up row at 7.54 in against 7.06 available.
            widest = got[-1]
            if widest <= 0:
                break
            corr = PUB_PANEL_IN / widest
            if abs(corr - 1.0) < 0.02:          # close enough; stop chasing
                break
            corr = max(0.5, min(2.0, corr))     # never take a wild step
            w, h = fig.get_size_inches()
            fig.set_size_inches(w * corr, h * corr)
            try:
                fig.tight_layout(pad=0.3)
            except Exception:
                pass

    # ── panel titles are STRIPPED before the panels are written ──────────────
    # They belong in the LaTeX \caption, not burnt into the artwork: there they
    # are set in the document font, are editable without regenerating anything,
    # and do not duplicate what the caption has to say anyway.
    #
    # Captured first, because after this the title survives ONLY in PANELS.md --
    # which is therefore written as caption-ready text, not as a description.
    #
    # Cleared BEFORE the bboxes are measured; measuring first would reserve the
    # empty band where the title used to be, and every panel would carry a strip
    # of white space at the top.
    titles = [ax.get_title() for ax in axes]
    for ax in axes:
        ax.set_title("")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()

    rows = []
    for letter, ax, title in zip(letters, axes, titles):
        try:
            bb = ax.get_tightbbox(renderer).transformed(inv)
            if PUB:
                # A HAIR of absolute padding, not a proportional expansion.
                # `expanded(1.0, 1.0)` is the identity, so the crop landed on the
                # exact ink boundary and the PDF MediaBox rounding shaved the
                # stem off the first glyph -- "binodal, 26 T*" printed as
                # "oinodal, 26 T*".  Most panels never showed it because their
                # y-axis LABEL sits outside the tick text and donates the slack;
                # a panel with categorical tick labels and no axis label has
                # none.  0.01 in = 0.72 pt per side, 0.4% of a 3.53 in panel, so
                # it does not disturb the printed-width targeting above.
                bb = bb.padded(0.01)
            else:
                bb = bb.expanded(PAD, PAD)
        except Exception:                                  # degenerate/empty axes
            rows.append((letter, title, "⚠ could not measure a bounding box — skipped"))
            continue
        for e in (".pdf", ".png"):
            _orig_savefig(fig, os.path.join(outdir, f"{stem}_{letter}{e}"),
                          bbox_inches=bb, dpi=200 if e == ".png" else None)
        pdf = os.path.join(outdir, f"{stem}_{letter}.pdf")
        kb = os.path.getsize(pdf) / 1024
        note = _describe(ax)
        # A vector panel is a few tens of kB.  Megabytes means a dense scatter or a
        # 10^5-point line went in verbatim: still vector, but slow to open and
        # awkward for a journal.  Worth flagging rather than discovering at proof.
        if kb > 900:
            note += (f"  ⚠ **{kb/1024:.1f} MB vector** — very many points; consider "
                     "`rasterized=True` on the dense artist (axes/text stay vector)")
        if _has_raster(ax):
            note += "  ⚠ contains a raster artist (image/pcolormesh) — not fully vector"
        rows.append((letter, title, note))

    for ax, title in zip(axes, titles):        # leave the live figure as we found it
        ax.set_title(title)

    # a figure-level legend belongs to no panel; say so rather than lose it
    fig_legends = [c for c in fig.legends]
    supt = fig._suptitle.get_text() if fig._suptitle is not None else ""

    with open(os.path.join(outdir, "PANELS.md"), "w") as fh:
        fh.write(f"# `{stem}` — {len(axes)} panels\n\n")
        fh.write(f"Composite: `{stem}_full{ext}`.  Panels are cropped from it in reading "
                 "order (top row first, left to right).\n\n")
        if supt:
            fh.write(f"**Composite suptitle** (NOT on the panels — this is caption text):\n\n"
                     f"> {supt}\n\n")
        if fig_legends:
            fh.write("⚠ **This figure has a figure-level legend**, which belongs to no single "
                     "panel and is therefore missing from all of them. Re-create it in the "
                     "caption or move it into an axes before using the panels separately.\n\n")
        fh.write("⚠ **Panel titles are stripped from the artwork.** They are reproduced "
                 "below and belong in the LaTeX `\\caption`; this file is the only place they "
                 "survive.\n\n")
        fh.write("| panel | file | title (for the caption) | content |\n|---|---|---|---|\n")
        for letter, title, desc in rows:
            t = _clean_title(title) or "*(untitled)*"
            fh.write(f"| ({letter}) | `{stem}_{letter}.pdf` | {t} | {desc} |\n")

        # ready to paste: the caption sentence and a subfigure skeleton
        fh.write("\n## Caption text\n\n```latex\n")
        for letter, title, _ in rows:
            t = _clean_title(title)
            if t:
                fh.write(f"({letter}) {t}.\n")
        fh.write("```\n\n## Skeleton\n\n```latex\n\\begin{figure*}[t]\n  \\centering\n")
        for letter, _, _ in rows:
            w = f"{1.0/max(len(rows),1):.2f}".rstrip('0').rstrip('.')
            fh.write(f"  \\includegraphics[width={w}\\textwidth]{{figures/{stem}_{letter}.pdf}}%\n")
        fh.write("  \\caption{...}\n  \\label{fig:" + stem + "}\n\\end{figure*}\n```\n")

    _report.append((stem, len(axes), outdir))
    print(f"    split {stem}: {len(axes)} panels → {outdir}")


def _patched_savefig(self, fname, *a, **kw):
    """Split, and by default do NOT overwrite the script's own output.

    ⚠ None of `numerical_notes/results/*.png` is tracked by git, so a regenerated
    composite that differs from the one on disk cannot be recovered.  Splitting is
    a packaging job and has no business rewriting the results tree, so the
    composite is written into the split folder instead and the original is left
    alone.  `--clobber` restores the normal behaviour when a genuine refresh is
    what you want.
    """
    out = _orig_savefig(self, fname, *a, **kw) if CLOBBER else None
    try:
        if isinstance(fname, (str, os.PathLike)):
            _split(self, os.fspath(fname))
    except Exception:
        print(f"    ⚠ split failed for {fname}:", file=sys.stderr)
        traceback.print_exc()
    return out


def candidates() -> list[str]:
    """Top-level scripts that save a figure — skipping backups and dead ends."""
    skip_dirs = ("old/", "backup/", "data_benchmark/", ".lake/")
    out = []
    for f in sorted(os.listdir(".")):
        if not f.endswith(".py") or f == os.path.basename(__file__):
            continue
        if any(f.startswith(d) for d in skip_dirs):
            continue
        try:
            s = open(f, errors="ignore").read()
        except OSError:
            continue
        if "savefig" in s and re.search(r"subplots?\(", s):
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="*", help="figure scripts to run and split")
    ap.add_argument("--all", action="store_true", help="every top-level script that saves a figure")
    ap.add_argument("--list", action="store_true", help="print what --all would run, then exit")
    ap.add_argument("--pub", action="store_true",
                    help="generate panels at the printed size (7.06/N in wide, "
                         f"{PUB_FONT_PT} pt; N from --pub-cols) into " + PUB_ROOT)
    ap.add_argument("--pub-cols", type=int, default=2, metavar="N",
                    help="panels per row in the printed figure (default 2). "
                         "Sets the generated width to 7.06/N inches.")
    ap.add_argument("--script-args", default="", metavar="'ARGS'",
                    help="arguments handed to each script, e.g. "
                         "--script-args='--ref lowk'.  Without this the scripts "
                         "run with no argv and always take their defaults.")
    ap.add_argument("--clobber", action="store_true",
                    help="also let each script overwrite its own output in "
                         "numerical_notes/results/ (default: leave it untouched — "
                         "those PNGs are NOT in git and cannot be restored)")
    args = ap.parse_args()

    global CLOBBER, PUB, OUT_ROOT, PUB_COLS, PUB_PANEL_IN
    CLOBBER = args.clobber
    PUB = args.pub
    PUB_COLS = max(1, args.pub_cols)
    PUB_PANEL_IN = PUB_TEXT_IN / PUB_COLS
    if PUB:
        OUT_ROOT = PUB_ROOT
        print(f"  pub: {PUB_COLS}-up -> panels {PUB_PANEL_IN:.2f} in wide")

    if args.list:
        for f in candidates():
            print(f)
        return

    scripts = candidates() if args.all else args.scripts
    if not scripts:
        ap.error("give one or more scripts, or --all")

    os.makedirs(OUT_ROOT, exist_ok=True)
    matplotlib.figure.Figure.savefig = _patched_savefig

    failed = []
    for i, script in enumerate(scripts, 1):
        print(f"[{i}/{len(scripts)}] {script}")
        sys.argv = [script] + shlex.split(args.script_args)
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit:
            pass
        except Exception as exc:
            failed.append((script, f"{type(exc).__name__}: {exc}"))
            print(f"    ⚠ {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            plt.close("all")

    print(f"\n{len(_report)} composite figure(s) split, "
          f"{sum(n for _, n, _ in _report)} panels total")
    if failed:
        print(f"\n⚠ {len(failed)} script(s) did not run to completion — their panels are "
              f"missing or partial:")
        for s, why in failed:
            print(f"    {s}: {why}")


if __name__ == "__main__":
    main()
