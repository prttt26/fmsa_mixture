"""Shared legend keys, emitted in BOTH shapes so the .tex chooses.

A figure's panels go into a 2-up grid at `0.49\\textwidth`.  Whether a shared key
costs any height is then a property of the PANEL COUNT, and nothing the plotting
script knows:

    even count   the grid is full, so the key needs a row of its own, and the
                 shape that fits a full row is a wide STRIP.
    odd count    the last row has an empty slot.  A key shaped like a panel
                 drops into it for FREE -- no extra row, no extra height.  This
                 is a BLOCK.

Panel selection changed several times while this paper was assembled, and each
time it flipped a figure between those two cases.  Baking one shape into the
script means regenerating the script to answer a LaTeX question, so every caller
here writes both files and the `.tex` picks one:

    \\subfloat[]{\\includegraphics[width=0.49\\textwidth]{..._c.pdf}} \\hfill
      \\raisebox{0.35\\height}{\\includegraphics{..._legend_block.pdf}}   % odd
    \\\\
    \\includegraphics[width=\\textwidth]{..._legend_strip.pdf}            % even

⚠ The block is FRAMED and the strip is not.  A strip sits directly under the
grid, where the panels' own axes frame it by proximity; a block sits in an empty
slot with nothing around it, and unframed it reads as labels adrift in white
space.

⚠ Sizes here are the PRINTED sizes -- 7.06 in is `\\textwidth` across both
columns, 3.53 in is one 2-up slot.  Generating at printed width is what keeps
the scale factor at 1, so 9 pt in the key is 9 pt on the page; see the header of
`figure_split.py`.  Both files are cropped to content, so these figsizes set the
LAYOUT (how many columns the entries wrap into), not the final width.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEXT_IN = 7.06                  # \textwidth, both columns
PANEL_IN = 3.53                 # one 2-up slot, 0.49\textwidth


# ⚠ The PAPER'S name for the method is "FMSA".  The data files, CSV columns and
# model keys all still say "FMSA-DP" (the doubly-propagated label from Tang and Lu),
# and lookups key on it, so the rename happens HERE, at the one point every key in
# the paper passes through -- not in the nine scripts that build the handles.
DISPLAY_RENAME = (("FMSA-DP", "FMSA"),)


def display_label(label):
    """The label as the paper prints it.  See DISPLAY_RENAME."""
    for old, new in DISPLAY_RENAME:
        label = label.replace(old, new)
    return label


def _column_major(seq, ncol):
    r"""Permute a row-major-intended list into the order Matplotlib will print.

    `Legend(ncol=k)` fills its columns TOP-TO-BOTTOM, left to right, so passing
    entries in reading order scatters them: seven entries at `ncol=4` come out
    `1 3 5 7 / 2 4 6`.  Callers that want to control which entries share a ROW
    pass reading order and set `strip_rowmajor=True`.
    """
    n = len(seq)
    nrow = -(-n // ncol)
    idx = [r * ncol + c for c in range(ncol) for r in range(nrow)
           if r * ncol + c < n]
    return [seq[i] for i in idx]


def emit(handles, labels, stem, out_dir='numerical_notes/results',
         strip_ncol=None, block_ncol=1, strip_fontsize=8, block_fontsize=9,
         handler_map=None, handlelength=2.2, quiet=False,
         strip_rowmajor=False, columnspacing=1.8):
    """Write `<stem>_legend_strip.png` and `<stem>_legend_block.png`.

    `handles`/`labels` are passed through untouched, so a caller that builds
    tuple handles for `HandlerTuple` keeps working -- pass its `handler_map`.

    Returns the two paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = [display_label(l) for l in labels]
    n = len(handles)
    out = []

    # ⚠ A ONE-ROW STRIP IS NOT FREE.  `bbox_inches='tight'` crops to the legend,
    # not to `figsize`, so entries that do not fit across TEXT_IN simply make the
    # file WIDER -- and `width=\textwidth` then scales the whole key down.  The
    # vle_hcy key ran to 817 pt, printing its 8 pt text at 4.6 pt.  Wrapping to
    # two rows halves the natural width and restores the scale factor to ~1.
    ns = strip_ncol or n
    nrow_s = -(-n // ns)
    for kind, figsize, ncol, fs, framed in (
            ('strip', (TEXT_IN, 0.40 * nrow_s), ns, strip_fontsize, False),
            ('block', (PANEL_IN, 0.52 * n + 0.5), block_ncol, block_fontsize, True)):
        h, l = handles, labels
        if kind == 'strip' and strip_rowmajor and ncol < n:
            h, l = _column_major(handles, ncol), _column_major(labels, ncol)
        fig = plt.figure(figsize=figsize)
        kw = dict(handles=h, labels=l, loc='center', ncol=ncol,
                  fontsize=fs, handlelength=handlelength, handletextpad=0.6,
                  columnspacing=columnspacing)
        if handler_map is not None:
            kw['handler_map'] = handler_map
        if framed:
            kw.update(frameon=True, edgecolor='0.6', facecolor='white',
                      framealpha=1.0, fancybox=False, borderpad=1.0,
                      labelspacing=1.2)
        else:
            kw.update(frameon=False, labelspacing=0.4)
        fig.legend(**kw)
        path = os.path.join(out_dir, f'{stem}_legend_{kind}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        # PDF as well: the .tex only ever cites the PDF, and a key that exists only
        # as a raster is the one piece of a vector figure that would print blurry.
        fig.savefig(path[:-4] + '.pdf', bbox_inches='tight')
        plt.close(fig)
        out.append(path)

    if not quiet:
        print(f'  wrote the shared key in both shapes: {stem}_legend_'
              f'{{strip,block}}  ({n} entries)')
    return tuple(out)
