#!/usr/bin/env python3
"""Digitize colour-separated g_ij(r) curves from a published figure.

This recovers APPROXIMATE data from a plot when a paper gives a figure but no
table.  Fidelity is a few percent at best -- always the lowest-quality data in
data_benchmark/, and always tagged `digitized-approx`.  It is only worth doing
for a curve at a state point (or size ratio) not otherwise tabulated.

Method
------
1. Render the figure page region to PNG (pdftoppm), or load an existing PNG.
2. Auto-detect the rectangular plot frame (the long dark axis lines).
3. Auto-detect tick marks just outside the bottom / left axes and calibrate
   pixel -> data by a linear fit against the tick VALUES you supply (the numbers
   printed on the axes).  This avoids hand-picking pixel coordinates.
4. For each named curve COLOUR, collect matching pixels inside the frame and, per
   x-pixel column, take the median data-y.  Colour separation handles overlapping
   solid/dashed/dotted styles that a shape-based tracer could not.
5. Write one CSV per figure (x + one column per curve) and a QA overlay PNG so
   the extraction can be eyeballed against the original.

A curve is only as good as its colour separation; where two curves share a
colour, or cross, expect local error.  Inspect the QA overlay before trusting it.

Usage (Yu & Jin 2008 Fig. 13, the worked example)
-------------------------------------------------
    python digitize_figure.py \
        --png fig13.png \
        --xticks 0 0.5 1 1.5 2 2.5 3 3.5 --yticks 0 0.5 1 1.5 2 2.5 \
        --curve g11 black --curve g22 red --curve g12 blue \
        --exclude 0.55 0.60 1.0 0.60 \
        --out literature_direct/rdf/yu_jin_2008_fig13.csv \
        --title "Yu & Jin 2008 Fig 13  HCTY ratio 2, T*=6 rho*=1.3"
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

COLOR_KEYS = {           # (predicate on R,G,B in 0..255) -> a curve colour
    'black': lambda r, g, b: (r < 90) & (g < 90) & (b < 90),
    'red':   lambda r, g, b: (r > 120) & (g < 90) & (b < 90),
    'blue':  lambda r, g, b: (b > 120) & (r < 90) & (g < 110),
    'green': lambda r, g, b: (g > 110) & (r < 90) & (b < 90),
}


def load_png(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'))


def render(pdf: Path, page: int, dpi: int, crop, out: Path) -> np.ndarray:
    cmd = ['pdftoppm', '-png', '-r', str(dpi), '-f', str(page), '-l', str(page)]
    if crop:
        x, y, w, h = crop
        cmd += ['-x', str(x), '-y', str(y), '-W', str(w), '-H', str(h)]
    cmd += [str(pdf), str(out.with_suffix(''))]
    subprocess.run(cmd, check=True)
    produced = out.with_name(out.stem + f'-{page:02d}.png')
    if not produced.exists():
        produced = out.with_name(out.stem + f'-{page}.png')
    return load_png(produced)


def detect_frame(img: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (xl, xr, yb, yt) of the plot frame, from the darkest long lines."""
    dark = (img.max(axis=2) < 110)
    col_dark = dark.mean(axis=0)     # fraction dark per column
    row_dark = dark.mean(axis=1)
    # frame lines are columns/rows that are dark over a large fraction of extent
    cand_c = np.where(col_dark > 0.5 * col_dark.max())[0]
    cand_r = np.where(row_dark > 0.5 * row_dark.max())[0]
    xl, xr = cand_c.min(), cand_c.max()
    yt, yb = cand_r.min(), cand_r.max()
    return int(xl), int(xr), int(yb), int(yt)


def detect_ticks(dark: np.ndarray, frame, axis: str, n: int) -> np.ndarray:
    """Pixel positions of tick marks just outside the bottom or left axis."""
    xl, xr, yb, yt = frame
    if axis == 'x':                       # ticks below the bottom edge
        strip = dark[yb + 3:yb + 12, xl:xr + 1].mean(axis=0)
        idx = _peaks(strip, n)
        return xl + idx
    strip = dark[yt:yb + 1, max(xl - 12, 0):xl - 2].mean(axis=1)   # left of left edge
    idx = _peaks(strip, n)
    return yt + idx


def _peaks(profile: np.ndarray, n: int) -> np.ndarray:
    """The n strongest, well-separated peak positions in a 1-D profile."""
    prof = profile.copy()
    if prof.max() <= 0:
        return np.array([])
    picks, guard = [], max(3, len(prof) // (2 * n))
    for _ in range(n):
        i = int(np.argmax(prof))
        if prof[i] <= 0:
            break
        picks.append(i)
        prof[max(0, i - guard):i + guard] = 0
    return np.array(sorted(picks))


def calibrate(pixels: np.ndarray, values: list[float]) -> tuple[float, float]:
    """Linear fit pixel -> data value; returns (slope, intercept)."""
    if len(pixels) != len(values):
        raise SystemExit(f'detected {len(pixels)} ticks but {len(values)} values '
                         f'given; adjust --x/yticks or the crop')
    a, b = np.polyfit(pixels, values, 1)
    return a, b


def _px_x(v, calib_x):
    return int(round((v - calib_x[1]) / calib_x[0]))


def _px_y(v, calib_y):
    return int(round((v - calib_y[1]) / calib_y[0]))


def extract(img, bounds, color: str, calib_x, calib_y, exclude):
    """Per x-column median data-y of the pixels of one curve colour.

    `bounds` = (xl, xr, y_top, y_bot) in pixels, derived from calibration and the
    plot's data range, so an open (L-shaped) axes with no frame box still works.
    """
    xl, xr, ytop, ybot = bounds
    R, G, B = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    mask = COLOR_KEYS[color](R, G, B)
    mask[:ytop, :] = mask[ybot + 1:, :] = False      # keep inside plotted rows
    mask[:, :xl] = mask[:, xr + 1:] = False          # keep inside plotted cols
    if exclude:                                       # blank a legend/label box (data units)
        ex0, ey0, ew, eh = exclude
        px0 = int((ex0 - calib_x[1]) / calib_x[0]); px1 = int((ex0 + ew - calib_x[1]) / calib_x[0])
        py0 = int((ey0 - calib_y[1]) / calib_y[0]); py1 = int((ey0 - eh - calib_y[1]) / calib_y[0])
        mask[min(py0, py1):max(py0, py1), min(px0, px1):max(px0, px1)] = False

    ax, bx = calib_x
    ay, by = calib_y
    xs, ys = [], []
    for px in range(xl, xr + 1):
        rows = np.nonzero(mask[:, px])[0]
        if rows.size == 0:
            continue
        xs.append(ax * px + bx)
        ys.append(ay * float(np.median(rows)) + by)
    return np.array(xs), np.array(ys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--png', type=Path)
    src.add_argument('--pdf', type=Path)
    ap.add_argument('--page', type=int)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--crop', type=int, nargs=4, default=None,
                    metavar=('x', 'y', 'w', 'h'))
    # Calibration: give EITHER explicit reference points (robust; the numbers
    # below map pixel->data and can come from axis ticks OR known curve features
    # such as a hard-core rise at a known r) OR tick lists for auto-detection.
    ap.add_argument('--xcal', type=float, nargs=4, default=None,
                    metavar=('px1', 'v1', 'px2', 'v2'))
    ap.add_argument('--ycal', type=float, nargs=4, default=None,
                    metavar=('py1', 'v1', 'py2', 'v2'))
    ap.add_argument('--xticks', type=float, nargs='+', default=None)
    ap.add_argument('--yticks', type=float, nargs='+', default=None)
    ap.add_argument('--curve', action='append', nargs=2, metavar=('name', 'color'),
                    required=True, help='repeatable; color in ' + '/'.join(COLOR_KEYS))
    ap.add_argument('--exclude', type=float, nargs=4, default=None,
                    metavar=('x0', 'y0', 'w', 'h'),
                    help='data-unit box to blank (e.g. the legend)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('--dx', type=float, default=0.05, help='output x sampling step')
    ap.add_argument('--axis-margin', type=int, default=6,
                    help='pixels to shrink off the left/bottom axis lines')
    ap.add_argument('--xrange', type=float, nargs=2, required=True,
                    metavar=('r_lo', 'r_hi'), help='data x-range of the plotted region')
    ap.add_argument('--yrange', type=float, nargs=2, required=True,
                    metavar=('g_lo', 'g_hi'), help='data y-range of the plotted region')
    args = ap.parse_args()

    scratch = args.out.parent / '_render'
    scratch.mkdir(parents=True, exist_ok=True)
    img = (load_png(args.png) if args.png
           else render(args.pdf, args.page, args.dpi, args.crop, scratch / 'fig.png'))

    frame = detect_frame(img)
    dark = (img.max(axis=2) < 110)

    def two_point(spec):
        px1, v1, px2, v2 = spec
        a = (v2 - v1) / (px2 - px1)
        return a, v1 - a * px1

    if args.xcal is not None:
        calib_x = two_point(args.xcal)
    elif args.xticks is not None:
        calib_x = calibrate(detect_ticks(dark, frame, 'x', len(args.xticks)), args.xticks)
    else:
        raise SystemExit('give --xcal or --xticks')
    if args.ycal is not None:
        calib_y = two_point(args.ycal)
    elif args.yticks is not None:
        calib_y = calibrate(detect_ticks(dark, frame, 'y', len(args.yticks)), args.yticks)
    else:
        raise SystemExit('give --ycal or --yticks')

    print(f'frame(xl,xr,yb,yt)={frame}')
    print(f'  calib x: {calib_x[0]:.5f}*px+{calib_x[1]:.3f}   '
          f'y: {calib_y[0]:.5f}*px+{calib_y[1]:.3f}')

    # plot region in pixels, from calibration + the data ranges (works for open axes)
    xlp, xrp = sorted((_px_x(args.xrange[0], calib_x), _px_x(args.xrange[1], calib_x)))
    ytp, ybp = sorted((_px_y(args.yrange[1], calib_y), _px_y(args.yrange[0], calib_y)))
    # Shrink inward by a margin so the black axis lines (which sit at r=r_lo and
    # g=g_lo) are not picked up as curve pixels -- otherwise every column carries
    # a spurious point on the bottom axis and a dark curve collapses toward it.
    m = args.axis_margin
    xlp, ytp = max(xlp + m, 0), max(ytp, 0)
    ybp = ybp - m
    bounds = (xlp, min(xrp, img.shape[1] - 1), ytp, min(ybp, img.shape[0] - 1))

    curves = {}
    for name, color in args.curve:
        xs, ys = extract(img, bounds, color, calib_x, calib_y, args.exclude)
        curves[name] = (xs, ys)
        print(f'  {name} ({color}): {len(xs)} sampled columns, '
              f'x in [{xs.min():.2f},{xs.max():.2f}]' if len(xs) else f'  {name}: none')

    # resample onto a shared grid spanning the UNION of the curves' supports; a
    # curve is written only where it has data (each pair has its own hard core,
    # so their supports differ) and left blank elsewhere -- never extrapolated.
    x_lo = min(c[0].min() for c in curves.values() if len(c[0]))
    x_hi = max(c[0].max() for c in curves.values() if len(c[0]))
    grid = np.arange(np.ceil(x_lo / args.dx) * args.dx, x_hi + args.dx / 2, args.dx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as fh:
        fh.write(f'# {args.title}\n')
        fh.write('# fidelity: digitized-approx (~few %); inspect the QA overlay\n')
        fh.write('# blank = outside that pair\'s digitized support (e.g. inside its core)\n')
        w = csv.writer(fh)
        w.writerow(['r'] + list(curves))
        for x in grid:
            row = [f'{x:.3f}']
            for name in curves:
                xs, ys = curves[name]
                row.append(f'{np.interp(x, xs, ys):.4f}'
                           if len(xs) and xs.min() - 1e-9 <= x <= xs.max() + 1e-9 else '')
            w.writerow(row)
    print(f'wrote {args.out}  ({len(grid)} rows on r={grid[0]:.2f}..{grid[-1]:.2f})')

    _qa_overlay(img, frame, curves, calib_x, calib_y, args.out.with_suffix('.qa.png'), args.title)
    return 0


def _qa_overlay(img, frame, curves, calib_x, calib_y, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax_slope, ax_int = calib_x
    ay_slope, ay_int = calib_y
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    style = {'g11': 'k', 'g22': 'r', 'g12': 'b'}
    for name, (xs, ys) in curves.items():
        px = (xs - ax_int) / ax_slope
        py = (ys - ay_int) / ay_slope
        ax.plot(px, py, '.', ms=2, color=style.get(name, 'lime'), label=f'{name} (extracted)')
    ax.legend(fontsize=8); ax.set_title(f'QA overlay: {title}', fontsize=9)
    ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f'wrote {path}  (check the dots lie on the printed curves)')


if __name__ == '__main__':
    raise SystemExit(main())
