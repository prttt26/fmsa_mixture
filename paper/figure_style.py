r"""figure_style.py — ONE colour, ONE marker and ONE linestyle per model, for every
figure in the paper.

Each figure script used to carry its own palette.  They agreed on FMSA-DP's orange
and on nothing else, and by the time there were eighteen figures the disagreements
had become four separate ways for a reader to be misled:

1. **purple meant three things.**  `#7a4fbf` was FMSA-DP(fmt) in the \DCF{} figures,
   `#7b3fa0` was exact MSA in the speed, VLE and LLE figures, and the *same*
   `#7b3fa0` was the compressibility *route* in `figure_lle_caccamo.py` -- two
   meanings inside one script.  The two hexes differ by one or two units per
   channel: on paper they are one colour.
2. **the OZ markers were swapped between figure families.**  `OZ+MSA` was a circle
   and `OZ+HNCB` a square in the \DCF{}/\RDF{} figures; in the speed figures it was
   the other way round.  Arbitrary would have been better -- a reader who learns one
   convention is actively wrong in the other.
3. **OZ-HNCB was red in Figs 5-11 and teal in Figs 12-14.**
4. **blue `#2a78d6` was OZ+MSA, and also the GEMC `N=216` reference series, and also
   the consolute reference points.**

⭐ **THE RULE, and it is the only one that matters:** a colour names a MODEL, a
linestyle names how that model was SOLVED or which ROUTE it was taken through, and
reference data is neutral -- black and greys, never a model colour.

    solid       analytic / closed form   (FMSA-DP, exact MSA)
    dashed      iterative OZ solve       (OZ+MSA, OZ+HNCB)
    dotted      a second thermodynamic route through the SAME model

⚠ ONE DOCUMENTED EXCEPTION.  `figure_dcf_samples.py` and `figure_rdf_samples.py`
spend the linestyle on the REGION (overall / in-core / out-core, whole-range /
first-shell) instead, because that is the axis those panels are about.  They take
their colour and marker from here and their linestyle from themselves, and their
keys say which is which.  Do not "fix" them to dashes: it would erase the split
the panels exist to show.

⚠ `FMSA-DP(fmt)` and `FMSA-DP(py)` are ONE model with two hard-sphere references, so
they are two lightnesses of one orange rather than two unrelated hues.  That is also
why dropping the `fmt` leg costs the palette nothing.

⚠ Deuteranopia was the constraint on the choice: orange/blue/purple separate under
red-green confusion, and the red `#c23b22` is dark enough to hold against the two
oranges by lightness alone.  Do not "fix" the red to something brighter.

Use:

    import figure_style as S
    ax.plot(x, y, **S.line('OZ+HNCB'))          # colour + ls + marker
    ax.bar(...,  color=S.color('FMSA-DP(py)'))
    ax.plot(x, y, **S.line('FMSA-DP', route='compressibility'))
"""

from __future__ import annotations

# ── linestyles ────────────────────────────────────────────────────────────────
# Explicit dash tuples, not '--'/':'.  Matplotlib's named styles scale with the
# line width, so a 0.9 pt reference line and a 2.2 pt model line drawn '--' do not
# read as the same style; the tuples are in points and do.
SOLID = '-'
DASH = (0, (5.0, 2.0))
DOT = (0, (1.0, 1.6))
DASHDOT = (0, (5.0, 1.6, 1.0, 1.6))

# ⚠ FOUR patterns, all distinct, because Fig. 13 carries three lines of ONE model:
# the compressibility route appears twice there, as the local Hessian `det M = 0`
# and as the ray-integrated free energy, and those are two CONSTRUCTIONS of one
# route rather than two routes.  Giving them one pattern would say they are the
# same curve; giving the ray `virial`'s pattern would make the two collide the
# first time anyone draws both.
DASHDOTDOT = (0, (7.0, 1.5, 1.0, 1.5, 1.0, 1.5))

ROUTE = {
    'energy': SOLID,
    'compressibility': DOT,               # det M = 0, the local Hessian
    'compressibility-ray': DASHDOT,       # ray-integrated free energy
    'virial': DASHDOTDOT,
}

# ── the models ────────────────────────────────────────────────────────────────
MODEL = {
    'FMSA-DP':      dict(color='#e08c36', ls=SOLID, marker='v'),
    'FMSA-DP(py)':  dict(color='#e08c36', ls=SOLID, marker='v'),
    'FMSA-DP(fmt)': dict(color='#a35c12', ls=SOLID, marker='^'),
    'exact MSA':    dict(color='#7b3fa0', ls=SOLID, marker='D'),
    'OZ+MSA':       dict(color='#2a78d6', ls=DASH,  marker='o'),
    'OZ+HNCB':      dict(color='#c23b22', ls=DASH,  marker='s'),
}

# The names that reach a reader.  Kept here so a rename is one edit and cannot
# leave two figures disagreeing about what the same curve is called.
DISPLAY = {
    'FMSA-DP':      'FMSA-DP',
    'FMSA-DP(py)':  'FMSA-DP(PY)',
    'FMSA-DP(fmt)': 'FMSA-DP(WBFMT)',
    'exact MSA':    'exact MSA',
    'OZ+MSA':       'OZ+MSA',
    'OZ+HNCB':      'OZ+HNCB',
}

# Aliases every script that predates this module used for the same four things.
ALIAS = {
    'MSA-exact': 'exact MSA',
    'exact-MSA': 'exact MSA',
    'OZ-MSA': 'OZ+MSA',
    'OZ-HNCB': 'OZ+HNCB',
    'FMSA-DP(WBFMT)': 'FMSA-DP(fmt)',
    'FMSA-DP(PY)': 'FMSA-DP(py)',
}

# ── reference data ────────────────────────────────────────────────────────────
# ⭐ A MEASUREMENT GETS ITS OWN HUE, and it is one no model may use.  Neutral black
# was the first attempt and it did not survive Figs 13-14: the LV binodal there is
# drawn in AUX['other'] = #5a5a5a, a grey close enough to #1a1a1a that black data
# markers sitting on it stopped reading as data.  Teal is the gap in the model
# palette -- orange, dark orange, blue, purple, red -- so a reader can find the
# measurement in any panel without consulting the key.
#
# ⚠ This is still the anti-collision rule, not a relaxation of it: a simulation
# series in a MODEL's colour remains the one mistake in this file's history that
# changed what a panel appeared to say.  Teal is reserved for data in exactly the
# way orange is reserved for FMSA-DP.
DATA = {
    'primary': '#0f7d76',      # the reference a panel is scored against
    'secondary': '#63b3ac',    # a second reference: system size, a second source
    'tertiary': '#9ecfca',
    'band': '#cfe6e3',         # a quoted uncertainty band
}

# Objects that are not models and not data: construction lines, phase boundaries,
# an interval where nothing exists.  Deliberately outside every model hue.
AUX = {
    'tie': '#555555',          # Maxwell tie line, common tangent
    'spinodal': '#4a4a4a',     # a derived boundary of the model it is drawn on
    'knot': '#8a6d3b',         # R_ij / lambda_ij markers
    'shade': '#ece7f2',        # a filled region under a purple curve
    'other': '#5a5a5a',        # a DIFFERENT transition, drawn for context only
    'bridge': '#9a9a9a',       # a COMPLETION across a gap: deliberately colourless
}


# Two ERRORS compared on one axis, which is not a model comparison and must not
# borrow model colours.  The method's own truncation error IS FMSA-DP's, so it
# takes FMSA-DP's orange; the parity error is a BUG -- a wrong branch of the
# closed form, belonging to no method -- so it takes a dark neutral.
ERR = {
    'bug': '#3f3f3f',
    'method': MODEL['FMSA-DP']['color'],
}


def _key(model: str) -> str:
    return ALIAS.get(model, model)


def color(model: str) -> str:
    return MODEL[_key(model)]['color']


def marker(model: str) -> str:
    return MODEL[_key(model)]['marker']


def linestyle(model: str, route: str | None = None):
    if route is not None:
        return ROUTE[route]
    return MODEL[_key(model)]['ls']


def disp(model: str) -> str:
    return DISPLAY[_key(model)]


def line(model: str, route: str | None = None, **kw) -> dict:
    """Keyword arguments for `ax.plot`: colour, linestyle and marker at once.

    Anything passed in `kw` wins, so a caller that wants the model's colour but no
    marker writes `S.line('OZ+MSA', marker='')`.
    """
    m = MODEL[_key(model)]
    out = dict(color=m['color'], ls=linestyle(model, route), marker=m['marker'])
    out.update(kw)
    return out


def styles(*models: str) -> dict:
    """`{model: (color, ls, marker)}` -- the shape the older scripts' `STYLE` had,
    so converting one is a substitution rather than a rewrite."""
    return {m: (color(m), linestyle(m), marker(m)) for m in models}


# ---------------------------------------------------------------------------
# No grids
# ---------------------------------------------------------------------------
# ⭐ PROJECT CONVENTION (2026-09-05): figures bound for the paper carry NO grid.
# 50 `ax.grid(...)` calls were removed across the 18 scripts that feed
# `FMSA_dp_output/tex/figures/`.  At 3.5 in column width a grid competes with the
# curves for the reader's attention and the tick labels already carry the scale;
# the same holds on a projector, so the slide strips do not draw one either.
#
# Vertical/horizontal REFERENCE lines are a different thing and stay: R_ij, the
# breakpoint |lambda_ij|, g = 1, a spinodal. Those are physics, not chrome.
#
# ⚠ Do not reintroduce `ax.grid(...)` in a script listed in any
# `tex/figures/*/panels.md`. There is no rcParams switch doing this for you --
# an explicit ax.grid() call would override one anyway.

# ---------------------------------------------------------------------------
# Slide type
# ---------------------------------------------------------------------------
# ⚠ These are NOT the paper's sizes.  The paper's figures are 9 pt in a 3.5 in
# column; a slide figure gets SCALED DOWN to fit the frame, so what has to be
# large is the type RELATIVE TO THE CANVAS, not its nominal point size.  A 16.5 in
# wide figure dropped on a 13.3 in slide shrinks by 0.8x, and a nominal 20 pt label
# lands at 16 pt.  Hence: bigger points AND a narrower canvas.
#
# Applies to the working-note strips only (figure_strips.py, figure_vle_sample.py).
# Never import this into a script that writes into FMSA_dp_output/tex/figures/.
SLIDE_RC = {
    'font.size': 23, 'axes.labelsize': 27, 'axes.titlesize': 27,
    'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 20,
    'figure.titlesize': 27, 'axes.linewidth': 1.3,
    'xtick.major.width': 1.3, 'ytick.major.width': 1.3,
    'xtick.major.size': 6.0, 'ytick.major.size': 6.0,
}
SLIDE_STRIP_IN = (15.5, 5.4)      # 1x3 strip; ~2.9:1, close to a 16:9 content box


def use_slide_fonts() -> None:
    """Switch the current pyplot session to slide type."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(SLIDE_RC)
