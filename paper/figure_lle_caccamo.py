#!/usr/bin/env python3
r"""figure_lle_caccamo.py — Figure 12: liquid-liquid demixing, the lambda line.

Two panels, each one 0.49\textwidth slot (3.53 in), generated at printed width so the
LaTeX scale factor is 1:

  (a) the T*-rho* phase portrait: BOTH lambda lines, LV binodal, crossover point,
      against Caccamo's GEMC binodal and consolute points
  (b) the demixing domes in T*-chi at five fixed densities: Caccamo's SGCMC data, the
      energy-route dome, and the DCF-route apex

⭐ THE POINT OF THIS FIGURE IS THAT THE TWO ROUTES BRACKET THE MEASUREMENT.  The same
demixing line computed two ways disagrees by more than either differs from experiment:

    compressibility route  `det M = 0` from chat_ij(0)   12-22 % BELOW the data
    energy route           d(mu_1-mu_2)/dchi = 0 from f  14-17 % ABOVE the data

and the measured consolute temperature sits between them at every one of the five
densities.  Neither line is "the" prediction; the PAIR is, which is why both are drawn
in both panels.

⚠ ONE COLOUR, THREE LINESTYLES.  Every curve of this model in both panels is FMSA-DP's
orange; what separates them is the CONSTRUCTION, and the key names it in three tokens:

    FMSA-E        energy route, d(mu_1 - mu_2)/dchi = 0 from f     SOLID
    FMSA-C-FE     compressibility route, ray-integrated f          DASHED
    FMSA-C-local c  compressibility route, chat(0) read pointwise  DOTTED

⛔ EXACT MSA IS DRAWN IN NEITHER PANEL (2026-09-04).  It agrees with FMSA-DP to
0.12-0.72 % on the lambda line and 0.1-0.7 % on the domes, so it printed as a purple
fringe under an orange stroke -- a curve a reader cannot resolve into a comparison.
The agreement is a NUMBER and the caption states it.  `domes_models.json` still holds
it; the figure stopped reading it.

THE DOME IS A REAL CURVE, NOT AN APEX MARKER.  At fixed (rho, T) the exchange potential
dmu = mu_1 - mu_2 is odd about chi = 1/2, so coexistence is the NON-TRIVIAL ROOT of
dmu = 0 -- no composition double-tangent search -- and equal pressure holds
automatically at fixed total density.  That is the ensemble Caccamo's SGCMC used.  The
compressibility route yields only the apex, because `det M = 0` is a spinodal condition and at
chi = 1/2 the spinodal touches the binodal; that is why it appears as a bar.

⚠ THIS IS THE FIGURE WHERE COMPOSITION IS A LIVE VARIABLE, AND THE ONLY ONE.  The
liquid-VAPOUR branch of this mixture carries chi_v = chi_l = 0.50 (1) in every row of
Caccamo's Table I -- forced by the sigma_1 = sigma_2, K_11 = K_22 symmetry.  A T-x-y
diagram of the VLE would be a vertical line at x = 0.5 and would measure nothing.

⚠ THE REFERENCE CONSOLUTE POINTS ARE A FIT, NOT PRINTED VALUES.  Caccamo et al. show
the consolute line in their Fig. 2 and print only the crossover; the per-isochore
T_cr plotted here is our fit of THEIR procedure (Delta_chi ~ |T - T_cr|^0.32, beta
fixed).  It is validated by extrapolating to rho* = 0.57, which returns T* = 0.6988
against their printed T*_e ~= 0.70.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LAM = 'numerical_notes/results/mixture_caccamo/lambda_line.json'
ENR = 'numerical_notes/results/mixture_caccamo/demixing_energy_route.json'
MOD = 'numerical_notes/results/mixture_caccamo/domes_models.json'
RAY = 'numerical_notes/results/mixture_caccamo/ray_route.json'
REF = 'data_benchmark/reference_caccamo1998.json'
OUT = 'numerical_notes/results/mixture_caccamo'

# Coloured by ROUTE, which is what this figure is about.
# ⭐⭐ ONE palette for the whole paper -- `figure_style.py` -- and this script was
# the worst offender: `C_CMP` and `C_EX` were the SAME `#7b3fa0`, so one purple
# meant "compressibility route" in Fig. 13 and "exact MSA" in Fig. 14, two floats
# apart.  Both lambda lines are FMSA-DP, differing only in the ROUTE, so they are
# now one orange separated by linestyle -- which is what the paper's other route
# comparisons do -- and purple is left to mean exact MSA and nothing else.
# `C_CON` was OZ+MSA's blue on a reference series; reference data is neutral.
import figure_style as S                                        # noqa: E402
C_CMP = S.color('FMSA-DP')          # compressibility route: same model,
LS_CMP = S.ROUTE['compressibility']       # det M = 0, DOTTED
# ⚠ DASH, overriding `S.ROUTE['compressibility-ray']`'s dash-dot, and the
# override is LOCAL on purpose.  The key now names the three constructions
# FMSA-E / FMSA-C-FE / FMSA-C-local c, so the two compressibility entries have to
# read as a PAIR against the solid energy line -- dashed and dotted are that
# pair, dash-dot and dotted are not.  `figure_style` keeps its dash-dot: DASH
# means "iterative OZ solve" there, and no OZ curve is drawn in this figure.
LS_RAY = S.DASH                           # ray-integrated f, DASHED
C_EN = S.color('FMSA-DP')
LS_EN = S.ROUTE['energy']
C_LV = S.AUX['other']     # the liquid-vapour binodal, a different transition
C_MC = S.DATA['primary']            # simulation
# ⚠ SAME source as C_MC -- Caccamo 1998, Table II against Table I -- so the same
# neutral black, separated by MARKER SHAPE rather than by colour.  Two greys said
# "two references" when there is one.
C_CON = S.DATA['primary']           # consolute (SGCMC) reference points
C_HB = S.color('OZ+HNCB')
PANEL_IN = 3.53
FS = 9

plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS,
                     'xtick.labelsize': FS - 1, 'ytick.labelsize': FS - 1,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.8})

ISO_MARK = {0.60: 'o', 0.70: 's', 0.75: '^', 0.80: 'D', 1.00: 'v'}


def load():
    return json.load(open(LAM)), json.load(open(REF)), json.load(open(ENR))


def panel_a(lam, ref, enr) -> None:
    # Both panels at ONE size: they sit side by side in a 2-up row now, and a
    # 1.55:1 panel beside a 1:0.86 one would be scaled to different text sizes
    # by \includegraphics.
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN * 0.86))

    # --- the two lambda lines: SAME transition, two routes ------------------
    g = lam['lambda_line']
    ax.plot([p['rho'] for p in g], [p['tstar'] for p in g],
            color=C_CMP, ls=LS_CMP, lw=1.8, zorder=4)
    ge = enr['lambda_line_energy']
    ax.plot([p['rho'] for p in ge], [p['tstar'] for p in ge],
            color=C_EN, ls=LS_EN, lw=1.8, zorder=4)
    # ⭐ THE THIRD LINE, and the accurate one.  Same route as the dotted curve --
    # the compressibility route -- but through the ray-integrated free energy
    # rather than the local Hessian `det M = 0`.  Measured at +5.4 to +9.0 %
    # against Caccamo on the five isochores, where the local Hessian runs -12 to
    # -22 % and the energy route +14 to +17 %.  Drawn heavier than the two it
    # sits between: it is the prediction, they are the bracket.
    try:
        gr = json.load(open(RAY))['lambda_line_ray']
    except (FileNotFoundError, KeyError):
        gr = []
    if gr:
        ax.plot([p['rho'] for p in gr], [p['tstar'] for p in gr],
                color=C_EN, ls=LS_RAY, lw=2.4, zorder=5)

    # ⛔ EXACT MSA IS NOT DRAWN, in either panel.  It agreed with FMSA-DP to
    # 0.12-0.72 % on this lambda line and to 0.1-0.7 % on the domes, so it printed
    # as a fringe under an orange curve -- a line a reader cannot resolve, and one
    # more entry in a key that had eleven.  The agreement is a NUMBER, and it is
    # stated in the text; drawing it added nothing the number does not say.

    # ⭐ WHERE PANEL (b) SAMPLES.  Each dome in (b) has its apex at that isochore's
    # consolute temperature, which is a point ON this lambda line -- so the five
    # apexes are five points here, in the SAME marker shapes (b) uses for its
    # isochores.  That correspondence was a sentence in the caption; now the
    # reader can follow a shape from one panel to the other.
    # ⚠ AS GUIDES, NOT AS MARKERS.  A filled marker sitting on the lambda line said
    # "a sampled point" in a figure where every other marker means "a point somebody
    # measured" -- one shape vocabulary doing two jobs.  A faint vertical at each of
    # (b)'s five densities makes the same statement: where it crosses a lambda line
    # is that isochore's consolute temperature, i.e. the apex of the matching dome.
    # STUBS, not full-height rules: each runs from the axis only as far as the
    # energy-route lambda line, so it points AT that isochore's consolute
    # temperature -- the apex of the matching dome in (b) -- instead of striping
    # the whole panel and competing with the data.
    tce_a = {r['rho']: r['tcr_energy'] for r in enr['isochores']}
    for rho_i in ISO_MARK:
        Tc_i = tce_a.get(rho_i)
        if Tc_i is None:
            continue
        ax.plot([rho_i, rho_i], [0.40, Tc_i], ls=(0, (1.0, 2.0)),
                color=S.AUX['tie'], lw=0.7, alpha=0.55, zorder=1)

    # --- the LV binodal (from the crossover walk) ---------------------------
    br = lam.get('crossover', {}).get('branch', [])
    if br:
        T = [b['tstar'] for b in br]
        ax.plot([b['rho_vap'] for b in br], T, color=C_LV, lw=1.4, zorder=3)
        ax.plot([b['rho_liq'] for b in br], T, color=C_LV, lw=1.4, zorder=3)

    # Close the dome on its critical point.  Same width law Figure 11 uses --
    # (rho_l - rho_v)^2 linear in T, rho_c by rectilinear diameters -- and the
    # extrapolation is 0.009 in T, so the dashed pair is short enough to draw.
    crit = lam.get('crossover', {}).get('critical')
    if br and crit:
        top = max(br, key=lambda b: b['tstar'])
        for key in ('rho_vap', 'rho_liq'):
            ax.plot([top[key], crit['rho']], [top['tstar'], crit['tstar']],
                    '--', color=C_LV, lw=1.0, zorder=3)
        # ⚠ The LV critical point is NOT marked.  The dashed closure already shows
        # where the dome ends, the panel's subject is the demixing line rather
        # than the LV one, and a third star competed with the two crossover
        # markers that ARE a measured comparison.  Its numbers stay in the caption.


    cep = lam.get('crossover', {}).get('cep')
    if cep:
        ax.plot([cep['rho']], [cep['tstar']], marker='*', ms=11, color=C_CMP,
                mec='k', mew=0.6, ls='none', zorder=7)

    # --- reference: GEMC liquid-vapour binodal, nu = 0.7 --------------------
    v = ref['vle']['nu_0.7']
    ic = {c: i for i, c in enumerate(ref['vle']['columns'])}
    for key in ('rho_vap', 'rho_liq'):
        # ⚠ Pentagon, not a circle.  Circle is ISO_MARK[0.60] now that shape means
        # isochore, and the LV binodal is a different observable entirely -- it must
        # not borrow a shape that names a demixing isochore.
        ax.plot([r[ic[key]] for r in v['data']], [r[ic['tstar']] for r in v['data']],
                marker='p', ms=5.5, mfc='white', mew=1.4, color=C_MC, ls='none',
                zorder=6)

    # --- reference: consolute points -- these land BETWEEN the two routes ---
    # ⭐ ONE SHAPE PER ISOCHORE, the same shapes (b) uses.  These five points sit at
    # exactly the five densities (b) resolves, so giving each the shape of its own
    # dome lets a reader carry a triangle from one panel to the other.  Plotted per
    # point rather than as one series for that reason.
    for r_ in lam['isochores']:
        ax.plot([r_['rho']], [r_['tcr_fit']],
                marker=ISO_MARK.get(round(r_['rho'], 2), '^'), ms=6.5, mfc='white',
                mew=1.4, color=C_CON, ls='none', zorder=8)

    c = ref['cep']
    # FILLED, not an outline.  An open star at this size disappeared into the
    # crowded crossover region -- the FMSA-DP star, the consolute circle and the
    # GEMC pentagons all overlap there.  Colour still says which is which: teal is
    # the measurement, orange the model.  The open/filled convention that matters
    # is (b)'s isochore markers, where it separates data from model pair by pair.
    ax.plot([c['rho_cep']], [c['tstar_cep']], marker='*', ms=12, color=C_MC,
            mec='k', mew=0.6, ls='none', zorder=9)

    ax.set_xlabel(r'$\rho^*$')
    ax.set_ylabel(r'$T^*$')
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(0.40, 1.75)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, 'lle_portrait.pdf'))
    fig.savefig(os.path.join(OUT, 'lle_portrait.png'), dpi=200)
    plt.close(fig)


def panel_b(lam, ref, enr) -> None:
    """Demixing domes, energy route, three models, against the SGCMC data.

    ⚠ NO COMPRESSIBILITY ROUTE HERE, and that is a result rather than an omission:
    for this model chat(0) is not integrable (the mixed partials differ by ~20 % of
    themselves), so its local Hessian and its ray-integrated free energy give
    demixing lines 64 % apart, and the ray-integrated one has no dome at all where
    the data sit.  Drawing one would be choosing between two constructions, not
    reporting a calculation.  The portrait figure carries its spinodal instead.
    """
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN * 0.86))

    L = ref['lle']
    ic = {c: i for i, c in enumerate(L['columns'])}
    by = {}
    for r in L['data']:
        by.setdefault(r[ic['rho']], []).append(
            (r[ic['tstar']], r[ic['chi1']], r[ic['chi2']]))

    try:
        mods = json.load(open(MOD))['models']
    except (FileNotFoundError, KeyError):
        mods = {}
    try:
        ray = json.load(open(RAY))
    except FileNotFoundError:
        ray = {}
    tce = {r['rho']: r['tcr_energy'] for r in enr['isochores']}

    def branches(pts, Tc):
        """(chi, T) of one dome, closed on its critical point."""
        T = [q['tstar'] for q in pts]
        c = [q['chi'] for q in pts]
        if Tc is not None:
            T, c = [Tc] + T, [0.5] + c
        return np.array(c), np.array(T)

    for rho in sorted(by):
        m = ISO_MARK.get(rho, 'o')
        pts = sorted(by[rho])
        T = [p[0] for p in pts]

        # --- measurement -------------------------------------------------
        for col in (1, 2):
            # ⚠ mfc='white', NOT 'none'.  A transparent face lets the grid and the
            # model curve through the marker, and at 4.2 pt the ring alone was not
            # reading as a data point.  Opaque white + a heavier edge is what makes
            # the measurement the most legible thing in the panel, which is what it
            # should be: the model points are filled, so open-vs-filled still says
            # data-vs-model.
            ax.plot([p[col] for p in pts], T, marker=m, ms=5.4, mfc='white', mew=1.4,
                    color=C_MC, ls='none', zorder=7)

        # --- the paper's model, plus markers AT THE MEASURED COMPOSITIONS --
        d = enr['domes'].get(f'{rho:.2f}', [])
        if d:
            ch, Td = branches(d, tce.get(rho))
            o = np.argsort(ch)
            for sgn in (+1, -1):
                ax.plot(np.where(sgn > 0, ch, 1.0 - ch), Td, color=C_EN, lw=1.5,
                        zorder=4)
            # Same marker shape, same x as the data, so each measured point has a
            # model point directly above or below it and the gap IS the error.
            # ⭐ RESTORED.  Removing these to make "marker = measurement" uniform
            # cost the thing the panel is for: FMSA-DP evaluated at the
            # measurement's OWN composition, so one shape's open-to-filled gap is
            # the error at that point and nothing has to be eyeballed against a
            # curve.  Open vs filled carries data vs model; the SHAPE carries the
            # isochore, and it is the same shape in (a).
            for col in (1, 2):
                xs = np.array([p[col] for p in pts])
                inside = np.minimum(xs, 1.0 - xs)
                ok = (inside >= ch[o].min()) & (inside <= ch[o].max())
                ax.plot(xs[ok], np.interp(inside[ok], ch[o], Td[o]), marker=m,
                        ms=4.2, color=C_EN, mec=C_EN, mew=0.9, ls='none', zorder=6)

        # --- the SAME model through the compressibility ray ----------------
        rr = ray.get('domes', {}).get(f'{rho:.2f}', [])
        if rr:
            ch, Td = branches(rr, ray.get('tcr', {}).get(f'{rho:.2f}'))
            o = np.argsort(ch)
            for sgn in (+1, -1):
                ax.plot(np.where(sgn > 0, ch, 1.0 - ch), Td, color=C_EN,
                        ls=LS_RAY, lw=1.6, zorder=5)
            # ⭐ The ray dome carries the measured-composition markers TOO, not just
            # the energy dome.  Both are FMSA-DP, so both deserve the open-to-filled
            # comparison at the measurement's own composition; giving it to only one
            # made the other look like a curve with nothing to check it against.
            # This is also why the key no longer names one route: the shape says
            # WHICH ISOCHORE, for either dome.
            for col in (1, 2):
                xs = np.array([p[col] for p in pts])
                inside = np.minimum(xs, 1.0 - xs)
                ok = (inside >= ch[o].min()) & (inside <= ch[o].max())
                ax.plot(xs[ok], np.interp(inside[ok], ch[o], Td[o]), marker=m,
                        ms=4.2, color=C_EN, mec=C_EN, mew=0.9, ls='none', zorder=6)

        # --- other models, curves only ------------------------------------
        # ⛔ 'exact MSA' is deliberately NOT in this list: see panel_a.  OZ-HNCB
        # stays because the loop is what a revived sweep would come back through.
        for key, col in (('OZ-HNCB', C_HB),):
            mm = mods.get(key)
            if not mm:
                continue
            dd = mm['domes'].get(f'{rho:.2f}', [])
            if not dd:
                continue
            ch, Td = branches(dd, mm['tcr'].get(f'{rho:.2f}'))
            for sgn in (+1, -1):
                ax.plot(np.where(sgn > 0, ch, 1.0 - ch), Td, color=col, lw=1.2,
                        zorder=3)

        # ⭐ The density is printed ONCE, and on the COMPRESSIBILITY-RAY dome.  The
        # energy-route dome already identifies itself -- it carries the filled
        # model markers, whose shape is the isochore -- while the dash-dot dome
        # carries no markers at all, so it is the one that needs the number.
        # Labelling both put ten numbers down the middle of the panel.
        # ⚠ ONE of the five carries `rho* =`, the rest only the number.  Five
        # bare decimals down the middle of a panel whose y axis is T* do not say
        # WHICH quantity they are; one labelled instance fixes the reading for
        # all five.  The topmost dome gets it -- it has the clearance.
        t_ray = ray.get('tcr', {}).get(f'{rho:.2f}')
        if t_ray is not None:
            lbl = rf'$\rho^*={rho:.2f}$' if rho == max(by) else f'{rho:.2f}'
            ax.annotate(lbl, xy=(0.5, t_ray), xytext=(0, 2),
                        textcoords='offset points', fontsize=FS - 2.5,
                        ha='center', va='bottom', color=C_EN)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$T^*$')
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, 'lle_domes.pdf'))
    fig.savefig(os.path.join(OUT, 'lle_domes.png'), dpi=200)
    plt.close(fig)


def legend_entries():
    """The figure's key, as (handles, labels, legend kwargs).

    ⚠ Extracted from main() so the SLIDE strip (figure_strips.py) draws this
    key rather than a second copy of it.  Two hand-maintained lists of the
    same eight entries would drift the moment either figure changed.
    """
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    # ⭐⭐ THE KEY NAMES THE CONSTRUCTION IN THREE TOKENS, not in a phrase.  Eleven
    # entries, two of them ten words long, made the key the widest object on the
    # page and pushed `width=\textwidth` into scaling its 8 pt type down.  The
    # three lambda lines are ONE model and three constructions, so the names are
    # built from one stem:
    #
    #     FMSA-E        energy route, d(mu_1 - mu_2)/dchi = 0 from f   SOLID
    #     FMSA-C-FE     compressibility route, ray-integrated f        DASHED
    #     FMSA-C-local c  compressibility route, chat(0) pointwise      DOTTED
    #
    # ⚠ The two C-entries must read as a PAIR -- one route, two constructions --
    # which is why they are dashed and dotted rather than dash-dot and dotted.
    # The caption expands all three; a key is a lookup, not a definition.
    # ⭐ ONE key for BOTH panels, now that they are one figure.  The two used to be
    # separate on the grounds that they "share almost no objects" -- true of the
    # objects, false of the ENCODING: solid orange is FMSA-DP's energy route in
    # both, dashed is the density ray in both, teal is the measurement in both.
    # Naming them once says that; naming them twice implied two vocabularies.
    #
    # ⭐ GEMC AND SGCMC ARE ONE ENTRY, `MC`.  They are one paper, one machine and
    # one class of method (Caccamo 1998, Table I against Table II); the split was
    # by ENSEMBLE, which is a fact about how the reference was produced and not
    # about anything a reader compares on this page.  Drawn as a two-marker
    # handle -- pentagon for the liquid-vapour branch, an isochore shape for the
    # consolute points -- so the merge costs no information: shape still says
    # which series, the label no longer spends two rows saying "reference".
    mc = (Line2D([], [], color=C_MC, marker='p', ms=5.5, mfc='white', mew=1.4,
                 ls='none'),
          Line2D([], [], color=C_CON, marker='^', ms=6.0, mfc='white', mew=1.4,
                 ls='none'))
    # ⭐ The two crossover stars are ONE entry for the same reason: they are the
    # same object -- where the lambda line meets the liquid-vapour branch --
    # measured once and computed once, and the caption already says they are told
    # apart by COLOUR and not by fill.  Two rows saying "crossover" said that the
    # figure carries two kinds of crossover, which it does not.
    xover = (Line2D([], [], color=C_EN, marker='*', ms=9, mec='k', mew=0.6,
                    ls='none'),
             Line2D([], [], color=C_MC, marker='*', ms=10, mec='k', mew=0.6,
                    ls='none'))
    # ⭐ ALL FIVE shapes side by side, not one stand-in.  The entry's job is to say
    # SHAPE IS THE ISOCHORE, and a single circle said the opposite -- that the model
    # points are circles, which they are for exactly one of the five densities.
    # In ISO_MARK order, so the row reads low rho to high, matching (b) bottom to top.
    iso = tuple(Line2D([], [], color=C_EN, marker=mk, ms=3.2, mew=0.8, ls='none')
                for mk in ISO_MARK.values())
    h = [Line2D([], [], color=C_EN, ls=LS_EN, lw=1.8),
         Line2D([], [], color=C_EN, ls=LS_RAY, lw=2.4),
         Line2D([], [], color=C_CMP, ls=LS_CMP, lw=1.8),
         # ⭐ ALL FIVE shapes side by side, not one stand-in.  The entry's job is to
         # say that SHAPE IS THE ISOCHORE, and a single circle said the opposite --
         # that the filled model points are circles, which they are for exactly one
         # of the five densities.  Drawn in ISO_MARK's own order, so the row reads
         # low rho to high, matching the domes bottom to top in (b).
         iso,
         mc,
         xover,
         Line2D([], [], color=C_LV, lw=1.4),
         Line2D([], [], color=S.AUX['tie'], lw=0.7, ls=(0, (1.0, 2.0)),
                alpha=0.7)]
    lab = ['FMSA-E',
           'FMSA-C-FE',
           'FMSA-C-local c',
           r'FMSA $\rho$ identification',
           'MC',
           'crossover: FMSA-DP / MC',
           'LV binodal',
           r'(a) $\to$ (b): the five isochores']
    return h, lab, dict(
        # ⚠ `iso` needs its OWN handler.  HandlerTuple's `pad` is per gap and in
        # units of the font size, so the 0.6 that spaces a two-marker handle nicely
        # spends 4x that on a five-marker one and crushes the shapes into a blob
        # inside a handlelength that has not grown.  pad=0 lets the five sections
        # use the width they were given.
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6),
                     iso: HandlerTuple(ndivide=None, pad=0.0)},
        strip_ncol=4, strip_rowmajor=True, handlelength=3.0,
        # ⚠ 0.85, not 1.1.  The five-shape isochore handle widened the strip to
        # 465.5 pt against a 466 pt \textwidth -- half a point of clearance, i.e.
        # none, and this budget has overrun before (9 entries at 5 columns printed
        # 43 pt into the margin).  The tighter column spacing buys ~9 pt back.
        columnspacing=0.85)


def main() -> None:
    lam, ref, enr = load()
    os.makedirs(OUT, exist_ok=True)
    panel_a(lam, ref, enr)
    panel_b(lam, ref, enr)

    import figure_legend
    h, lab, kw = legend_entries()
    figure_legend.emit(h, lab, 'lle_caccamo', out_dir=OUT, **kw)

    cep = lam.get('crossover', {}).get('cep')
    print(f'  wrote {OUT}/lle_caccamo_[ab].pdf')
    for r in enr['isochores']:
        b = r['tcr_fit']
        print(f"    rho*={r['rho']:.2f}  measured {b:.4f} | DCF {r['tcr_dcf']:.4f} "
              f"({100*(r['tcr_dcf']-b)/b:+.1f} %) | energy {r['tcr_energy']:.4f} "
              f"({100*(r['tcr_energy']-b)/b:+.1f} %)"
              f"   {'BRACKETED' if r['tcr_dcf'] < b < r['tcr_energy'] else 'NOT bracketed'}")
    cr = lam.get('crossover', {}).get('critical')
    if cr:
        print(f"    LV critical point: T*={cr['tstar']:.4f} rho*={cr['rho']:.4f} "
              f"(width-law, extrapolated {cr['extrapolation']:.4f})  "
              f"vs GEMC 0.72 / 0.34")
    if cep:
        print(f"    crossover: FMSA-DP ({cep['rho']:.3f}, {cep['tstar']:.3f})  "
              f"vs Caccamo (0.57, 0.70)")


if __name__ == '__main__':
    main()
