# Directly-tabulated DCF c_ij(r) from the literature — none beyond TN 1331

Surveying the 14 paywalled PDFs in `../../../pdf/sources_paywall/` plus the open
sources, **no paper tabulates the full direct correlation function c_ij(r) of a
binary mixture** as numbers. The only tabulated c_ij(r) available to this project
is NIST TN 1331 (Appendix C), extracted separately into `../../tn1331/dcf/`.

Checked and rejected as c_ij sources:

- **Enciso, Lado, Lombardero, Abascal & Lago, JCP 87, 2249 (1987)** — reports
  *pair distribution functions* and *bridge functions* as FIGURES (Figs. 1–4),
  and thermodynamics as tables (Tables II–VI). No tabulated c_ij(r).
- All other binary-LJ / Yukawa-mixture papers give structure only as figures or
  as contact values, never c_ij(r) tables.

To obtain c_ij(r) at any state point where only g_ij(r) is available, run the
g → c conversion (`../../convert.py --from g ...`), keeping the truncation caveat
in mind (see `../../README.md`).
