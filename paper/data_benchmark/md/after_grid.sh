#!/bin/bash
# Wait for the gridded MD batch, then rebuild both benchmark pools and re-score.
#
# The grid adds ~496 state points to a pool that had 111, and it adds a new
# potential class (LJ + hard sphere), so both pools have to be rebuilt from
# scratch rather than appended to: the DCF pool re-grades every entry's outer
# region and the RDF pool now carries a `potential` column.
set -u
cd "$(dirname "$0")"
REPO=../..

echo "=== waiting for the grid to finish ==="
until grep -q '^grid complete' grid.log 2>/dev/null; do sleep 60; done
grep '^grid complete' grid.log
echo "  failures: $(grep -c 'FAILED' grid.log)"

cd "$REPO"
echo
echo "=== rebuilding the DCF pool ==="
python3 data_benchmark/build_dcf_pool.py --force --skip-convert 2>&1 | tail -8
echo
echo "=== scoring the DCF pool ==="
python3 compare_models_dcf_multistate.py --workers 12 2>&1 | tail -6
echo
echo "=== rebuilding the RDF pool ==="
python3 compare_models_rdf_multistate.py --build-pool 2>&1 | tail -8
echo
echo "=== scoring the RDF pool ==="
python3 compare_models_rdf_multistate.py --workers 12 2>&1 | tail -6
echo
echo "AFTER_GRID_DONE"
