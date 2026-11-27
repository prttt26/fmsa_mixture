#!/usr/bin/env python3
"""Run a classified script from the repository root: python code/run.py path/to/script.py [args]."""
from pathlib import Path
import os
import runpy
import sys

root = Path(__file__).resolve().parent.parent
code = root / "code"
archive = root / "archive" / "fmsa_dcf_wrong"
sys.path.insert(0, str(root))
for tree in (code, archive):
    for directory in sorted(p for p in tree.rglob("*") if p.is_dir()):
        sys.path.append(str(directory))
if len(sys.argv) < 2:
    raise SystemExit("Usage: python code/run.py <path within code or archive/fmsa_dcf_wrong> [args]")
name = sys.argv.pop(1)
script = Path(name)
if not script.is_file():
    if os.sep in name or (os.altsep and os.altsep in name):
        raise SystemExit(f'Script path does not exist: {name}')
    matches = [p for tree in (code, archive) for p in tree.rglob(script.name)]
    if len(matches) != 1:
        raise SystemExit(f"Script name must identify exactly one file: {name}")
    script = matches[0]
script = script.resolve()
sys.path.insert(0, str(script.parent))
# Child Python processes need the same module search path as this runner.
inherited = os.environ.get('PYTHONPATH', '')
module_dirs = [str(script.parent), str(root), str(code)]
module_dirs.extend(str(p) for tree in (code, archive) for p in sorted(tree.rglob('*')) if p.is_dir())
os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(module_dirs + ([inherited] if inherited else [])))
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
