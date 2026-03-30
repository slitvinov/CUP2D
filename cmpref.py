#!/usr/bin/env python3
import sys, struct, math, os, glob

ref_dir = os.path.expanduser('~/bak/ref')
cur_dir = os.path.expanduser('~/CUP2D')

ok = True
for ref_path in sorted(glob.glob(os.path.join(ref_dir, '*.raw'))):
    name = os.path.basename(ref_path)
    cur_path = os.path.join(cur_dir, name)
    if not os.path.exists(cur_path):
        print(f'MISSING {name}')
        ok = False
        continue
    ref_data = open(ref_path, 'rb').read()
    cur_data = open(cur_path, 'rb').read()
    if len(ref_data) != len(cur_data):
        print(f'SIZE  {name}  ref={len(ref_data)}  cur={len(cur_data)}')
        ok = False
        continue
    n = len(ref_data) // 8
    rv = struct.unpack(str(n) + 'd', ref_data)
    cv = struct.unpack(str(n) + 'd', cur_data)
    diffs = [abs(a - b) for a, b in zip(rv, cv)
             if math.isfinite(a) and math.isfinite(b)
             and abs(a) < 1e6 and abs(b) < 1e6]
    if not diffs:
        continue
    maxd = max(diffs)
    tol = 1e-4
    status = 'OK  ' if maxd <= tol else 'FAIL'
    if maxd > tol:
        ok = False
    print(f'{status}  {name}  max_diff={maxd:.2e}')
sys.exit(0 if ok else 1)
