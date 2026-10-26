"""Does the Table 1 seed trace, and how much of the Table 1 box is feasible? (convention sanity check)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prism as M  # noqa: E402

r = M.trace(M.SEED[:16])
print('seed counts [B, A2, C, img] =', r['counts'][0], ' S =', round(r['S'][0], 4), ' Pwedge =', round(r['Pwedge'][0], 3),
      ' W =', round(r['W'][0], 2))
for k in ('A1', 'B', 'A2', 'C', 'I'):
    print(f'  mean {k:2s} hit', np.round(np.nanmean(r['points'][k][0], 0), 1))
rng = np.random.default_rng(0)
n = 20000
X = M.LO[:16] + rng.random((n, 16))*(M.HI[:16] - M.LO[:16])
r = M.trace(X)
c = r['counts']/81
print(f'random Table-1 designs: >10% reach B {np.mean(c[:, 0] > .1):.3f}, A2-TIR {np.mean(c[:, 1] > .1):.3f}, '
      f'C {np.mean(c[:, 2] > .1):.3f}; >90% image {np.mean(c[:, 3] > .9):.4f}; '
      f'and Pwedge = 0 {np.mean((c[:, 3] > .9) & (r["Pwedge"] == 0)):.4f}; 100% image {np.mean(c[:, 3] == 1):.4f}')
