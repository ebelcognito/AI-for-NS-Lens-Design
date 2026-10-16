"""Versioned, bounded numerical observations from the actual spectrometer tracer."""
import numpy as np
from nslens.spectrometer import model

SCHEMA = 'spectrometer-optical-tokens-v1'
CELL_DIM = 10
GLOBAL_DIM = model.N_TOTAL + 1
PROMPT = ('Improve the current spectrometer using a bounded correction to its 36 normalized '
          'geometry parameters. Minimize the fixed geometric spot, ray survival, dispersion '
          'and magnification loss. Optical measurements follow. Output a continuous action.')


def validate_genes(genes):
    g = np.asarray(genes, dtype=float)
    if g.shape != (model.N_TOTAL,) or not np.isfinite(g).all() or np.any(abs(g) > 1):
        raise ValueError('Expected 36 finite normalized genes in [-1, 1]')
    return g


def apply_action(genes, action, radius):
    g = validate_genes(genes)
    a = np.asarray(action, dtype=float)
    if a.shape != g.shape or not np.isfinite(a).all() or np.any(abs(a) > 1 + 1e-6):
        raise ValueError('Action must be 36 finite values in [-1, 1]')
    if not np.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError('Radius must be in (0, 1]')
    return np.clip(g + radius * np.clip(a, -1, 1), -1, 1)


def observe(genes, radius=.05, fields=3, waves=3, pupils=5):
    """One trace. Cell positions remain explicit; missing rays never disappear.

    Columns: slit/5mm, (wavelength-.0008)/.0003mm, survival,
    signed-log centroid x/y (10mm scale), log1p spot RMS/mm,
    log1p WFE/waves, obstruction fraction, empty-cell flag,
    fraction of rays with finite detector coordinates. Log features clip at 10.
    Global token: 36 normalized genes followed by action radius.
    """
    g = validate_genes(genes)
    apply_action(g, np.zeros_like(g), radius)
    with np.errstate(all='ignore'):
        r = model.trace(g, fields=fields, waves=waves, pupils=pupils)
        finite = np.isfinite(r['xy']).all(axis=-1)
        good = r['good'] & finite
        count = good.sum(axis=2)
        cent = np.where(good[..., None], r['xy'], 0).sum(axis=2) / np.maximum(count[..., None], 1)
        distance = ((r['xy'] - cent[:, :, None, :]) ** 2).sum(axis=-1)
        rms = np.sqrt(np.where(good, distance, 0).sum(axis=2) / np.maximum(count, 1))
        m = r['model']
        target = m['image'] + cent[..., 0, None] * m['image_x'] + cent[..., 1, None] * m['image_y']
        path = r['opl'] + np.linalg.norm(target[:, :, None] - r['last_surface'], axis=-1) + r['phase']
        piston = np.where(good, path, 0).sum(axis=2) / np.maximum(count, 1)
        wfe = np.sqrt(np.where(good, (path - piston[:, :, None]) ** 2, 0).sum(axis=2) / np.maximum(count, 1)) / r['wavelength'][:, :, 0]
        signed_log = lambda x: np.sign(x) * np.log1p(abs(x))
        cells = np.stack([r['slit'][:, :, 0] / 5,
                          (r['wavelength'][:, :, 0] - .0008) / .0003,
                          good.mean(axis=2), signed_log(cent[..., 0] / 10),
                          signed_log(cent[..., 1] / 10), np.log1p(rms), np.log1p(wfe),
                          r['obstructed'].mean(axis=2), (count == 0).astype(float),
                          finite.mean(axis=2)], axis=-1)
    cells = np.clip(np.nan_to_num(cells, nan=10, posinf=10, neginf=-10), -10, 10)
    return cells.reshape(-1, CELL_DIM).astype('float32'), np.r_[g, radius].astype('float32')
