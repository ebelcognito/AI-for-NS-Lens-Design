"""Nearest-neighbor retrieval of observed geometry moves by optical state.

This is a lightweight proposal baseline: it retrieves complete teacher moves
from similar optical states instead of blending potentially conflicting moves.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class TransitionPolicy:
    """Retrieve distinct observed 9-gene moves for a 13-gene system state."""

    def __init__(self):
        self.scaler = None
        self.neighbors = None
        self._deltas = None

    def fit(self, X, D):
        """Fit on optical features ``X`` and corresponding geometry deltas ``D``."""
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2D array")
        if D.ndim != 2 or D.shape != (X.shape[0], 9):
            raise ValueError("D must have shape (len(X), 9)")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(D)):
            raise ValueError("X and D must contain only finite values")

        scaler = StandardScaler().fit(X)
        transformed = scaler.transform(X)
        neighbors = NearestNeighbors(n_neighbors=len(X), algorithm="auto")
        neighbors.fit(transformed)
        self.scaler = scaler
        self.neighbors = neighbors
        self._deltas = D.copy()
        return self

    def propose(self, features, state, count=8, radius=0.15):
        """Return nearby teacher moves applied to ``state``.

        The first nine genes are updated by retrieved deltas, capped in norm at
        ``radius`` and clipped to their legal range. The trailing four genes
        are copied exactly from the input state.
        """
        if self.scaler is None:
            raise RuntimeError("fit must be called before propose")
        features = np.asarray(features, dtype=float)
        state = np.asarray(state, dtype=float)
        if features.ndim != 1 or features.shape[0] != self.scaler.n_features_in_:
            raise ValueError("features must be a 1D vector matching fitted X")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite 13-element vector")
        if np.any(state < -1) or np.any(state > 1):
            raise ValueError("state values must be within [-1, 1]")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("count must be a positive integer")
        if not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be a positive finite scalar")

        query = self.scaler.transform(features.reshape(1, -1))
        order = self.neighbors.kneighbors(query, return_distance=False)[0]
        proposals = []
        seen = set()
        for idx in order:
            delta = self._deltas[idx].copy()
            norm = float(np.linalg.norm(delta))
            if norm > radius:
                delta *= radius / norm
            key = tuple(np.round(delta, decimals=10))
            if key in seen:
                continue
            seen.add(key)
            child = state.copy()
            child[:9] = np.clip(state[:9] + delta, -1.0, 1.0)
            proposals.append(child)
            if len(proposals) == count:
                break

        while len(proposals) < count:
            proposals.append(state.copy())
        return np.asarray(proposals, dtype=float)
