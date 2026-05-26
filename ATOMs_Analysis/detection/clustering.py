"""
detection/clustering.py
-----------------------
Gaussian Mixture Model clustering on ATOMs attention profiles.

Motivation: the attention profile distribution may be multimodal — different
driving situations (intersection, highway, narrow street) may produce distinct
attention clusters rather than a single Gaussian. A GMM can capture this
structure and may improve anomaly detection over a single Mahalanobis baseline.

Classes
-------
GMMClustering
    Fits a GMM to a baseline attention series, assigns cluster labels to
    any new profile, and exposes per-cluster Mahalanobis detectors so that
    anomaly scoring uses the nearest cluster's baseline rather than the
    global mean.

Usage
-----
# Fit
gmm = GMMClustering(n_components=4, covariance_type="full")
gmm.fit(baseline_series)          # [N, C] attention profiles
gmm.save("detection/gmm.npz")

# Assign & score
cluster_id = gmm.predict(profile)
distance   = gmm.score(profile)   # distance to nearest cluster centre

# Visualise (see visualization.py for PCA overlay)
labels = gmm.predict_batch(baseline_series)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ATOMs_Analysis.utils.distance_computer import DistanceComputer


class GMMClustering:
    """
    GMM fitted on ATOMs attention profiles.

    Parameters
    ----------
    n_components    : int   number of Gaussian components.
    covariance_type : str   passed to sklearn GaussianMixture.
                            'full' (default) is most expressive but requires
                            N >> C^2 samples.  Use 'diag' if N is small.
    random_state    : int   for reproducibility.
    ridge           : float regularisation on per-cluster covariance matrices
                            (same role as in MahalanobisDetector).
    """

    def __init__(
        self,
        n_components:    int   = 4,
        covariance_type: str   = "full",
        random_state:    int   = 42,
        ridge:           float = 1e-6,
    ):
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.random_state    = random_state
        self.ridge           = ridge
        self._fitted         = False

        # Filled by fit()
        self._gmm            = None
        self.means_:         Optional[np.ndarray] = None   # [K, C]
        self.covariances_:   Optional[np.ndarray] = None   # [K, C, C] or [K, C]
        self.weights_:       Optional[np.ndarray] = None   # [K]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray) -> "GMMClustering":
        """
        Fit a GMM to the baseline attention series.

        Parameters
        ----------
        data : np.ndarray [N, C]  — per-frame attention profiles (row-normalized).

        Returns self for chaining.
        """
        from sklearn.mixture import GaussianMixture

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected [N, C], got {data.shape}")

        n_samples = data.shape[0]
        if n_samples < self.n_components:
            raise ValueError(
                f"n_components={self.n_components} > n_samples={n_samples}. "
                "Reduce n_components or collect more data."
            )

        self._gmm = GaussianMixture(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
            random_state    = self.random_state,
            max_iter        = 200,
            n_init          = 5,            # multiple initialisations for stability
        )
        self._gmm.fit(data)

        self.means_   = self._gmm.means_.copy()    # [K, C]
        self.weights_ = self._gmm.weights_.copy()  # [K]

        # Normalise covariances to full [K, C, C] regardless of covariance_type
        self.covariances_ = self._expand_covariances(self._gmm)

        # Pre-compute per-cluster precision matrices
        C = data.shape[1]
        ridge_mat = self.ridge * np.eye(C)

        self._fitted = True
        print(
            f"[GMMClustering] Fitted {self.n_components} components on "
            f"{n_samples} samples.  "
            f"Log-likelihood: {self._gmm.lower_bound_:.4f}"
        )
        return self

    # ------------------------------------------------------------------
    # Prediction & scoring
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> int:
        """
        Return the most probable cluster index for a single profile.

        Parameters
        ----------
        x : np.ndarray [C]

        Returns
        -------
        int — cluster index in [0, n_components)
        """
        self._check_fitted()
        return int(self._gmm.predict(np.asarray(x, dtype=np.float64).reshape(1, -1))[0])

    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster indices for a batch.

        Parameters
        ----------
        data : np.ndarray [N, C]

        Returns
        -------
        np.ndarray [N] int
        """
        self._check_fitted()
        return self._gmm.predict(np.asarray(data, dtype=np.float64))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Return soft cluster membership probabilities [K] for a single profile.
        """
        self._check_fitted()
        return self._gmm.predict_proba(
            np.asarray(x, dtype=np.float64).reshape(1, -1)
        )[0]

    def score(self, x: np.ndarray) -> float:
        """
        Mahalanobis distance to the nearest cluster centre.

        This is the anomaly score used for OOD detection with the GMM
        baseline.  A profile that belongs to no cluster will have large
        distance to all cluster centres and therefore a large score.

        Parameters
        ----------
        x : np.ndarray [C]

        Returns
        -------
        float — distance to nearest cluster mean.
        """
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        distances = []
        for k in range(self.n_components):
            dist2 = DistanceComputer.compute_mahalanobis(self.means_[k], self.covariances_[k], x, self.ridge)
            distances.append(dist2)
        return float(min(distances))

    def score_batch(self, data: np.ndarray) -> np.ndarray:
        return np.array([self.score(row) for row in data])

    def score_per_cluster(self, x: np.ndarray) -> np.ndarray:
        """
        Return Mahalanobis distance to every cluster centre.

        Parameters
        ----------
        x : np.ndarray [C]

        Returns
        -------
        np.ndarray [K]
        """
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        distances = []
        for k in range(self.n_components):
            dist2 = DistanceComputer.compute_mahalanobis(self.means_[k], self.covariances_[k], x, self.ridge)
            distances.append(dist2)
        return np.array(distances)

    # ------------------------------------------------------------------
    # BIC / AIC for model selection
    # ------------------------------------------------------------------

    def bic(self, data: np.ndarray) -> float:
        """BIC score (lower is better) — use for selecting n_components."""
        self._check_fitted()
        return float(self._gmm.bic(np.asarray(data, dtype=np.float64)))

    def aic(self, data: np.ndarray) -> float:
        """AIC score (lower is better)."""
        self._check_fitted()
        return float(self._gmm.aic(np.asarray(data, dtype=np.float64)))

    @staticmethod
    def select_n_components(
        data: np.ndarray,
        max_components: int = 10,
        criterion: str = "bic",
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> Tuple[int, np.ndarray]:
        """
        Fit GMMs with 1..max_components and return the optimal K by BIC or AIC.

        Parameters
        ----------
        data            : np.ndarray [N, C]
        max_components  : int
        criterion       : 'bic' or 'aic'
        covariance_type : str
        random_state    : int

        Returns
        -------
        best_k   : int
        scores   : np.ndarray [max_components]  — BIC/AIC for K=1..max_components
        """
        from sklearn.mixture import GaussianMixture

        data   = np.asarray(data, dtype=np.float64)
        scores = []
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=random_state,
                n_init=5,
            )
            gmm.fit(data)
            scores.append(gmm.bic(data) if criterion == "bic" else gmm.aic(data))

        scores   = np.array(scores)
        best_k   = int(np.argmin(scores)) + 1
        print(f"[GMMClustering] Optimal K={best_k} by {criterion.upper()}.")
        return best_k, scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            means          = self.means_.astype(np.float32),
            covariances    = self.covariances_.astype(np.float32),
            weights        = self.weights_.astype(np.float32),
            n_components   = np.array([self.n_components]),
            ridge          = np.array([self.ridge]),
            covariance_type = np.array([self.covariance_type], dtype=object),
        )
        print(f"[GMMClustering] Saved → {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)

        self.n_components    = int(data["n_components"][0])
        self.ridge           = float(data["ridge"][0])
        self.covariance_type = str(data["covariance_type"][0])
        self.means_          = data["means"].astype(np.float64)
        self.covariances_    = data["covariances"].astype(np.float64)
        self.weights_        = data["weights"].astype(np.float64)

        C         = self.means_.shape[1]
        ridge_mat = self.ridge * np.eye(C)

        # Reconstruct a sklearn GMM for predict/predict_proba
        self._reconstruct_gmm()
        self._fitted = True
        print(f"[GMMClustering] Loaded {self.n_components} components ← {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_covariances(gmm) -> np.ndarray:
        """
        Expand sklearn's compressed covariance representation to full
        [K, C, C] matrices regardless of covariance_type.
        """
        K, C = gmm.means_.shape
        full = np.zeros((K, C, C))
        if gmm.covariance_type == "full":
            full = gmm.covariances_.copy()
        elif gmm.covariance_type == "diag":
            for k in range(K):
                full[k] = np.diag(gmm.covariances_[k])
        elif gmm.covariance_type == "spherical":
            for k in range(K):
                full[k] = np.eye(C) * gmm.covariances_[k]
        elif gmm.covariance_type == "tied":
            for k in range(K):
                full[k] = gmm.covariances_.copy()
        return full

    def _reconstruct_gmm(self):
        """Reconstruct a minimal sklearn GaussianMixture for prediction only."""
        from sklearn.mixture import GaussianMixture
        self._gmm = GaussianMixture(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
        )
        # Manually inject fitted parameters
        self._gmm.means_              = self.means_
        self._gmm.weights_            = self.weights_
        self._gmm.covariances_        = self.covariances_
        # sklearn needs precisions_chol for predict; refit is cleaner
        # We do a dummy fit on the means themselves to initialise internals,
        # then overwrite.  Not elegant but avoids reimplementing sklearn internals.
        self._gmm.fit(self.means_)
        self._gmm.means_      = self.means_
        self._gmm.weights_    = self.weights_
        self._gmm.covariances_ = self.covariances_
        # Recompute precision Cholesky as sklearn does internally
        from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
        self._gmm.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("GMMClustering has not been fitted. Call fit() first.")
