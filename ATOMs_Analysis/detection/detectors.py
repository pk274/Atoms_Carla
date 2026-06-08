"""
detection/detectors.py
----------------------
Common detector interface and concrete implementations for ATOMs-based
anomaly / OOD detection.

All detectors share the same BaseDetector interface:
    fit(data)        build internal baseline from a [N, D] feature matrix
    score(x)         return a scalar anomaly score for a single sample [D]
    save/load        persist to / restore from disk

Concrete detectors
------------------
MahalanobisDetector
    Works on any fixed-size feature vector (ATOMs attention profiles,
    deep backbone features, …).  The same class is reused for both;
    what differs is only the feature extractor that produces the input.

EuclideanDetector
    L2 distance from the training mean.  Fast, parameter-free baseline.

KNNDetector
    Mean L2 distance to the k nearest training neighbours.
    Non-parametric; captures multi-modal in-distribution clusters.
    Default k=50 (configurable).

WassersteinDetector
    Sliced Wasserstein-1 distance: averages 1-D W1 over random unit-vector
    projections.  In 1-D, W1 between a point mass and an empirical
    distribution reduces to the mean absolute deviation of training samples
    from the test point.  Distribution-free and rotation-invariant.

JensenShannonDetector
    Sliced Jensen-Shannon divergence: projects to random 1-D directions,
    represents training data as a KDE and the test point as a narrow
    Gaussian, then computes JS divergence numerically on a grid.
    Symmetric and bounded in [0, log2].

MDXDetector
    Mahalanobis Distance-based (MDX) detector from Zhang et al. (2024).
    Operates on penultimate-layer features reduced to 50 PCA dimensions.
    Builds class-conditional Gaussian distributions (μ_c, Σ_c) for each
    discretised action class.  Scores via the minimum Mahalanobis distance
    to any class centroid (Eq. 5 in the paper).  Action space is discretised
    into 3 steer × 2 throt × 2 brake = 12 classes.

ActionEntropyDetector
    Model-free baseline.  Treats high entropy of the action distribution
    as a proxy for uncertainty / anomaly.  Requires no baseline fitting.

FeatureExtractor
    Utility that attaches a PyTorch forward hook to a named layer of the
    policy network and captures the penultimate-layer activation vector
    needed by MDXDetector.

DetectorEvaluator
    Given a labeled test set (scores + binary labels), computes ROC curve,
    AUC, and optimal threshold via Youden's J statistic.  Produces a
    results dict that can be saved as JSON for later comparison.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ATOMs_Analysis.utils.distance_computer import DistanceComputer

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """
    Common interface for all anomaly detectors.

    Subclasses must implement fit(), score(), save(), load().
    """

    def __init__(self):
        self._fitted: bool = False

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Build internal baseline statistics from a feature matrix.

        Parameters
        ----------
        data : np.ndarray [N, D]  — in-distribution feature vectors.
        """

    @abstractmethod
    def score(self, x: np.ndarray) -> float:
        """
        Return a scalar anomaly score for a single feature vector.
        Higher score = more anomalous.

        Parameters
        ----------
        x : np.ndarray [D]
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist fitted parameters to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore fitted parameters from disk."""

    def score_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Score a batch of feature vectors.

        Parameters
        ----------
        data : np.ndarray [N, D]

        Returns
        -------
        np.ndarray [N]  — anomaly scores
        """
        return np.array([self.score(row) for row in data])

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call fit() first."
            )


# ---------------------------------------------------------------------------
# Mahalanobis detector
# ---------------------------------------------------------------------------

class MahalanobisDetector(BaseDetector):
    """
    Anomaly detection via Mahalanobis distance from a fitted Gaussian baseline.

    Works on any fixed-size feature vector.  Intended uses:
      - ATOMs attention profiles  [num_classes]
      - Backbone feature vectors  [512]  (deep Mahalanobis baseline)
      - FC layer activations      [256]

    Parameters
    ----------
    ridge : float
        Regularisation added to the diagonal of the covariance matrix before
        inversion.  Prevents singularity for features that are near-constant
        across the baseline (e.g. rare CARLA classes present in <5% of frames).
        Default 1e-6 is conservative; increase to 1e-4 if inversion is unstable.
    """

    def __init__(self, ridge: float = 1e-6):
        super().__init__()
        self.ridge = ridge
        self.mean:      Optional[np.ndarray] = None   # [D]
        self.cov:       Optional[np.ndarray] = None   # [D, D]
        self._precision: Optional[np.ndarray] = None  # [D, D]  pre-inverted

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray) -> None:
        """
        Compute mean and covariance from baseline feature matrix.

        Parameters
        ----------
        data : np.ndarray [N, D]
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array [N, D], got shape {data.shape}")

        self.mean = data.mean(axis=0)
        self.cov  = np.cov(data.T)

        # Scalar case (D=1): np.cov returns a scalar
        if self.cov.ndim == 0:
            self.cov = np.array([[float(self.cov)]])

        ridge_mat = self.ridge * np.eye(len(self.mean))
        self._precision = np.linalg.inv(self.cov + ridge_mat)
        self._fitted = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, x: np.ndarray) -> float:
        self._check_fitted()
        # compute_mahalanobis already returns the Mahalanobis DISTANCE (it takes
        # the sqrt internally).  (Fixed 2026-06-08, docs/code_review.md §3.1: this
        # previously applied a second sqrt — returning sqrt(distance) — which put
        # the single-Gaussian scores on a different scale from the GMM path.)
        return float(DistanceComputer.compute_mahalanobis(self.mean, self.cov, x, self.ridge))

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def fit_threshold(
        self,
        baseline_data: np.ndarray,
        percentile: float = 99.0,
    ) -> float:
        """
        Set the anomaly threshold from in-distribution scores.

        Parameters
        ----------
        baseline_data : np.ndarray [N, D] — same data used for fit(), or a
                        held-out in-distribution validation set.
        percentile    : float in (0, 100).  99 → flag top 1% as anomalous.

        Returns
        -------
        float — threshold (also stored as self.threshold)
        """
        self._check_fitted()
        scores = self.score_batch(baseline_data)
        self.threshold = float(np.percentile(scores, percentile))
        return self.threshold

    def is_anomalous(self, x: np.ndarray) -> Tuple[bool, float]:
        """Return (is_anomalous, score). Requires threshold to be set."""
        if not hasattr(self, "threshold") or self.threshold is None:
            raise RuntimeError("Call fit_threshold() first.")
        d = self.score(x)
        return d > self.threshold, d

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean      = self.mean.astype(np.float32),
            cov       = self.cov.astype(np.float32),
            ridge     = np.array([self.ridge]),
            threshold = np.array([getattr(self, "threshold", np.nan)]),
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        self.mean      = data["mean"].astype(np.float64)
        self.cov       = data["cov"].astype(np.float64)
        self.ridge     = float(data["ridge"][0])
        threshold_val  = float(data["threshold"][0])
        self.threshold = None if np.isnan(threshold_val) else threshold_val
        ridge_mat      = self.ridge * np.eye(len(self.mean))
        self._precision = np.linalg.inv(self.cov + ridge_mat)
        self._fitted   = True


# ---------------------------------------------------------------------------
# Action entropy detector
# ---------------------------------------------------------------------------

class ActionEntropyDetector(BaseDetector):
    """
    Model-free OOD baseline: high entropy of the action distribution
    signals that the agent is uncertain, which correlates with anomalous input.

    No baseline fitting is required; fit() is a no-op provided for interface
    compatibility.

    The input to score() is the raw action logit vector [num_acts] or a
    softmax probability vector.  Both are handled automatically.

    Parameters
    ----------
    from_logits : bool
        If True (default), input to score() is logits and softmax is applied
        internally.  Set False if you pass probabilities directly.
    cmd : int or None
        If set, only the logits for this navigation command are used.
        If None, the full logit vector is used.
    num_steers : int
    num_throts : int
        Required if cmd is set, to slice the correct logit range.
    """

    def __init__(
        self,
        from_logits: bool = True,
        cmd: Optional[int] = None,
        num_steers: int = 0,
        num_throts: int = 0,
    ):
        super().__init__()
        self.from_logits = from_logits
        self.cmd         = cmd
        self.num_steers  = num_steers
        self.num_throts  = num_throts
        self._fitted     = True   # no fitting needed

    def fit(self, data: np.ndarray) -> None:
        """No-op — entropy detector requires no baseline."""
        self._fitted = True

    def score(self, x: np.ndarray) -> float:
        """
        Return Shannon entropy of the (softmaxed) action distribution.

        Parameters
        ----------
        x : np.ndarray [num_acts] — raw logits or probabilities.

        Returns
        -------
        float — entropy in nats.  Higher = more uncertain = more anomalous.
        """
        x = np.asarray(x, dtype=np.float64)

        # Slice to the active command if specified
        if self.cmd is not None and self.num_steers > 0:
            stride = self.num_steers + self.num_throts + 1
            start  = self.cmd * stride
            x = x[start : start + stride]

        if self.from_logits:
            x = x - x.max()   # numerical stability
            exp_x = np.exp(x)
            probs = exp_x / exp_x.sum()
        else:
            probs = x / (x.sum() + 1e-12)

        # Shannon entropy: H = -sum(p log p)
        probs = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "from_logits": self.from_logits,
            "cmd":         self.cmd,
            "num_steers":  self.num_steers,
            "num_throts":  self.num_throts,
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(Path(path)) as f:
            meta = json.load(f)
        self.from_logits = meta["from_logits"]
        self.cmd         = meta["cmd"]
        self.num_steers  = meta["num_steers"]
        self.num_throts  = meta["num_throts"]
        self._fitted     = True


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DetectorEvaluator:
    """
    Evaluates any BaseDetector on a labeled test set.

    Computes ROC curve, AUC, and the optimal threshold via Youden's J
    statistic (maximises sensitivity + specificity − 1).

    Usage
    -----
    evaluator = DetectorEvaluator()
    results   = evaluator.evaluate(detector, scores, labels)
    evaluator.save_results(results, "results/atoms_mahal.json")
    """

    def evaluate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        detector_name: str = "detector",
    ) -> Dict:
        """
        Parameters
        ----------
        scores : np.ndarray [N]  — anomaly scores (higher = more anomalous).
        labels : np.ndarray [N]  — binary ground truth (1 = anomalous / perturbed,
                                   0 = clean).
        detector_name : str      — used only for labeling the results dict.

        Returns
        -------
        dict with keys:
            auc, optimal_threshold, tpr_at_threshold, fpr_at_threshold,
            youden_j, roc_fpr [N+1], roc_tpr [N+1], detector_name
        """
        from sklearn.metrics import roc_curve, roc_auc_score

        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)

        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Youden's J = TPR - FPR, maximised at optimal operating point
        j = tpr - fpr
        best_idx = int(np.argmax(j))

        return {
            "detector_name":      detector_name,
            "auc":                auc,
            "optimal_threshold":  float(thresholds[best_idx]),
            "tpr_at_threshold":   float(tpr[best_idx]),
            "fpr_at_threshold":   float(fpr[best_idx]),
            "youden_j":           float(j[best_idx]),
            "roc_fpr":            fpr.tolist(),
            "roc_tpr":            tpr.tolist(),
            "n_samples":          int(len(labels)),
            "n_anomalous":        int(labels.sum()),
        }

    def evaluate_from_detector(
        self,
        detector: BaseDetector,
        features: np.ndarray,
        labels: np.ndarray,
        detector_name: str = "detector",
    ) -> Dict:
        """
        Convenience wrapper: score all features with detector then evaluate.

        Parameters
        ----------
        features : np.ndarray [N, D]
        labels   : np.ndarray [N]
        """
        scores = detector.score_batch(features)
        return self.evaluate(scores, labels, detector_name)

    def save_results(self, results: Dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[DetectorEvaluator] Results saved → {path}")

    def compare(self, results_list: List[Dict]) -> None:
        """Print a comparison table for multiple detectors."""
        header = f"{'Detector':<30} {'AUC':>6} {'Youden J':>9} {'TPR':>6} {'FPR':>6}"
        print(header)
        print("-" * len(header))
        for r in sorted(results_list, key=lambda x: -x["auc"]):
            print(
                f"{r['detector_name']:<30} "
                f"{r['auc']:>6.3f} "
                f"{r['youden_j']:>9.3f} "
                f"{r['tpr_at_threshold']:>6.3f} "
                f"{r['fpr_at_threshold']:>6.3f}"
            )


# ---------------------------------------------------------------------------
# Euclidean distance detector
# ---------------------------------------------------------------------------

class EuclideanDetector(BaseDetector):
    """
    Anomaly detection via L2 distance from the training mean.

    The simplest possible baseline: no covariance, no neighbours, no
    distributional assumptions.  Useful for sanity-checking that the
    richer detectors do actually add value.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__()
        self.mean: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array [N, D], got shape {data.shape}")
        self.mean = data.mean(axis=0)
        self._fitted = True

    def score(self, x: np.ndarray) -> float:
        self._check_fitted()
        return float(np.linalg.norm(np.asarray(x, dtype=np.float64) - self.mean))

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self.mean.astype(np.float32))

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path))
        self.mean = data["mean"].astype(np.float64)
        self._fitted = True


# ---------------------------------------------------------------------------
# K-nearest-neighbours detector
# ---------------------------------------------------------------------------

class KNNDetector(BaseDetector):
    """
    Anomaly detection via mean L2 distance to the k nearest training neighbours.

    Non-parametric and naturally multi-modal: if in-distribution data forms
    several clusters, the detector scores well for any cluster rather than
    penalising points that are far from a single global mean.

    Internally uses sklearn's BallTree for O(N log N) queries.

    Parameters
    ----------
    k : int
        Number of neighbours to average over.  Default 50 matches the value
        that worked well in prior experiments.  Larger k = smoother, more
        global score; smaller k = more sensitive to local density.
    """

    def __init__(self, k: int = 50):
        super().__init__()
        self.k = k
        self._train_data: Optional[np.ndarray] = None
        self._tree = None

    def fit(self, data: np.ndarray) -> None:
        from sklearn.neighbors import BallTree
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array [N, D], got shape {data.shape}")
        if len(data) < self.k:
            raise ValueError(
                f"KNNDetector requires at least k={self.k} training samples, "
                f"got {len(data)}."
            )
        self._train_data = data
        self._tree = BallTree(data, metric="euclidean")
        self._fitted = True

    def score(self, x: np.ndarray) -> float:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        dists, _ = self._tree.query(x, k=self.k)
        return float(dists.mean())

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_data=self._train_data.astype(np.float32),
            k=np.array([self.k]),
        )

    def load(self, path: str | Path) -> None:
        from sklearn.neighbors import BallTree
        data = np.load(Path(path))
        self._train_data = data["train_data"].astype(np.float64)
        self.k = int(data["k"][0])
        self._tree = BallTree(self._train_data, metric="euclidean")
        self._fitted = True


# ---------------------------------------------------------------------------
# Sliced Wasserstein detector
# ---------------------------------------------------------------------------

class WassersteinDetector(BaseDetector):
    """
    Anomaly scoring via sliced Wasserstein-1 distance.

    Projects both training data and the test point onto n_projections random
    unit vectors.  Along each 1-D projection, W1 between a point mass at
    the test value and the empirical training distribution equals the mean
    absolute deviation of training projections from the test projection:

        W1(δ_x, P_train) = (1/N) Σ_i |v^T x - v^T x_i|

    The final score is the average over all projections, which approximates
    the sliced Wasserstein distance between the test point and the training
    distribution.  This is rotation-invariant and distribution-free.

    Parameters
    ----------
    n_projections : int
        Number of random 1-D projections.  200 gives stable estimates for
        feature vectors up to ~512 dimensions.
    random_state : int
        Seed for the random projection directions (for reproducibility).
    """

    def __init__(self, n_projections: int = 200, random_state: int = 42):
        super().__init__()
        self.n_projections = n_projections
        self.random_state = random_state
        self._projections: Optional[np.ndarray] = None   # [n_proj, D]
        self._proj_train:  Optional[np.ndarray] = None   # [n_proj, N]

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array [N, D], got shape {data.shape}")
        rng = np.random.RandomState(self.random_state)
        D = data.shape[1]
        proj = rng.randn(self.n_projections, D)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True)  # unit vectors
        self._projections = proj
        self._proj_train  = proj @ data.T   # [n_proj, N]
        self._fitted = True

    def score(self, x: np.ndarray) -> float:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        x_proj = self._projections @ x          # [n_proj]
        # 1-D W1: mean absolute deviation of training projections from test value
        w1_per_proj = np.abs(
            self._proj_train - x_proj[:, None]  # [n_proj, N]
        ).mean(axis=1)                           # [n_proj]
        return float(w1_per_proj.mean())

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            projections   = self._projections.astype(np.float32),
            proj_train    = self._proj_train.astype(np.float32),
            n_projections = np.array([self.n_projections]),
            random_state  = np.array([self.random_state]),
        )

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path))
        self._projections = data["projections"].astype(np.float64)
        self._proj_train  = data["proj_train"].astype(np.float64)
        self.n_projections = int(data["n_projections"][0])
        self.random_state  = int(data["random_state"][0])
        self._fitted = True


# ---------------------------------------------------------------------------
# Sliced Jensen-Shannon divergence detector
# ---------------------------------------------------------------------------

class JensenShannonDetector(BaseDetector):
    """
    Anomaly scoring via sliced Jensen-Shannon divergence.

    Uses the same random-projection scheme as WassersteinDetector.  Along
    each 1-D projection the training distribution is represented as a KDE
    (Gaussian kernel with Scott's rule bandwidth) and the test point as a
    narrow Gaussian (σ = bandwidth / 4).  JS divergence is computed
    numerically on a fine grid and averaged over all projections.

    JS divergence is bounded in [0, ln 2] ≈ [0, 0.693] (using natural log).
    Higher = more anomalous.

    Parameters
    ----------
    n_projections : int
        Number of random 1-D projections.  Fewer than WassersteinDetector
        because each projection is more expensive to evaluate.  50 is a
        practical default; increase to 100–200 for higher accuracy.
    n_grid : int
        Number of grid points for numerical integration.  1000 is accurate
        enough for smooth KDEs.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_projections: int = 50,
        n_grid: int = 1000,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_projections = n_projections
        self.n_grid        = n_grid
        self.random_state  = random_state
        self._projections: Optional[np.ndarray] = None   # [n_proj, D]
        self._proj_train:  Optional[np.ndarray] = None   # [n_proj, N]
        self._bandwidths:  Optional[np.ndarray] = None   # [n_proj]

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array [N, D], got shape {data.shape}")
        rng = np.random.RandomState(self.random_state)
        N, D = data.shape
        proj = rng.randn(self.n_projections, D)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True)
        proj_train = proj @ data.T   # [n_proj, N]
        # Scott's rule bandwidth: h = N^(-1/5) * std  (per projection)
        bandwidths = N ** (-0.2) * proj_train.std(axis=1)  # [n_proj]
        self._projections = proj
        self._proj_train  = proj_train
        self._bandwidths  = np.maximum(bandwidths, 1e-8)   # avoid zero bw
        self._fitted = True

    @staticmethod
    def _kde_1d(grid: np.ndarray, samples: np.ndarray, bw: float) -> np.ndarray:
        """Evaluate Gaussian KDE at grid points."""
        # [grid, samples] pairwise differences
        z = (grid[:, None] - samples[None, :]) / bw
        density = np.exp(-0.5 * z ** 2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
        return density

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        JS divergence between two non-negative arrays (treated as unnormalised
        densities on a uniform grid).  Returns value in [0, ln 2].
        """
        p = p / (p.sum() + 1e-300)
        q = q / (q.sum() + 1e-300)
        m = 0.5 * (p + q)
        eps = 1e-300
        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
        return float(0.5 * (kl_pm + kl_qm))

    def score(self, x: np.ndarray) -> float:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        jsd_scores = []
        for i in range(self.n_projections):
            bw      = self._bandwidths[i]
            samples = self._proj_train[i]          # [N]
            x_proj  = float(self._projections[i] @ x)

            lo = min(samples.min(), x_proj) - 4 * bw
            hi = max(samples.max(), x_proj) + 4 * bw
            grid = np.linspace(lo, hi, self.n_grid)

            # Training distribution: KDE
            p_train = self._kde_1d(grid, samples, bw)
            # Test point: very narrow Gaussian (σ = bw/4)
            sigma_test = bw / 4.0
            p_test  = np.exp(-0.5 * ((grid - x_proj) / sigma_test) ** 2)
            p_test /= p_test.sum() + 1e-300

            jsd_scores.append(self._js_divergence(p_train, p_test))

        return float(np.mean(jsd_scores))

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            projections   = self._projections.astype(np.float32),
            proj_train    = self._proj_train.astype(np.float32),
            bandwidths    = self._bandwidths.astype(np.float32),
            n_projections = np.array([self.n_projections]),
            n_grid        = np.array([self.n_grid]),
            random_state  = np.array([self.random_state]),
        )

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path))
        self._projections = data["projections"].astype(np.float64)
        self._proj_train  = data["proj_train"].astype(np.float64)
        self._bandwidths  = data["bandwidths"].astype(np.float64)
        self.n_projections = int(data["n_projections"][0])
        self.n_grid        = int(data["n_grid"][0])
        self.random_state  = int(data["random_state"][0])
        self._fitted = True


# ---------------------------------------------------------------------------
# MDX detector  (Zhang et al., 2024 — Transactions on ML Research)
# ---------------------------------------------------------------------------

class MDXDetector(BaseDetector):
    """
    Mahalanobis Distance-based (MDX) detector for deep RL.

    Implements the vanilla MD variant from Zhang et al. (2024) with the
    following pipeline:

        1.  Reduce input features to `n_pca_components` dimensions with PCA
            (paper uses 50).
        2.  Discretise the continuous policy outputs into action classes:
                steer  → n_steer_bins  bins  (default 3: left / straight / right)
                throt  → n_throt_bins  bins  (default 2: low / high)
                brake  → n_brake_bins  bins  (default 2: off / on)
            giving n_steer × n_throt × n_brake = 12 classes by default.
        3.  Estimate per-class mean μ_c and covariance Σ_c from the baseline.
        4.  Score a new state as the minimum Mahalanobis distance to any class
            centroid (Eq. 5 in the paper):
                M(s) = min_c  (f(s) − μ_c)^T  Σ_c^{−1}  (f(s) − μ_c)

    The raw score is the squared Mahalanobis distance (Chi-squared distributed
    under the Gaussian assumption); under Proposition 1 a threshold can be
    derived from χ²_p(1 − α) if desired.

    Note on action-space size
    -------------------------
    3 steer × 2 throt × 2 brake = **12** classes (not 20).

    Parameters
    ----------
    n_pca_components : int
        PCA output dimension.  Paper uses 50.
    n_steer_bins : int
        Number of steering bins.  Default 3 (left / straight / right).
    n_throt_bins : int
        Number of throttle bins.  Default 2 (low / high).
    n_brake_bins : int
        Number of brake bins.  Default 2 (off / on).
    ridge : float
        Regularisation for per-class covariance inversion.
    alpha : float
        Significance level for the chi-squared threshold (unused in score(),
        but available via threshold property for hard decisions).
    """

    def __init__(
        self,
        n_pca_components:  int   = 50,
        n_steer_bins:      int   = 3,
        n_throt_bins:      int   = 2,
        n_brake_bins:      int   = 2,
        ridge:             float = 1e-6,
        alpha:             float = 0.05,
        calibration_split: float = 0.2,
        bin_strategy:      str   = "equal-width",
    ):
        super().__init__()
        self.n_pca_components = n_pca_components
        self.n_steer_bins     = n_steer_bins
        self.n_throt_bins     = n_throt_bins
        self.n_brake_bins     = n_brake_bins
        self.ridge            = ridge
        self.alpha            = alpha

        self.calibration_split        = calibration_split
        self._conformal_threshold: Optional[float] = None

        self.n_classes:    int = n_steer_bins * n_throt_bins * n_brake_bins
        self.bin_strategy: str = bin_strategy

        self._pca:        Optional[object]              = None   # sklearn PCA
        self._class_means: Optional[Dict[int, np.ndarray]] = None
        self._class_precs: Optional[Dict[int, np.ndarray]] = None

        # Bin edges — set during fit() from training data ranges
        self._steer_edges: Optional[np.ndarray] = None
        self._throt_edges: Optional[np.ndarray] = None
        self._brake_edges: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Action discretisation
    # ------------------------------------------------------------------

    def _build_bin_edges(
        self,
        steers: np.ndarray,
        throts: np.ndarray,
        brakes: np.ndarray,
    ) -> None:
        """Derive bin edges from the range of training actions.

        With bin_strategy="equal-width" (default, preserves original behaviour):
            edges are evenly spaced between min and max.
        With bin_strategy="quantile":
            edges are placed at quantiles of the empirical distribution, so each
            bin contains approximately the same number of training samples.
            Constant dimensions (e.g. steer always 0) collapse to a single bin.
        """
        def _edges(v: np.ndarray, nb: int) -> np.ndarray:
            v = np.asarray(v, float)
            if self.bin_strategy == "quantile":
                e = np.unique(np.quantile(v, np.linspace(0.0, 1.0, nb + 1)))
                if e.size < 2:              # constant dimension — single bin
                    e = np.array([v.min() - 1e-6, v.max() + 1e-6])
            else:                           # "equal-width" — legacy default
                e = np.linspace(v.min(), v.max(), nb + 1)
            e[0] -= 1e-6; e[-1] += 1e-6
            return e

        self._steer_edges = _edges(steers, self.n_steer_bins)
        self._throt_edges = _edges(throts, self.n_throt_bins)
        self._brake_edges = _edges(brakes, self.n_brake_bins)

    def discretise_action(
        self,
        steer_logit: float,
        throt_logit: float,
        brake_logit: float,
    ) -> int:
        """
        Convert continuous policy logits to an integer action class index.

        Returns an int in [0, n_classes).  Requires fit() to have been called
        (bin edges are derived from training data).

        The index encodes: class = steer_bin * (n_throt * n_brake)
                                  + throt_bin * n_brake
                                  + brake_bin
        """
        if self._steer_edges is None:
            raise RuntimeError("Call fit() before discretise_action().")
        sb = int(np.clip(np.digitize(steer_logit, self._steer_edges) - 1,
                         0, self.n_steer_bins - 1))
        tb = int(np.clip(np.digitize(throt_logit, self._throt_edges) - 1,
                         0, self.n_throt_bins - 1))
        bb = int(np.clip(np.digitize(brake_logit, self._brake_edges) - 1,
                         0, self.n_brake_bins - 1))
        return sb * (self.n_throt_bins * self.n_brake_bins) + tb * self.n_brake_bins + bb

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        data:    np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """
        Fit the MDX detector.

        Parameters
        ----------
        data : np.ndarray [N, D]
            Penultimate-layer feature vectors for baseline (clean) episodes.
            Obtain these via FeatureExtractor (see below).
        actions : np.ndarray [N, 3]
            Corresponding continuous policy outputs, columns ordered as
            [steer_logit, throt_logit, brake_logit].

        Notes
        -----
        PCA is fit on the full data matrix before class splitting, matching
        the paper's procedure.  Class-conditional statistics are then
        estimated in PCA space.
        """
        from sklearn.decomposition import PCA

        data    = np.asarray(data,    dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected data shape [N, D], got {data.shape}")
        if actions.ndim != 2 or actions.shape[1] != 3:
            raise ValueError(
                f"Expected actions shape [N, 3] (steer, throt, brake), "
                f"got {actions.shape}"
            )
        if len(data) != len(actions):
            raise ValueError("data and actions must have the same length.")
        
        # --- train / calibration split -------------------------------------------
        n_total   = len(data)
        n_cal     = max(1, int(n_total * self.calibration_split))
        rng       = np.random.default_rng(seed=0)   # fix seed for reproducibility
        perm      = rng.permutation(n_total)
        cal_idx   = perm[:n_cal]
        train_idx = perm[n_cal:]

        data_cal    = data[cal_idx]       # held out — never seen by PCA or covariance estimator
        data        = data[train_idx]     # shadow the variable so the rest of fit() is unchanged
        actions     = actions[train_idx]
        # -------------------------------------------------------------------------

        n_comp = min(self.n_pca_components, data.shape[1], data.shape[0])
        if n_comp < self.n_pca_components:
            print(
                f"[MDXDetector] Warning: n_pca_components reduced from "
                f"{self.n_pca_components} to {n_comp} (insufficient data/features)."
            )
            self.n_pca_components = n_comp

        # Step 1: PCA
        self._pca = PCA(n_components=self.n_pca_components)
        data_pca  = self._pca.fit_transform(data)   # [N, n_pca]

        # Step 2: bin edges from training action range
        self._build_bin_edges(actions[:, 0], actions[:, 1], actions[:, 2])

        # Step 3: discretise all training actions
        class_labels = np.array([
            self.discretise_action(float(a[0]), float(a[1]), float(a[2]))
            for a in actions
        ])

        # Step 4: per-class mean and precision
        self._class_means = {}
        self._class_precs = {}
        for c in range(self.n_classes):
            mask = class_labels == c
            n_c  = mask.sum()
            if n_c < 2:
                # Not enough samples for this class — skip (will be ignored in score)
                continue
            X_c       = data_pca[mask]
            mu_c      = X_c.mean(axis=0)
            cov_c     = np.cov(X_c.T) + self.ridge * np.eye(self.n_pca_components)
            if cov_c.ndim == 0:   # single-feature edge case
                cov_c = np.array([[float(cov_c)]])
            self._class_means[c] = mu_c
            self._class_precs[c] = np.linalg.inv(cov_c)

        if not self._class_means:
            raise RuntimeError("No action class had ≥2 training samples.")

        self._fitted = True

        # --- conformal threshold --------------------------------------------------
        cal_scores = np.array([self.score(x) for x in data_cal])
        n_cal      = len(cal_scores)
        # Standard finite-sample adjustment: use the ceil((1-α)(n+1))/n quantile.
        level      = min(1.0, (1.0 - self.alpha) * (n_cal + 1) / n_cal)
        self._conformal_threshold = float(np.quantile(cal_scores, level))
        print(
            f"[MDXDetector] Conformal threshold (α={self.alpha}): "
            f"{self._conformal_threshold:.4f}  "
            f"(chi2 would be {self.chi2_threshold:.4f})"
        )
        # -------------------------------------------------------------------------

        print(
            f"[MDXDetector] Fitted on {len(data)} samples | "
            f"{len(self._class_means)}/{self.n_classes} classes populated | "
            f"PCA: {data.shape[1]}D → {self.n_pca_components}D"
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, x: np.ndarray) -> float:
        """
        Return the minimum squared Mahalanobis distance to any class centroid
        (Eq. 5 in Zhang et al., 2024).

        Parameters
        ----------
        x : np.ndarray [D]  — raw (pre-PCA) feature vector.

        Returns
        -------
        float — M(s) = min_c (f(s)−μ_c)^T Σ_c^{−1} (f(s)−μ_c).
                Chi-squared distributed under the Gaussian assumption.
                Higher = more anomalous.
        """
        self._check_fitted()
        x_pca = self._pca.transform(
            np.asarray(x, dtype=np.float64).reshape(1, -1)
        )[0]
        min_dist = np.inf
        for mu, prec in zip(
            self._class_means.values(), self._class_precs.values()
        ):
            diff = x_pca - mu
            d2   = float(diff @ prec @ diff)
            if d2 < min_dist:
                min_dist = d2
        return min_dist

    @property
    def chi2_threshold(self) -> float:
        """
        Chi-squared threshold at significance level self.alpha
        (Proposition 1 in the paper).  Use as a hard-decision boundary:
            is_anomalous = score(x) > detector.chi2_threshold
        """
        from scipy.stats import chi2
        return float(chi2.ppf(1.0 - self.alpha, df=self.n_pca_components))
    
    @property
    def conformal_threshold(self) -> float:
        """
        Conformal threshold calibrated on the held-out split of the baseline data.
        Preferred over chi2_threshold when you cannot assume a Gaussian feature
        distribution.  Use as:
            is_anomalous = score(x) > detector.conformal_threshold
        """
        if self._conformal_threshold is None:
            raise RuntimeError("Call fit() first.")
        return self._conformal_threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import pickle
        self._check_fitted()
        path = Path(path).with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[MDXDetector] Saved → {path}")

    def load(self, path: str | Path) -> None:
        import pickle
        path = Path(path).with_suffix(".pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)

