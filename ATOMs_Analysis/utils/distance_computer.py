import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DistanceResult:
    """Container for distance computation results."""
    distance: float
    distance_type: str  # "mahalanobis", "euclidean", "knn", etc.

    # GMM-specific
    nearest_component: Optional[int] = None
    component_distances: Optional[List[float]] = None
    component_probabilities: Optional[List[float]] = None

    # KNN-specific
    k_value: Optional[int] = None


class DistanceComputer:
    """
    Core distance computation logic.
    Stateless – all parameters are passed explicitly.
    """

    @staticmethod
    def compute_mahalanobis(
        mu_ref: np.ndarray,
        cov_ref: np.ndarray,
        mu_target: np.ndarray,
        regularization: float = 0.01
    ) -> float:
        """
        Compute Mahalanobis distance.

        Parameters
        ----------
        mu_ref : np.ndarray [K]
            Reference mean
        cov_ref : np.ndarray [K, K]
            Reference covariance
        mu_target : np.ndarray [K]
            Target point
        regularization : float
            Regularization term for numerical stability

        Returns
        -------
        float
            Mahalanobis distance
        """
        diff = mu_target - mu_ref
        cov_reg = cov_ref + regularization * np.eye(cov_ref.shape[0])
        cov_inv = np.linalg.pinv(cov_reg)
        mahal_sq = float(diff.T @ cov_inv @ diff)

        if np.isnan(mahal_sq) or np.isinf(mahal_sq) or mahal_sq < 0:
            mahal_sq = max(0.0, np.nan_to_num(mahal_sq, nan=0.0, posinf=1e6))

        distance = float(np.sqrt(mahal_sq))
        if not np.isfinite(distance):
            distance = 1e6
        return distance

    @staticmethod
    def compute_euclidean(
        mu_ref: np.ndarray,
        mu_target: np.ndarray
    ) -> float:
        """
        Compute Euclidean distance.

        Parameters
        ----------
        mu_ref : np.ndarray [K]
        mu_target : np.ndarray [K]

        Returns
        -------
        float
        """
        distance = float(np.linalg.norm(mu_target - mu_ref))
        if not np.isfinite(distance):
            distance = 1e6
        return distance

    @staticmethod
    def compute_knn_distance(
        reference_samples: np.ndarray,
        target_point: np.ndarray,
        k: int = 100,
        normalize: bool = False
    ) -> float:
        """
        Compute k-th nearest neighbor distance.

        Based on "Out-of-Distribution Detection with Deep Nearest Neighbors"
        (Sun et al., ICML 2022).

        Parameters
        ----------
        reference_samples : np.ndarray [N, K]
        target_point : np.ndarray [K]
        k : int
        normalize : bool
            L2-normalize before distance computation (recommended: True)

        Returns
        -------
        float
        """
        if reference_samples.ndim != 2:
            raise ValueError(f"reference_samples must be 2D [N, K], got {reference_samples.shape}")
        if target_point.ndim != 1:
            raise ValueError(f"target_point must be 1D [K], got {target_point.shape}")

        N, K = reference_samples.shape
        if target_point.shape[0] != K:
            raise ValueError(f"Dimension mismatch: reference K={K}, target K={target_point.shape[0]}")
        if k > N:
            raise ValueError(f"k={k} exceeds number of reference samples N={N}")

        if normalize:
            ref_n = reference_samples / (np.linalg.norm(reference_samples, axis=1, keepdims=True) + 1e-12)
            tgt_n = target_point / (np.linalg.norm(target_point) + 1e-12)
        else:
            ref_n, tgt_n = reference_samples, target_point

        distances = cdist(tgt_n.reshape(1, -1), ref_n, metric='euclidean')[0]
        knn_distance = float(np.sort(distances)[k - 1])
        if not np.isfinite(knn_distance):
            knn_distance = 1e6
        return knn_distance

    @staticmethod
    def compute_gmm_distance(
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
        mu_target: np.ndarray,
        mode: str = "nearest",
        regularization: float = 1e-6
    ) -> DistanceResult:
        """
        Compute distance to GMM baseline.

        Parameters
        ----------
        means : np.ndarray [n_components, K]
        covariances : np.ndarray [n_components, K, K]
        weights : np.ndarray [n_components]
        mu_target : np.ndarray [K]
        mode : str
            'nearest' or 'weighted'
        regularization : float

        Returns
        -------
        DistanceResult
        """
        n_components = len(means)
        distances = np.array([
            DistanceComputer.compute_mahalanobis(means[i], covariances[i], mu_target, regularization)
            for i in range(n_components)
        ])
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])

        if mode == "nearest":
            return DistanceResult(
                distance=nearest_distance,
                distance_type="mahalanobis_gmm",
                nearest_component=nearest_idx,
                component_distances=distances.tolist()
            )

        elif mode == "weighted":
            log_probs = -0.5 * distances ** 2 + np.log(weights + 1e-10)
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= probs.sum()
            return DistanceResult(
                distance=float(np.sum(probs * distances)),
                distance_type="mahalanobis_gmm_weighted",
                nearest_component=nearest_idx,
                component_distances=distances.tolist(),
                component_probabilities=probs.tolist()
            )

        else:
            raise ValueError(f"Unknown GMM mode: {mode}")

    @staticmethod
    def compute_jsd(
        p: np.ndarray,
        q: np.ndarray,
        base: float = np.e
    ) -> float:
        """
        Compute Jensen-Shannon Divergence between two probability distributions.

        JSD is a symmetrized and smoothed version of KL divergence:
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)

        Parameters
        ----------
        p : np.ndarray [K]
            First probability distribution (should sum to 1)
        q : np.ndarray [K]
            Second probability distribution (should sum to 1)
        base : float
            Logarithm base (np.e for nats, 2 for bits)

        Returns
        -------
        float
            JSD distance in [0, log(2)] for base e, [0, 1] for base 2
        """
        # Normalize to ensure they're valid probability distributions
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)

        # Compute mixture distribution
        m = 0.5 * (p + q)

        # Add small epsilon to avoid log(0)
        eps = 1e-12
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        m_safe = np.clip(m, eps, 1.0)

        # Compute KL divergences
        kl_pm = np.sum(p * np.log(p_safe / m_safe))
        kl_qm = np.sum(q * np.log(q_safe / m_safe))

        # JSD
        jsd = 0.5 * (kl_pm + kl_qm)

        # Convert base if requested
        if base != np.e:
            jsd = jsd / np.log(base)

        # Numerical safety
        jsd = float(jsd)
        if not np.isfinite(jsd) or jsd < 0:
            jsd = 0.0

        return jsd


    @staticmethod
    def compute_gmm_euclidean(
        means: np.ndarray,
        mu_target: np.ndarray,
    ) -> float:
        """
        Min L2 distance from mu_target to any cluster centroid.

        Parameters
        ----------
        means     : np.ndarray [K, D]
        mu_target : np.ndarray [D]

        Returns
        -------
        float — distance to nearest cluster mean.
        """
        dists = np.linalg.norm(means - mu_target[None, :], axis=1)
        return float(dists.min())

    @staticmethod
    def compute_gmm_jsd(
        means: np.ndarray,
        mu_target: np.ndarray,
    ) -> float:
        """
        Min JSD between mu_target and any cluster mean profile.

        Treats each cluster mean as a reference probability distribution
        and computes JSD(mu_target || mu_k) for each cluster k.

        Parameters
        ----------
        means     : np.ndarray [K, D]
        mu_target : np.ndarray [D]

        Returns
        -------
        float — minimum JSD over all cluster means.
        """
        min_jsd = np.inf
        for k in range(len(means)):
            jsd = DistanceComputer.compute_jsd(means[k], mu_target)
            if jsd < min_jsd:
                min_jsd = jsd
        return float(min_jsd)

    @staticmethod
    def compute_gmm_wasserstein(
        means: np.ndarray,
        mu_target: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> float:
        """
        Min Wasserstein-1 distance between mu_target and any cluster mean profile.

        Parameters
        ----------
        means     : np.ndarray [K, D]
        mu_target : np.ndarray [D]
        positions : np.ndarray [D], optional — passed to compute_wasserstein.

        Returns
        -------
        float — minimum W1 over all cluster means.
        """
        min_w = np.inf
        for k in range(len(means)):
            w = DistanceComputer.compute_wasserstein(means[k], mu_target, positions)
            if w < min_w:
                min_w = w
        return float(min_w)

    @staticmethod
    def compute_wasserstein(
        p: np.ndarray,
        q: np.ndarray,
        positions: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Wasserstein distance (Earth Mover's Distance) between distributions.

        For attention distributions over K objects, interprets each distribution
        as weights over object indices [0, 1, ..., K-1].

        Parameters
        ----------
        p : np.ndarray [K]
            First probability distribution
        q : np.ndarray [K]
            Second probability distribution  
        positions : np.ndarray [K], optional
            Positions of the K objects. Defaults to [0, 1, 2, ..., K-1]

        Returns
        -------
        float
            Wasserstein-1 distance
        """
        from scipy.stats import wasserstein_distance

        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        # Normalize
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)

        # Default positions: treat as uniform spacing [0, 1, 2, ...]
        if positions is None:
            positions = np.arange(len(p), dtype=np.float64)

        # scipy's wasserstein_distance expects sample values and weights
        distance = float(wasserstein_distance(positions, positions, p, q))

        if not np.isfinite(distance):
            distance = 1e6

        return distance
