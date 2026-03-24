"""
The paper states that the discrepancy term can be modeled with
other basis functions such as radial basis functions, and all coefficients
can still be solved in one augmented linear-regression system.

This file therefore implements a faithful and practical variant:

    \hat f_H(x) = rho * f_L(x) + delta_RBF(x)

where delta_RBF(x) is expanded by RBF basis functions and solved together
with rho by least squares / ridge-regularized least squares.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np


ArrayLike = np.ndarray
KernelName = Literal["gaussian", "mq", "imq", "thin_plate"]


def lhs(n_samples: int, n_dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simple Latin Hypercube Sampling in [0, 1]^d.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        X[:, j] = (perm + rng.random(n_samples)) / n_samples
    return X


def pairwise_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Euclidean distance matrix between rows of A and rows of B.
    """
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D2 = np.maximum(A2 + B2 - 2.0 * A @ B.T, 0.0)
    return np.sqrt(D2)


def default_epsilon(X: np.ndarray) -> float:
    """
    Heuristic RBF width based on non-zero median pairwise distance.
    """
    D = pairwise_distance(X, X)
    vals = D[np.triu_indices_from(D, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def rbf_kernel(D: np.ndarray, epsilon: float, kind: KernelName = "gaussian") -> np.ndarray:
    """
    Build RBF matrix from distances.
    """
    eps = max(float(epsilon), 1e-12)
    R = D / eps

    if kind == "gaussian":
        return np.exp(-(R ** 2))
    if kind == "mq":
        return np.sqrt(1.0 + R ** 2)
    if kind == "imq":
        return 1.0 / np.sqrt(1.0 + R ** 2)
    if kind == "thin_plate":
        out = np.zeros_like(R)
        mask = R > 0
        out[mask] = (R[mask] ** 2) * np.log(R[mask])
        return out

    raise ValueError(f"Unsupported kernel kind: {kind}")


@dataclass
class FitResult:
    rho: float
    intercept: float
    weights: np.ndarray
    epsilon: float
    train_rmse: float


class CSFRBF:
    """
    Practical reproduction of an RBF-based multifidelity surrogate.

    Model
    -----
        f_H_hat(x) = rho * f_L(x) + b0 + sum_i w_i * phi(||x - c_i|| / eps)

    where:
    - f_L(x): low-fidelity response
    - rho   : scaling coefficient
    - b0    : intercept term
    - c_i   : RBF centers (default: HF training points)
    - phi   : RBF kernel

    This matches the uploaded paper's augmented-regression idea, but uses
    RBF discrepancy instead of polynomial discrepancy.
    """

    def __init__(
        self,
        kernel: KernelName = "gaussian",
        epsilon: Optional[float] = None,
        reg_lambda: float = 1e-10,
        normalize_x: bool = True,
        normalize_y: bool = False,
    ) -> None:
        self.kernel = kernel
        self.epsilon = epsilon
        self.reg_lambda = float(reg_lambda)
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)

        self.is_fitted_ = False

    def _fit_scaler_x(self, X: np.ndarray) -> None:
        self.x_min_ = X.min(axis=0)
        self.x_max_ = X.max(axis=0)
        span = self.x_max_ - self.x_min_
        span[span == 0] = 1.0
        self.x_span_ = span

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.normalize_x:
            return X
        return (X - self.x_min_) / self.x_span_

    def _fit_scaler_y(self, y: np.ndarray) -> None:
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y))
        if self.y_std_ == 0:
            self.y_std_ = 1.0

    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if not self.normalize_y:
            return y
        return (y - self.y_mean_) / self.y_std_

    def _inverse_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if not self.normalize_y:
            return y
        return y * self.y_std_ + self.y_mean_

    def _design_matrix(self, X_query: np.ndarray, lf_values: np.ndarray) -> np.ndarray:
        D = pairwise_distance(X_query, self.centers_)
        Phi = rbf_kernel(D, self.epsilon_, self.kernel)

        cols = [
            lf_values.reshape(-1, 1),           # rho * f_L(x)
            np.ones((X_query.shape[0], 1)),     # intercept
            Phi,                                # RBF discrepancy
        ]
        return np.hstack(cols)

    def fit(
        self,
        X_hf: np.ndarray,
        y_hf: np.ndarray,
        f_low: Callable[[np.ndarray], np.ndarray],
        centers: Optional[np.ndarray] = None,
    ) -> FitResult:
        """
        Train the multifidelity RBF model from high-fidelity samples and a callable LF model.

        Parameters
        ----------
        X_hf : (n, d)
            High-fidelity sample locations.
        y_hf : (n,)
            High-fidelity responses.
        f_low : callable
            Low-fidelity model/function; must accept (m, d) ndarray and return (m,) ndarray.
        centers : (m, d), optional
            RBF centers. Default uses X_hf itself.

        Returns
        -------
        FitResult
        """
        X_hf = np.asarray(X_hf, dtype=float)
        y_hf = np.asarray(y_hf, dtype=float).reshape(-1)

        if X_hf.ndim != 2:
            raise ValueError("X_hf must be a 2D array.")
        if y_hf.ndim != 1 or y_hf.shape[0] != X_hf.shape[0]:
            raise ValueError("y_hf must be a 1D array with the same number of rows as X_hf.")

        self._fit_scaler_x(X_hf)
        Xs = self._transform_x(X_hf)

        if centers is None:
            centers = X_hf
        centers = np.asarray(centers, dtype=float)
        self.centers_raw_ = centers.copy()
        self.centers_ = self._transform_x(centers)

        self.epsilon_ = float(self.epsilon) if self.epsilon is not None else default_epsilon(self.centers_)

        lf_hf = np.asarray(f_low(X_hf), dtype=float).reshape(-1)
        if lf_hf.shape[0] != X_hf.shape[0]:
            raise ValueError("f_low(X_hf) must return a 1D vector with length len(X_hf).")

        self._fit_scaler_y(y_hf)
        y_train = self._transform_y(y_hf)

        # When normalize_y=True, we also normalize LF values into the HF response scale
        if self.normalize_y:
            lf_hf_used = self._transform_y(lf_hf)
        else:
            lf_hf_used = lf_hf

        A = self._design_matrix(Xs, lf_hf_used)

        # Ridge-regularized least squares:
        # theta = (A^T A + lambda I)^(-1) A^T y
        ATA = A.T @ A
        reg = self.reg_lambda * np.eye(ATA.shape[0], dtype=float)
        theta = np.linalg.solve(ATA + reg, A.T @ y_train)

        self.theta_ = theta
        self.rho_ = float(theta[0])
        self.intercept_ = float(theta[1])
        self.weights_ = theta[2:].copy()
        self.f_low_ = f_low

        pred_train = self.predict(X_hf)
        train_rmse = float(np.sqrt(np.mean((pred_train - y_hf) ** 2)))

        self.is_fitted_ = True
        return FitResult(
            rho=self.rho_,
            intercept=self.intercept_,
            weights=self.weights_,
            epsilon=self.epsilon_,
            train_rmse=train_rmse,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict high-fidelity response at X.
        """
        if not hasattr(self, "theta_"):
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        Xs = self._transform_x(X)

        lf = np.asarray(self.f_low_(X), dtype=float).reshape(-1)
        if self.normalize_y:
            lf_used = self._transform_y(lf)
        else:
            lf_used = lf

        A = self._design_matrix(Xs, lf_used)
        y_hat = A @ self.theta_
        return self._inverse_y(y_hat)

    def score_rmse(self, X: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = self.predict(X)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ---------------------------------------------------------------------
# Example: Modified Currin function used in the uploaded LR-MFS paper
# ---------------------------------------------------------------------

def currin_hf(X: np.ndarray) -> np.ndarray:
    """
    High-fidelity modified Currin function from the uploaded paper.
    Domain: x1, x2 in [0, 1]
    """
    X = np.asarray(X, dtype=float)
    x1 = X[:, 0]
    x2 = X[:, 1]

    # avoid division by zero at x2=0 in the exponential term
    x2_safe = np.maximum(x2, 1e-12)

    term1 = 1.0 - np.exp(-1.0 / (2.0 * x2_safe))
    num = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    den = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
    return term1 * num / den


def _currin_hf_point(x1: float, x2: float) -> float:
    return float(currin_hf(np.array([[x1, x2]], dtype=float))[0])


def currin_lf(X: np.ndarray) -> np.ndarray:
    """
    Low-fidelity modified Currin function from the uploaded paper.
    """
    X = np.asarray(X, dtype=float)
    out = []
    for x1, x2 in X:
        val = (
            _currin_hf_point(x1 + 0.05, x2 + 0.05)
            + _currin_hf_point(x1 + 0.05, max(0.0, x2 - 0.05))
            + _currin_hf_point(x1 - 0.05, x2 + 0.05)
            + _currin_hf_point(x1 - 0.05, max(0.0, x2 - 0.05))
        ) / 8.0 + ((-5.0 * x1 - 7.0 * x2) ** 2) / 8.0
        out.append(val)
    return np.asarray(out, dtype=float)


def demo_currin(
    n_hf: int = 10,
    seed: int = 0,
    noise_std: float = 0.0,
    kernel: KernelName = "gaussian",
) -> Tuple[CSFRBF, float]:
    """
    Quick end-to-end demo on the modified Currin function.
    """
    rng = np.random.default_rng(seed)

    X_train = lhs(n_hf, 2, seed=seed)
    y_train = currin_hf(X_train)
    if noise_std > 0:
        y_train = y_train + rng.normal(0.0, noise_std, size=n_hf)

    model = CSFRBF(
        kernel=kernel,
        epsilon=None,
        reg_lambda=1e-8,
        normalize_x=True,
        normalize_y=False,
    )
    result = model.fit(X_train, y_train, currin_lf)

    # 100 x 100 test grid
    grid_1d = np.linspace(0.0, 1.0, 100)
    xx, yy = np.meshgrid(grid_1d, grid_1d)
    X_test = np.column_stack([xx.ravel(), yy.ravel()])

    y_true = currin_hf(X_test)
    rmse = model.score_rmse(X_test, y_true)

    print("=== Demo: CSF-RBF style multifidelity RBF model ===")
    print(f"HF samples       : {n_hf}")
    print(f"Noise STD        : {noise_std}")
    print(f"Kernel           : {kernel}")
    print(f"Estimated rho    : {result.rho:.6f}")
    print(f"Estimated eps    : {result.epsilon:.6f}")
    print(f"Train RMSE       : {result.train_rmse:.6f}")
    print(f"Grid test RMSE   : {rmse:.6f}")

    return model, rmse


if __name__ == "__main__":
    # Example 1: 10 HF samples, no noise
    demo_currin(n_hf=10, seed=42, noise_std=0.0, kernel="gaussian")

    # Example 2: 10 HF samples, noisy HF data
    demo_currin(n_hf=10, seed=42, noise_std=0.2, kernel="gaussian")
