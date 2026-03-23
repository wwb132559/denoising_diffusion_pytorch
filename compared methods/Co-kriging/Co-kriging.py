from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

KernelName = Literal["matern52", "sqexp", "exp"]


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a[:, None]
    return a


def lhs(n_samples: int, n_dim: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        X[:, j] = (perm + rng.random(n_samples)) / n_samples
    return X


def _pairwise_absdiff(XA: np.ndarray, XB: np.ndarray, dim: int) -> np.ndarray:
    return np.abs(XA[:, None, dim] - XB[None, :, dim])


def correlation_matrix(
    XA: np.ndarray,
    XB: np.ndarray,
    theta: np.ndarray,
    kernel: KernelName = "matern52",
) -> np.ndarray:
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    if theta.size == 1:
        theta = np.repeat(theta[0], XA.shape[1])
    if theta.size != XA.shape[1]:
        raise ValueError("theta must have length 1 or match the input dimension.")

    theta = np.maximum(theta, 1e-12)

    if kernel == "sqexp":
        acc = np.zeros((XA.shape[0], XB.shape[0]), dtype=float)
        for k in range(XA.shape[1]):
            d = _pairwise_absdiff(XA, XB, k) / theta[k]
            acc += d * d
        return np.exp(-0.5 * acc)

    if kernel == "exp":
        acc = np.zeros((XA.shape[0], XB.shape[0]), dtype=float)
        for k in range(XA.shape[1]):
            d = _pairwise_absdiff(XA, XB, k) / theta[k]
            acc += d
        return np.exp(-acc)

    if kernel == "matern52":
        R = np.ones((XA.shape[0], XB.shape[0]), dtype=float)
        sqrt5 = np.sqrt(5.0)
        for k in range(XA.shape[1]):
            d = _pairwise_absdiff(XA, XB, k) / theta[k]
            R *= (1.0 + sqrt5 * d + (5.0 / 3.0) * d * d) * np.exp(-sqrt5 * d)
        return R

    raise ValueError(f"Unsupported kernel: {kernel}")


def _safe_cholesky(K: np.ndarray, jitter: float) -> Tuple[np.ndarray, bool]:
    n = K.shape[0]
    eye = np.eye(n)
    current = max(float(jitter), 1e-12)
    for _ in range(8):
        try:
            return cho_factor(K + current * eye, lower=True, check_finite=False), True
        except np.linalg.LinAlgError:
            current *= 10.0
    raise np.linalg.LinAlgError("Covariance matrix is not positive definite even after adding jitter.")


def _solve_chol(chol, B: np.ndarray) -> np.ndarray:
    return cho_solve(chol, B, check_finite=False)


def _logdet_from_chol(chol) -> float:
    c, lower = chol
    diag = np.diag(c)
    return 2.0 * float(np.sum(np.log(np.maximum(diag, 1e-300))))


def _default_theta0(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        vals = np.abs(X[:, None, j] - X[None, :, j])
        vals = vals[np.triu_indices_from(vals, k=1)]
        vals = vals[vals > 0]
        out[j] = np.median(vals) if vals.size > 0 else 0.2
    out[out <= 0] = 0.2
    return out


def _row_quadratic(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk,ik->i", A, M, A)


def constant_trend(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.ones((X.shape[0], 1), dtype=float)


@dataclass
class LevelFit:
    X: np.ndarray
    y: np.ndarray
    F: np.ndarray
    H: np.ndarray
    theta: np.ndarray
    beta_all: np.ndarray
    beta_rho: np.ndarray
    beta: np.ndarray
    Sigma: np.ndarray
    Sigma_rho: np.ndarray
    sigma2_reml: float
    sigma2_post: float
    q_value: float
    dof: int
    chol: Tuple[np.ndarray, bool]
    R_inv_H: np.ndarray
    alpha: np.ndarray
    rho_train: np.ndarray
    y_prev_on_train: Optional[np.ndarray]
    idx_prev: Optional[np.ndarray]


class RecursiveCoKriging:
    """
    Recursive multi-fidelity co-kriging with nested training sets.

    The levels must be ordered from low fidelity to high fidelity:
        X_levels = [X1, X2, ..., Xs]
        y_levels = [y1, y2, ..., ys]
    with X_s ⊆ X_{s-1} ⊆ ... ⊆ X_1.
    """

    def __init__(
        self,
        kernel: KernelName = "matern52",
        jitter: float = 1e-10,
        normalize_x: bool = True,
    ) -> None:
        self.kernel = kernel
        self.jitter = float(jitter)
        self.normalize_x = bool(normalize_x)
        self.levels_: List[LevelFit] = []

    def _fit_x_scaler(self, X_ref: np.ndarray) -> None:
        self.x_min_ = np.min(X_ref, axis=0)
        self.x_max_ = np.max(X_ref, axis=0)
        span = self.x_max_ - self.x_min_
        span[span == 0] = 1.0
        self.x_span_ = span

    def _scale_x(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.normalize_x:
            return X
        return (X - self.x_min_) / self.x_span_

    @staticmethod
    def _build_subset_indices(X_small: np.ndarray, X_large: np.ndarray, decimals: int = 12) -> np.ndarray:
        table = {}
        for i, row in enumerate(np.round(X_large, decimals=decimals)):
            key = tuple(row.tolist())
            table.setdefault(key, []).append(i)

        idx = []
        for row in np.round(X_small, decimals=decimals):
            key = tuple(row.tolist())
            if key not in table or len(table[key]) == 0:
                raise ValueError("Training sets must be nested exactly: each higher-fidelity point must appear in the lower-fidelity set.")
            idx.append(table[key].pop(0))
        return np.asarray(idx, dtype=int)

    def _prepare_functions(
        self,
        n_levels: int,
        f_designs: Optional[Sequence[Callable[[np.ndarray], np.ndarray]]],
        g_designs: Optional[Sequence[Callable[[np.ndarray], np.ndarray]]],
    ) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], List[Callable[[np.ndarray], np.ndarray]]]:
        if f_designs is None:
            f_list = [constant_trend for _ in range(n_levels)]
        else:
            if len(f_designs) != n_levels:
                raise ValueError("f_designs must have one callable per level.")
            f_list = list(f_designs)

        if g_designs is None:
            g_list = [constant_trend for _ in range(max(n_levels - 1, 0))]
        else:
            if len(g_designs) != n_levels - 1:
                raise ValueError("g_designs must have one callable for each adjustment level.")
            g_list = list(g_designs)

        return f_list, g_list

    def _fit_one_level(
        self,
        X: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        theta0: Optional[np.ndarray],
        theta_bounds: Optional[Sequence[Tuple[float, float]]],
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, bool], np.ndarray, np.ndarray, np.ndarray, float, float, int]:
        n, d = X.shape
        p_total = H.shape[1]
        dof = n - p_total
        if dof <= 0:
            raise ValueError("Each level must satisfy n_samples > number of regression coefficients.")

        theta0 = _default_theta0(X) if theta0 is None else np.asarray(theta0, dtype=float).reshape(-1)
        if theta0.size == 1:
            theta0 = np.repeat(theta0[0], d)
        if theta0.size != d:
            raise ValueError("Each theta0 must have length 1 or equal to the input dimension.")

        if theta_bounds is None:
            theta_bounds = [(1e-3, 10.0)] * d
        if len(theta_bounds) == 1 and d > 1:
            theta_bounds = list(theta_bounds) * d
        if len(theta_bounds) != d:
            raise ValueError("theta_bounds must have one bound per input dimension.")

        log_bounds = [(np.log(max(lo, 1e-12)), np.log(max(hi, lo * 1.0001))) for lo, hi in theta_bounds]

        def objective(log_theta: np.ndarray) -> float:
            theta = np.exp(log_theta)
            R = correlation_matrix(X, X, theta, self.kernel)
            chol, _ = _safe_cholesky(R, self.jitter)
            R_inv_y = _solve_chol(chol, y)
            R_inv_H = _solve_chol(chol, H)
            M = H.T @ R_inv_H
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                return 1e30
            beta_hat = M_inv @ (H.T @ R_inv_y)
            resid = y - H @ beta_hat
            q_value = float(resid.T @ _solve_chol(chol, resid))
            q_value = max(q_value, 1e-300)
            return _logdet_from_chol(chol) + dof * np.log(q_value / dof)

        opt = minimize(
            objective,
            x0=np.log(np.maximum(theta0, 1e-6)),
            method="L-BFGS-B",
            bounds=log_bounds,
        )
        theta = np.exp(opt.x if opt.success else np.log(np.maximum(theta0, 1e-6)))

        R = correlation_matrix(X, X, theta, self.kernel)
        chol, _ = _safe_cholesky(R, self.jitter)
        R_inv_y = _solve_chol(chol, y)
        R_inv_H = _solve_chol(chol, H)
        M = H.T @ R_inv_H
        M_inv = np.linalg.inv(M)
        beta_hat = M_inv @ (H.T @ R_inv_y)
        resid = y - H @ beta_hat
        q_value = float(resid.T @ _solve_chol(chol, resid))
        sigma2_reml = q_value / dof
        sigma2_post = q_value / (dof - 2) if dof > 2 else sigma2_reml
        alpha = _solve_chol(chol, resid)
        return theta, chol, beta_hat, M_inv, alpha, sigma2_reml, sigma2_post, q_value

    def fit(
        self,
        X_levels: Sequence[np.ndarray],
        y_levels: Sequence[np.ndarray],
        f_designs: Optional[Sequence[Callable[[np.ndarray], np.ndarray]]] = None,
        g_designs: Optional[Sequence[Callable[[np.ndarray], np.ndarray]]] = None,
        theta0_levels: Optional[Sequence[np.ndarray]] = None,
        theta_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> "RecursiveCoKriging":
        if len(X_levels) != len(y_levels):
            raise ValueError("X_levels and y_levels must have the same number of levels.")
        if len(X_levels) < 1:
            raise ValueError("At least one fidelity level is required.")

        X_levels = [np.asarray(X, dtype=float) for X in X_levels]
        y_levels = [np.asarray(y, dtype=float).reshape(-1) for y in y_levels]

        d = X_levels[0].shape[1]
        for X, y in zip(X_levels, y_levels):
            if X.ndim != 2 or X.shape[1] != d:
                raise ValueError("All design matrices must be 2D and share the same input dimension.")
            if X.shape[0] != y.shape[0]:
                raise ValueError("Each level must satisfy len(y) == X.shape[0].")

        self._fit_x_scaler(X_levels[0])
        Xs_levels = [self._scale_x(X) for X in X_levels]
        self.n_levels_ = len(X_levels)

        f_list, g_list = self._prepare_functions(self.n_levels_, f_designs, g_designs)
        self.f_designs_ = f_list
        self.g_designs_ = g_list

        self.levels_ = []
        self.idx_maps_ = [None]
        for t in range(1, self.n_levels_):
            idx_prev = self._build_subset_indices(Xs_levels[t], Xs_levels[t - 1])
            self.idx_maps_.append(idx_prev)

        if theta0_levels is None:
            theta0_levels = [None] * self.n_levels_
        if len(theta0_levels) != self.n_levels_:
            raise ValueError("theta0_levels must have one entry per level.")

        for t in range(self.n_levels_):
            X_t = Xs_levels[t]
            y_t = y_levels[t]
            F_t = _as_2d(f_list[t](X_t))

            if t == 0:
                H_t = F_t
                y_prev_on_train = None
                rho_train = np.zeros_like(y_t)
                idx_prev = None
            else:
                idx_prev = self.idx_maps_[t]
                y_prev_on_train = y_levels[t - 1][idx_prev]
                G_t = _as_2d(g_list[t - 1](X_t))
                H_t = np.hstack([G_t * y_prev_on_train[:, None], F_t])
                rho_train = G_t @ np.zeros(G_t.shape[1])

            theta, chol, beta_hat, Sigma, alpha, sigma2_reml, sigma2_post, q_value = self._fit_one_level(
                X=X_t,
                y=y_t,
                H=H_t,
                theta0=theta0_levels[t],
                theta_bounds=theta_bounds,
            )

            if t == 0:
                beta_rho = np.zeros(0, dtype=float)
                beta = beta_hat
                Sigma_rho = np.zeros((0, 0), dtype=float)
                rho_train = np.zeros_like(y_t)
            else:
                q_rho = _as_2d(g_list[t - 1](X_t)).shape[1]
                beta_rho = beta_hat[:q_rho]
                beta = beta_hat[q_rho:]
                Sigma_rho = Sigma[:q_rho, :q_rho]
                rho_train = _as_2d(g_list[t - 1](X_t)) @ beta_rho

            R_inv_H = _solve_chol(chol, H_t)
            level = LevelFit(
                X=X_t,
                y=y_t,
                F=F_t,
                H=H_t,
                theta=theta,
                beta_all=beta_hat,
                beta_rho=beta_rho,
                beta=beta,
                Sigma=Sigma,
                Sigma_rho=Sigma_rho,
                sigma2_reml=sigma2_reml,
                sigma2_post=sigma2_post,
                q_value=q_value,
                dof=X_t.shape[0] - H_t.shape[1],
                chol=chol,
                R_inv_H=R_inv_H,
                alpha=alpha,
                rho_train=rho_train,
                y_prev_on_train=y_prev_on_train,
                idx_prev=idx_prev,
            )
            self.levels_.append(level)

        self.is_fitted_ = True
        return self

    def _predict_level(self, level_idx: int, X: np.ndarray, mode: Literal["simple", "universal"]) -> Tuple[np.ndarray, np.ndarray]:
        level = self.levels_[level_idx]
        Xs = self._scale_x(X)
        K = correlation_matrix(Xs, level.X, level.theta, self.kernel)
        R_inv_Kt = _solve_chol(level.chol, K.T)
        qf = np.sum(K * R_inv_Kt.T, axis=1)

        F_x = _as_2d(self.f_designs_[level_idx](Xs))

        if level_idx == 0:
            mean = (F_x @ level.beta).reshape(-1) + K @ level.alpha
            if mode == "simple":
                var = level.sigma2_reml * np.maximum(1.0 - qf, 0.0)
            else:
                c = F_x - K @ level.R_inv_H
                var = level.sigma2_post * np.maximum(1.0 - qf, 0.0) + _row_quadratic(c, level.Sigma)
            return mean, np.maximum(var, 0.0)

        prev_mean, prev_var = self._predict_level(level_idx - 1, X, mode)
        G_x = _as_2d(self.g_designs_[level_idx - 1](Xs))
        rho_x = (G_x @ level.beta_rho).reshape(-1)
        mean = rho_x * prev_mean + (F_x @ level.beta).reshape(-1) + K @ level.alpha

        if mode == "simple":
            var = (rho_x ** 2) * prev_var + level.sigma2_reml * np.maximum(1.0 - qf, 0.0)
        else:
            sigma_rho2 = rho_x ** 2 + _row_quadratic(G_x, level.Sigma_rho)
            h_x = np.hstack([G_x * prev_mean[:, None], F_x])
            c = h_x - K @ level.R_inv_H
            var = sigma_rho2 * prev_var + level.sigma2_post * np.maximum(1.0 - qf, 0.0) + _row_quadratic(c, level.Sigma)
        return mean, np.maximum(var, 0.0)

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        mode: Literal["simple", "universal"] = "universal",
        level: Optional[int] = None,
    ):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("The model must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        target_level = self.n_levels_ - 1 if level is None else int(level)
        mean, var = self._predict_level(target_level, X, mode)
        if return_std:
            return mean, np.sqrt(np.maximum(var, 0.0))
        return mean

    def predict_all_levels(
        self,
        X: np.ndarray,
        return_std: bool = False,
        mode: Literal["simple", "universal"] = "universal",
    ):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("The model must be fitted before prediction.")
        means = []
        stds = []
        X = np.asarray(X, dtype=float)
        for lvl in range(self.n_levels_):
            mean, var = self._predict_level(lvl, X, mode)
            means.append(mean)
            stds.append(np.sqrt(np.maximum(var, 0.0)))
        if return_std:
            return means, stds
        return means

    def score_rmse(self, X: np.ndarray, y_true: np.ndarray, level: Optional[int] = None, mode: str = "universal") -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = self.predict(X, return_std=False, mode=mode, level=level)
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ---------- Example test functions ----------

def currin_hf(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    x1 = X[:, 0]
    x2 = np.maximum(X[:, 1], 1e-12)
    num = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
    den = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0
    return (1.0 - np.exp(-1.0 / (2.0 * x2))) * (num / den)


def _currin_point(x1: float, x2: float) -> float:
    return float(currin_hf(np.array([[x1, x2]], dtype=float))[0])


def currin_lf(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = []
    for x1, x2 in X:
        val = (
            _currin_point(x1 + 0.05, x2 + 0.05)
            + _currin_point(x1 + 0.05, max(0.0, x2 - 0.05))
            + _currin_point(x1 - 0.05, x2 + 0.05)
            + _currin_point(x1 - 0.05, max(0.0, x2 - 0.05))
        ) / 8.0 + ((-5.0 * x1 - 7.0 * x2) ** 2) / 8.0
        out.append(val)
    return np.asarray(out, dtype=float)


def build_nested_doe(n_low: int, n_high: int, d: int = 2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if n_high > n_low:
        raise ValueError("n_high must not exceed n_low.")
    X_low = lhs(n_low, d, seed=seed)
    rng = np.random.default_rng(seed + 1)
    idx = np.sort(rng.choice(n_low, size=n_high, replace=False))
    X_high = X_low[idx]
    return X_low, X_high


def demo_currin(seed: int = 0) -> Tuple[RecursiveCoKriging, float]:
    X_low, X_high = build_nested_doe(n_low=25, n_high=8, d=2, seed=seed)
    y_low = currin_lf(X_low)
    y_high = currin_hf(X_high)

    model = RecursiveCoKriging(kernel="matern52", jitter=1e-10)
    model.fit([X_low, X_high], [y_low, y_high])

    grid = np.linspace(0.0, 1.0, 80)
    xx, yy = np.meshgrid(grid, grid)
    X_test = np.column_stack([xx.ravel(), yy.ravel()])
    y_test = currin_hf(X_test)

    rmse = model.score_rmse(X_test, y_test, mode="universal")
    print("RMSE:", rmse)
    print("theta level 1:", model.levels_[0].theta)
    print("theta level 2:", model.levels_[1].theta)
    print("rho level 2:", model.levels_[1].beta_rho)
    return model, rmse


if __name__ == "__main__":
    demo_currin(seed=42)
