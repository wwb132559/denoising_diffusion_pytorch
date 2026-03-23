import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def add_repo_paths(repo_root: Path):
    sys.path.insert(0, str(repo_root / "denoising_diffusion_pytorch"))


def pairwise_distance(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D2 = np.maximum(A2 + B2 - 2.0 * A @ B.T, 0.0)
    return np.sqrt(D2)


def default_epsilon(X):
    D = pairwise_distance(X, X)
    vals = D[np.triu_indices_from(D, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def rbf_kernel(D, epsilon, kind="gaussian"):
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


def load_csfrbf_module(csfrbf_dir: Path):
    mod_path = csfrbf_dir / "CSF-RBF.py"
    spec = importlib.util.spec_from_file_location("csfrbf_core", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class RBFLowModel:
    """Lightweight RBF interpolator used as LF callable in CSF-RBF."""

    def __init__(self, kernel="gaussian", epsilon=None, reg_lambda=1e-8):
        self.kernel = kernel
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        self.centers_ = X
        self.epsilon_ = float(self.epsilon) if self.epsilon is not None else default_epsilon(X)

        D = pairwise_distance(X, self.centers_)
        Phi = rbf_kernel(D, self.epsilon_, self.kernel)

        reg = self.reg_lambda * np.eye(Phi.shape[0], dtype=np.float64)
        self.alpha_ = np.linalg.solve(Phi + reg, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("RBFLowModel is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        D = pairwise_distance(X, self.centers_)
        Phi = rbf_kernel(D, self.epsilon_, self.kernel)
        return Phi @ self.alpha_


class PCACSFRBFStressAdapter(nn.Module):
    """
    CSF-RBF adapter aligned with denoising_diffusion_pytorch I/O.

    Input:
      - img: [B, 1, 24, 88]
      - classes: [B, 64]

    Output:
      - forward(...) -> scalar regression loss
      - sample(classes=...) -> [B, 1, 24, 88]
    """

    def __init__(self, *, height=24, width=88, channels=1, n_components=16, kernel="gaussian"):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.out_dim = height * width * channels
        self.n_components = int(n_components)
        self.kernel = kernel

        self.is_fitted = False
        self.mean_vec = None
        self.basis = None
        self.models = []
        self.low_models = []

    @staticmethod
    def _to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().astype(np.float64)
        return np.asarray(t, dtype=np.float64)

    @staticmethod
    def _to_tensor(x, ref_device):
        return torch.from_numpy(x.astype(np.float32)).to(ref_device)

    def _to_flat(self, img):
        b, c, h, w = img.shape
        if c != self.channels or h != self.height or w != self.width:
            raise ValueError(f"Expected [B,{self.channels},{self.height},{self.width}], got {tuple(img.shape)}")
        return img.reshape(b, -1)

    def _from_flat(self, y_flat):
        b = y_flat.shape[0]
        return y_flat.reshape(b, self.channels, self.height, self.width)

    def _fit_pca(self, y_ref):
        self.mean_vec = y_ref.mean(axis=0, keepdims=True)
        yc = y_ref - self.mean_vec
        _, _, vt = np.linalg.svd(yc, full_matrices=False)
        self.n_components = min(self.n_components, vt.shape[0])
        self.basis = vt[: self.n_components]

    def _project(self, y):
        return (y - self.mean_vec) @ self.basis.T

    def _reconstruct(self, z):
        return z @ self.basis + self.mean_vec

    def bind_core(self, core_module):
        self.core = core_module

    def fit_offline(self, x_lf, y_lf):
        self._fit_pca(y_lf)
        z_lf = self._project(y_lf)

        self.low_models = []
        for j in range(self.n_components):
            m = RBFLowModel(kernel=self.kernel, epsilon=None, reg_lambda=1e-8)
            m.fit(x_lf, z_lf[:, j])
            self.low_models.append(m)

    def fit_csfrbf(self, x_lf, y_lf, x_hf, y_hf):
        self._fit_pca(y_hf)
        z_lf = self._project(y_lf)
        z_hf = self._project(y_hf)

        self.low_models = []
        self.models = []

        for j in range(self.n_components):
            low = RBFLowModel(kernel=self.kernel, epsilon=None, reg_lambda=1e-8)
            low.fit(x_lf, z_lf[:, j])
            self.low_models.append(low)

            cmodel = self.core.CSFRBF(
                kernel=self.kernel,
                epsilon=None,
                reg_lambda=1e-8,
                normalize_x=True,
                normalize_y=False,
            )
            cmodel.fit(
                X_hf=x_hf,
                y_hf=z_hf[:, j],
                f_low=low.predict,
                centers=x_hf,
            )
            self.models.append(cmodel)

        self.is_fitted = True

    def predict_low(self, x):
        if len(self.low_models) == 0:
            raise RuntimeError("Offline low-fidelity models not fitted.")
        z = np.column_stack([m.predict(x) for m in self.low_models])
        return self._reconstruct(z)

    def predict_high(self, x):
        if not self.is_fitted:
            raise RuntimeError("CSF-RBF model not fitted.")
        z = np.column_stack([m.predict(x) for m in self.models])
        return self._reconstruct(z)

    def forward(self, img, *args, classes, **kwargs):
        _ = args, kwargs
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forward.")

        x = self._to_numpy(classes)
        y_true = self._to_numpy(self._to_flat(img))
        y_pred = self.predict_high(x)

        y_true_t = self._to_tensor(y_true, img.device)
        y_pred_t = self._to_tensor(y_pred, img.device)
        return torch.mean((y_pred_t - y_true_t) ** 2)

    @torch.no_grad()
    def sample(self, classes, cond_scale=1.0, rescaled_phi=0.0):
        _ = cond_scale, rescaled_phi
        x = self._to_numpy(classes)
        y_pred = self.predict_high(x)
        out = self._to_tensor(y_pred, classes.device)
        return self._from_flat(out)

    def save(self, path: Path):
        payload = {
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "n_components": self.n_components,
            "kernel": self.kernel,
            "mean_vec": self.mean_vec,
            "basis": self.basis,
            "is_fitted": self.is_fitted,
            "models": self.models,
            "low_models": self.low_models,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: Path, core_module):
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.height = payload["height"]
        self.width = payload["width"]
        self.channels = payload["channels"]
        self.out_dim = self.height * self.width * self.channels
        self.n_components = payload["n_components"]
        self.kernel = payload["kernel"]
        self.mean_vec = payload["mean_vec"]
        self.basis = payload["basis"]
        self.is_fitted = payload["is_fitted"]
        self.models = payload["models"]
        self.low_models = payload["low_models"]
        self.core = core_module


@torch.no_grad()
def evaluate_metrics(pred, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_res / ss_tot

    tot = torch.sum((target - target_mean) ** 2)
    de = torch.sqrt(torch.mean(tot))
    rmae = torch.abs(torch.max(target - pred)) / de

    rrmse = torch.sqrt(torch.mean((target - pred) ** 2))
    return r2.item(), rrmse.item(), rmae.item()


def load_data(args, repo_root: Path, device: str):
    add_repo_paths(repo_root)
    from stress_data_utils import load_real_stress_data, make_synthetic_stress_data

    if args.use_synthetic_data:
        return make_synthetic_stress_data(
            num_samples=args.synthetic_num_samples,
            height=args.synthetic_height,
            width=args.synthetic_width,
            cond_dim=args.synthetic_cond_dim,
            seed=args.synthetic_seed,
            device=device
        )

    d_lf, d_hf, classes_emb, _, _, _ = load_real_stress_data(
        group_id=args.group_id,
        device=device
    )
    return d_lf, d_hf, classes_emb


def run_pipeline(args):
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[1]
    core = load_csfrbf_module(this_dir)

    device = args.device if torch.cuda.is_available() else "cpu"
    d_lf, d_hf, classes = load_data(args, repo_root, device)

    model = PCACSFRBFStressAdapter(
        height=d_hf.shape[2],
        width=d_hf.shape[3],
        channels=d_hf.shape[1],
        n_components=args.n_components,
        kernel=args.kernel,
    )
    model.bind_core(core)

    x_lf = classes.detach().cpu().numpy().astype(np.float64)
    y_lf = d_lf.reshape(d_lf.shape[0], -1).detach().cpu().numpy().astype(np.float64)

    x_hf = classes.detach().cpu().numpy().astype(np.float64)
    y_hf = d_hf.reshape(d_hf.shape[0], -1).detach().cpu().numpy().astype(np.float64)

    out_dir = this_dir / "results_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    offline_ckpt = out_dir / f"csfrbf_{args.offline_tag}.pkl"
    online_ckpt = out_dir / f"csfrbf_{args.online_tag}.pkl"

    if args.stage in ("offline", "both"):
        print("[CSF-RBF] fitting offline baseline ...")
        model.fit_offline(x_lf, y_lf)
        model.save(offline_ckpt)
        print(f"saved offline model: {offline_ckpt}")

    if args.stage in ("online", "both"):
        print("[CSF-RBF] fitting multifidelity model ...")
        model.fit_csfrbf(x_lf, y_lf, x_hf, y_hf)
        model.save(online_ckpt)
        print(f"saved online model: {online_ckpt}")

    if args.do_sample:
        if not model.is_fitted:
            if not online_ckpt.exists():
                raise RuntimeError("No fitted CSF-RBF model found. Run with --stage both first.")
            model.load(online_ckpt, core)

        pred = model.sample(classes=classes, cond_scale=args.cond_scale)
        pred_path = out_dir / "csfrbf_pred_stress.pt"
        torch.save(pred.cpu(), pred_path)

        r2, rrmse, rmae = evaluate_metrics(pred, d_hf)

        print("=" * 72)
        print("[CSF-RBF result]")
        print(f"pred shape: {tuple(pred.shape)}")
        print(f"target shape: {tuple(d_hf.shape)}")
        print(f"R2={r2:.4f}, RRMSE={rrmse:.4f}, RMAE={rmae:.4f}")
        print(f"saved pred: {pred_path}")
        print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="CSF-RBF stress-field reproduction under unified I/O")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", default="both", choices=["offline", "online", "both"])
    parser.add_argument("--group-id", default="1")
    parser.add_argument("--kernel", default="gaussian", choices=["gaussian", "mq", "imq", "thin_plate"])
    parser.add_argument("--n_components", type=int, default=16)
    parser.add_argument("--offline_tag", default="offline")
    parser.add_argument("--online_tag", default="online")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--cond_scale", type=float, default=8.0)
    parser.add_argument("--use-synthetic-data", action="store_true")
    parser.add_argument("--synthetic-seed", type=int, default=20260320)
    parser.add_argument("--synthetic-num-samples", type=int, default=24)
    parser.add_argument("--synthetic-height", type=int, default=24)
    parser.add_argument("--synthetic-width", type=int, default=88)
    parser.add_argument("--synthetic-cond-dim", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
