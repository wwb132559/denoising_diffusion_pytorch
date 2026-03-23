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


def load_cokriging_module(cokriging_dir: Path):
    mod_path = cokriging_dir / "Co-kriging.py"
    spec = importlib.util.spec_from_file_location("cokriging_core", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class PCACoKrigingStressAdapter(nn.Module):
    """
    Co-kriging adapter aligned with denoising_diffusion_pytorch I/O.

    Input:
      - img: [B, 1, 24, 88]
      - classes: [B, 64]

    Output:
      - forward(...) -> scalar regression loss
      - sample(classes=...) -> [B, 1, 24, 88]
    """

    def __init__(self, *, height=24, width=88, channels=1, n_components=16, kernel="matern52"):
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
        self.low_model = None

    def _to_flat(self, img):
        b, c, h, w = img.shape
        if c != self.channels or h != self.height or w != self.width:
            raise ValueError(f"Expected [B,{self.channels},{self.height},{self.width}], got {tuple(img.shape)}")
        return img.reshape(b, -1)

    def _from_flat(self, y_flat):
        b = y_flat.shape[0]
        return y_flat.reshape(b, self.channels, self.height, self.width)

    @staticmethod
    def _to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().astype(np.float64)
        return np.asarray(t, dtype=np.float64)

    @staticmethod
    def _to_tensor(x, ref_device):
        return torch.from_numpy(x.astype(np.float32)).to(ref_device)

    def _fit_pca(self, y_hf):
        self.mean_vec = y_hf.mean(axis=0, keepdims=True)
        yc = y_hf - self.mean_vec

        # Right singular vectors define output basis.
        _, _, vt = np.linalg.svd(yc, full_matrices=False)
        max_comp = min(self.n_components, vt.shape[0])
        self.basis = vt[:max_comp]
        self.n_components = max_comp

    def _project(self, y):
        return (y - self.mean_vec) @ self.basis.T

    def _reconstruct(self, z):
        return z @ self.basis + self.mean_vec

    def fit_offline(self, x_lf, y_lf):
        core = self.core
        self.low_model = core.RecursiveCoKriging(kernel=self.kernel, jitter=1e-10)
        # Single-level kriging for low-fidelity baseline on PCA components.
        self._fit_pca(y_lf)
        z_lf = self._project(y_lf)

        self.low_models = []
        for j in range(self.n_components):
            m = core.RecursiveCoKriging(kernel=self.kernel, jitter=1e-10)
            m.fit([x_lf], [z_lf[:, j]])
            self.low_models.append(m)

    def fit_cokriging(self, x_lf, y_lf, x_hf, y_hf):
        core = self.core
        self._fit_pca(y_hf)
        z_lf = self._project(y_lf)
        z_hf = self._project(y_hf)

        self.models = []
        for j in range(self.n_components):
            m = core.RecursiveCoKriging(kernel=self.kernel, jitter=1e-10)
            m.fit([x_lf, x_hf], [z_lf[:, j], z_hf[:, j]])
            self.models.append(m)

        self.is_fitted = True

    def predict_low(self, x):
        if not hasattr(self, "low_models") or len(self.low_models) == 0:
            raise RuntimeError("Offline model is not fitted. Run fit_offline first.")
        z = np.column_stack([m.predict(x) for m in self.low_models])
        y = self._reconstruct(z)
        return y

    def predict_high(self, x):
        if not self.is_fitted:
            raise RuntimeError("Co-kriging model is not fitted. Run fit_cokriging first.")
        z = np.column_stack([m.predict(x) for m in self.models])
        y = self._reconstruct(z)
        return y

    def forward(self, img, *args, classes, **kwargs):
        _ = args, kwargs
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward.")

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

    def bind_core(self, core_module):
        self.core = core_module

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
            "low_models": getattr(self, "low_models", []),
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
        self.low_models = payload.get("low_models", [])
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
    core = load_cokriging_module(this_dir)

    device = args.device if torch.cuda.is_available() else "cpu"
    d_lf, d_hf, classes = load_data(args, repo_root, device)

    model = PCACoKrigingStressAdapter(
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

    ckpt_dir = this_dir / "results_stress"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    offline_ckpt = ckpt_dir / f"cokriging_{args.offline_tag}.pkl"
    online_ckpt = ckpt_dir / f"cokriging_{args.online_tag}.pkl"

    if args.stage in ("offline", "both"):
        print("[Co-kriging] fitting offline baseline ...")
        model.fit_offline(x_lf, y_lf)
        model.save(offline_ckpt)
        print(f"saved offline model: {offline_ckpt}")

    if args.stage in ("online", "both"):
        if not model.is_fitted:
            if offline_ckpt.exists():
                model.load(offline_ckpt, core)
            print("[Co-kriging] fitting co-kriging (LF + HF) ...")
        model.fit_cokriging(x_lf, y_lf, x_hf, y_hf)
        model.save(online_ckpt)
        print(f"saved online model: {online_ckpt}")

    if args.do_sample:
        if not model.is_fitted:
            if not online_ckpt.exists():
                raise RuntimeError("No fitted co-kriging model found. Run with --stage both first.")
            model.load(online_ckpt, core)

        pred = model.sample(classes=classes, cond_scale=args.cond_scale)
        out_path = ckpt_dir / "cokriging_pred_stress.pt"
        torch.save(pred.cpu(), out_path)

        r2, rrmse, rmae = evaluate_metrics(pred, d_hf)

        print("=" * 72)
        print("[Co-kriging result]")
        print(f"pred shape: {tuple(pred.shape)}")
        print(f"target shape: {tuple(d_hf.shape)}")
        print(f"R2={r2:.4f}, RRMSE={rrmse:.4f}, RMAE={rmae:.4f}")
        print(f"saved pred: {out_path}")
        print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="Co-kriging stress-field reproduction under unified I/O")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", default="both", choices=["offline", "online", "both"])
    parser.add_argument("--group-id", default="1")
    parser.add_argument("--kernel", default="matern52", choices=["matern52", "sqexp", "exp"])
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
