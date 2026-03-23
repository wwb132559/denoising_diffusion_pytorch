import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_vqvae_module(vqvae_dir: Path):
    mod_path = vqvae_dir / "VQ_VAE.py"
    spec = importlib.util.spec_from_file_location("vqvae_core", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class VQVAEStressAdapter(nn.Module):
    """
    VQ-VAE adapter aligned with denoising_diffusion_pytorch I/O.

    Input:
      - img: [B, 1, 24, 88]
      - classes: [B, 64]

    Output:
      - forward(...) -> scalar regression loss
      - sample(classes=...) -> [B, 1, 24, 88]
    """

    def __init__(
        self,
        vqvae_mod,
        *,
        height=24,
        width=88,
        channels=1,
        cond_dim=64,
        num_hiddens=128,
        num_residual_layers=4,
        num_residual_hiddens=128,
        num_embeddings=256,
        embedding_dim=32,
        commitment_cost=0.25,
        decay=0.99,
        vq_weight=1.0,
        rec_weight=1.0,
        device="cuda:0",
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.seq_len = height * width
        self.cond_dim = cond_dim
        self.embedding_dim = embedding_dim
        self.vq_weight = vq_weight
        self.rec_weight = rec_weight

        self.model = vqvae_mod.VQ_VAE(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        self.class_proj = nn.Sequential(
            nn.Linear(cond_dim, embedding_dim),
            nn.Tanh(),
        )

        # A small latent template helps class-only sampling be less degenerate.
        with torch.no_grad():
            dummy = torch.zeros(1, channels, self.seq_len)
            z = self.model._pre_vq_conv(self.model._encoder(dummy))
        self.latent_len = z.shape[-1]
        self.latent_template = nn.Parameter(torch.zeros(1, embedding_dim, self.latent_len))

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _to_seq(self, img):
        b, c, h, w = img.shape
        if c != self.channels or h != self.height or w != self.width:
            raise ValueError(f"Expected [B,{self.channels},{self.height},{self.width}], got {tuple(img.shape)}")
        return img.reshape(b, c, self.seq_len)

    def _to_map(self, seq):
        b, c, l = seq.shape
        if c != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {c}")
        if l != self.seq_len:
            seq = F.interpolate(seq, size=self.seq_len, mode="linear", align_corners=False)
        return seq.reshape(b, c, self.height, self.width)

    def _inject_condition(self, z, classes, cond_scale=1.0):
        cond = self.class_proj(classes).unsqueeze(-1)
        return z + cond_scale * cond

    def _reconstruct_seq(self, seq, classes, cond_scale=1.0):
        z = self.model._encoder(seq)
        z = self.model._pre_vq_conv(z)
        z = self._inject_condition(z, classes, cond_scale=cond_scale)
        vq_loss, quantized, _, _ = self.model._vq_vae(z)
        recon = self.model._decoder(quantized)
        return vq_loss, recon

    def forward(self, img, *args, classes, **kwargs):
        _ = args, kwargs
        seq = self._to_seq(img)
        vq_loss, recon = self._reconstruct_seq(seq, classes, cond_scale=1.0)
        rec_loss = F.mse_loss(recon, seq)
        return self.rec_weight * rec_loss + self.vq_weight * vq_loss

    @torch.no_grad()
    def reconstruct(self, img, classes, cond_scale=1.0):
        seq = self._to_seq(img)
        _, recon = self._reconstruct_seq(seq, classes, cond_scale=cond_scale)
        return self._to_map(recon)

    @torch.no_grad()
    def sample(self, classes, cond_scale=1.0, rescaled_phi=0.0):
        _ = rescaled_phi
        b = classes.shape[0]
        cond = self.class_proj(classes).unsqueeze(-1)
        z0 = self.latent_template.expand(b, -1, -1)
        z = z0 + cond_scale * cond
        _, quantized, _, _ = self.model._vq_vae(z)
        recon = self.model._decoder(quantized)
        return self._to_map(recon)


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
    sys.path.insert(0, str(repo_root / "denoising_diffusion_pytorch"))
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

    d_lf, d_hf, classes, _, _, _ = load_real_stress_data(
        group_id=args.group_id,
        device=device
    )
    return d_lf, d_hf, classes


def minmax_norm(x):
    x_max = x.max()
    x_min = x.min()
    xn = (x - x_min) / (x_max - x_min)
    return xn, x_min, x_max


def train_one_stage(
    model,
    data,
    classes,
    *,
    epochs,
    lr,
    batch_size,
):
    dataset = TensorDataset(data, classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    t0 = time.time()
    for ep in range(epochs):
        loss_ep = 0.0
        for stress_map, cls in loader:
            stress_map = stress_map.to(model.device)
            cls = cls.to(model.device)

            optimizer.zero_grad()
            loss = model(stress_map, classes=cls)
            loss.backward()
            optimizer.step()

            loss_ep += loss.item()

        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {ep + 1:03d}/{epochs} | loss={loss_ep / len(loader):.6f}")

    elapsed = time.time() - t0
    return elapsed


def run_pipeline(args):
    vqvae_dir = Path(__file__).resolve().parent
    repo_root = vqvae_dir.parents[1]
    vqvae_mod = load_vqvae_module(vqvae_dir)

    device = args.device if torch.cuda.is_available() else "cpu"
    d_lf, d_hf, classes = load_data(args, repo_root, device)

    d_lf_n01, _, _ = minmax_norm(d_lf)
    d_hf_n01, hf_min, hf_max = minmax_norm(d_hf)

    model = VQVAEStressAdapter(
        vqvae_mod,
        height=d_hf.shape[2],
        width=d_hf.shape[3],
        channels=d_hf.shape[1],
        cond_dim=classes.shape[1],
        num_hiddens=args.num_hiddens,
        num_residual_layers=args.num_residual_layers,
        num_residual_hiddens=args.num_residual_hiddens,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        vq_weight=args.vq_weight,
        rec_weight=args.rec_weight,
        device=device,
    )

    out_dir = vqvae_dir / "results_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    offline_ckpt = out_dir / f"vqvae_{args.offline_tag}.pt"
    online_ckpt = out_dir / f"vqvae_{args.online_tag}.pt"

    if args.stage in ("offline", "both"):
        print("[VQ_VAE] offline training on LF data ...")
        t_off = train_one_stage(
            model,
            d_lf_n01,
            classes,
            epochs=args.offline_epochs,
            lr=args.offline_lr,
            batch_size=args.batch_size,
        )
        torch.save(
            {
                "model": model.model.state_dict(),
                "class_proj": model.class_proj.state_dict(),
                "latent_template": model.latent_template.detach().cpu(),
                "offline_time": t_off,
            },
            offline_ckpt,
        )
        print(f"saved offline model: {offline_ckpt}")

    if args.stage in ("online", "both"):
        if offline_ckpt.exists() and args.stage == "online":
            ckpt = torch.load(offline_ckpt, map_location=device)
            model.model.load_state_dict(ckpt["model"])
            model.class_proj.load_state_dict(ckpt["class_proj"])
            model.latent_template.data.copy_(ckpt["latent_template"].to(device))

        print("[VQ_VAE] online fine-tuning on HF data ...")
        t_on = train_one_stage(
            model,
            d_hf_n01,
            classes,
            epochs=args.online_epochs,
            lr=args.online_lr,
            batch_size=args.batch_size,
        )
        torch.save(
            {
                "model": model.model.state_dict(),
                "class_proj": model.class_proj.state_dict(),
                "latent_template": model.latent_template.detach().cpu(),
                "online_time": t_on,
            },
            online_ckpt,
        )
        print(f"saved online model: {online_ckpt}")

    if args.do_sample:
        if online_ckpt.exists():
            ckpt = torch.load(online_ckpt, map_location=device)
            model.model.load_state_dict(ckpt["model"])
            model.class_proj.load_state_dict(ckpt["class_proj"])
            model.latent_template.data.copy_(ckpt["latent_template"].to(device))

        model.eval()
        with torch.no_grad():
            pred_n01 = model.sample(classes=classes, cond_scale=args.cond_scale)
            pred = pred_n01 * (hf_max - hf_min) + hf_min

        pred_path = out_dir / "vqvae_pred_stress.pt"
        torch.save(pred.cpu(), pred_path)

        r2, rrmse, rmae = evaluate_metrics(pred, d_hf)

        print("=" * 72)
        print("[VQ_VAE result]")
        print(f"pred shape: {tuple(pred.shape)}")
        print(f"target shape: {tuple(d_hf.shape)}")
        print(f"R2={r2:.4f}, RRMSE={rrmse:.4f}, RMAE={rmae:.4f}")
        print(f"saved pred: {pred_path}")
        print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="VQ_VAE stress-field reproduction under unified I/O")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", default="both", choices=["offline", "online", "both"])
    parser.add_argument("--group-id", default="1")

    parser.add_argument("--batch_size", type=int, default=24)

    parser.add_argument("--offline_epochs", type=int, default=300)
    parser.add_argument("--online_epochs", type=int, default=300)
    parser.add_argument("--offline_lr", type=float, default=2e-4)
    parser.add_argument("--online_lr", type=float, default=1e-4)

    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--num_residual_layers", type=int, default=4)
    parser.add_argument("--num_residual_hiddens", type=int, default=128)
    parser.add_argument("--num_embeddings", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)

    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=1.0)

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
