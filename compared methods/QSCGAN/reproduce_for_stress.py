import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_qscgan_module(qscgan_dir: Path, repo_root: Path):
    # QSCGAN.py depends on utils.attention; reuse the implementation shipped with VGCDM.
    sys.path.insert(0, str(repo_root / "compared methods" / "VGCDM"))
    sys.path.insert(0, str(repo_root / "denoising_diffusion_pytorch"))

    mod_path = qscgan_dir / "QSCGAN.py"
    spec = importlib.util.spec_from_file_location("qscgan_core", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class QSCGANStressAdapter(nn.Module):
    """
    QSCGAN adapter aligned with denoising_diffusion_pytorch I/O.

    Input:
      - img: [B, 1, 24, 88]
      - classes: [B, 64]

    Output:
      - forward(...) -> scalar regression loss
      - sample(classes=...) -> [B, 1, 24, 88]
    """

    def __init__(
        self,
        qscgan_mod,
        *,
        height=24,
        width=88,
        channels=1,
        cond_dim=64,
        nz=256,
        device="cuda:0",
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.seq_len = height * width
        self.cond_dim = cond_dim
        self.nz = nz

        self.generator = qscgan_mod.Generator(nz)
        self.discriminator = qscgan_mod.Discriminator()
        self.class_proj = nn.Sequential(
            nn.Linear(cond_dim, nz),
            nn.Tanh(),
        )

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

    def _make_latent(self, classes, noise_scale=0.1):
        cond_z = self.class_proj(classes)
        noise = torch.randn_like(cond_z) * noise_scale
        z = (cond_z + noise).unsqueeze(-1)
        return z

    def forward(self, img, *args, classes, **kwargs):
        _ = args, kwargs
        z = self._make_latent(classes, noise_scale=0.0)
        fake_2048 = self.generator(z)
        fake_seq = F.interpolate(fake_2048, size=self.seq_len, mode="linear", align_corners=False)

        real_seq = self._to_seq(img)
        return torch.mean((fake_seq - real_seq) ** 2)

    @torch.no_grad()
    def sample(self, classes, cond_scale=1.0, rescaled_phi=0.0):
        _ = cond_scale, rescaled_phi
        z = self._make_latent(classes, noise_scale=0.0)
        fake_2048 = self.generator(z)
        fake_seq = F.interpolate(fake_2048, size=self.seq_len, mode="linear", align_corners=False)
        return self._to_map(fake_seq)


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
    lr_g,
    lr_d,
    batch_size,
    adv_weight,
    rec_weight,
):
    dataset = TensorDataset(data, classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    bce = nn.BCELoss()
    opt_g = torch.optim.Adam(list(model.generator.parameters()) + list(model.class_proj.parameters()), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    model.generator.train()
    model.discriminator.train()
    model.class_proj.train()

    t0 = time.time()
    for ep in range(epochs):
        g_loss_ep = 0.0
        d_loss_ep = 0.0

        for real_map, cls in loader:
            real_map = real_map.to(model.device)
            cls = cls.to(model.device)

            real_seq_2112 = model._to_seq(real_map)
            real_seq_2048 = F.interpolate(real_seq_2112, size=2048, mode="linear", align_corners=False)

            # train D
            opt_d.zero_grad()
            real_label = torch.ones((real_map.size(0), 1, 1), device=model.device)
            fake_label = torch.zeros((real_map.size(0), 1, 1), device=model.device)

            d_real = model.discriminator(real_seq_2048)
            loss_d_real = bce(d_real, real_label)

            z = model._make_latent(cls, noise_scale=0.1)
            fake_2048 = model.generator(z)
            d_fake = model.discriminator(fake_2048.detach())
            loss_d_fake = bce(d_fake, fake_label)

            d_loss = 0.5 * (loss_d_real + loss_d_fake)
            d_loss.backward()
            opt_d.step()

            # train G
            opt_g.zero_grad()
            z = model._make_latent(cls, noise_scale=0.1)
            fake_2048 = model.generator(z)
            d_fake_for_g = model.discriminator(fake_2048)
            adv_loss = bce(d_fake_for_g, real_label)

            fake_2112 = F.interpolate(fake_2048, size=model.seq_len, mode="linear", align_corners=False)
            rec_loss = F.l1_loss(fake_2112, real_seq_2112)

            g_loss = adv_weight * adv_loss + rec_weight * rec_loss
            g_loss.backward()
            opt_g.step()

            g_loss_ep += g_loss.item()
            d_loss_ep += d_loss.item()

        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {ep + 1:03d}/{epochs} | G={g_loss_ep / len(loader):.4f} D={d_loss_ep / len(loader):.4f}")

    elapsed = time.time() - t0
    return elapsed


def run_pipeline(args):
    qscgan_dir = Path(__file__).resolve().parent
    repo_root = qscgan_dir.parents[1]
    qscgan_mod = load_qscgan_module(qscgan_dir, repo_root)

    device = args.device if torch.cuda.is_available() else "cpu"
    d_lf, d_hf, classes = load_data(args, repo_root, device)

    d_lf_n01, _, _ = minmax_norm(d_lf)
    d_hf_n01, hf_min, hf_max = minmax_norm(d_hf)

    # map to [-1,1] for tanh-based GAN
    d_lf_n11 = d_lf_n01 * 2 - 1
    d_hf_n11 = d_hf_n01 * 2 - 1

    model = QSCGANStressAdapter(
        qscgan_mod,
        height=d_hf.shape[2],
        width=d_hf.shape[3],
        channels=d_hf.shape[1],
        cond_dim=classes.shape[1],
        nz=args.nz,
        device=device,
    )

    out_dir = qscgan_dir / "results_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    offline_ckpt = out_dir / f"qscgan_{args.offline_tag}.pt"
    online_ckpt = out_dir / f"qscgan_{args.online_tag}.pt"

    if args.stage in ("offline", "both"):
        print("[QSCGAN] offline training on LF data ...")
        t_off = train_one_stage(
            model,
            d_lf_n11,
            classes,
            epochs=args.offline_epochs,
            lr_g=args.offline_lr_g,
            lr_d=args.offline_lr_d,
            batch_size=args.batch_size,
            adv_weight=args.adv_weight,
            rec_weight=args.rec_weight,
        )
        torch.save(
            {
                "generator": model.generator.state_dict(),
                "discriminator": model.discriminator.state_dict(),
                "class_proj": model.class_proj.state_dict(),
                "offline_time": t_off,
            },
            offline_ckpt,
        )
        print(f"saved offline model: {offline_ckpt}")

    if args.stage in ("online", "both"):
        if offline_ckpt.exists() and args.stage == "online":
            ckpt = torch.load(offline_ckpt, map_location=device)
            model.generator.load_state_dict(ckpt["generator"])
            model.discriminator.load_state_dict(ckpt["discriminator"])
            model.class_proj.load_state_dict(ckpt["class_proj"])

        print("[QSCGAN] online fine-tuning on HF data ...")
        t_on = train_one_stage(
            model,
            d_hf_n11,
            classes,
            epochs=args.online_epochs,
            lr_g=args.online_lr_g,
            lr_d=args.online_lr_d,
            batch_size=args.batch_size,
            adv_weight=args.adv_weight,
            rec_weight=args.rec_weight,
        )
        torch.save(
            {
                "generator": model.generator.state_dict(),
                "discriminator": model.discriminator.state_dict(),
                "class_proj": model.class_proj.state_dict(),
                "online_time": t_on,
            },
            online_ckpt,
        )
        print(f"saved online model: {online_ckpt}")

    if args.do_sample:
        if online_ckpt.exists():
            ckpt = torch.load(online_ckpt, map_location=device)
            model.generator.load_state_dict(ckpt["generator"])
            model.class_proj.load_state_dict(ckpt["class_proj"])

        model.generator.eval()
        model.class_proj.eval()

        with torch.no_grad():
            pred_n11 = model.sample(classes=classes, cond_scale=args.cond_scale)
            pred_n01 = (pred_n11 + 1.0) * 0.5
            pred = pred_n01 * (hf_max - hf_min) + hf_min

        pred_path = out_dir / "qscgan_pred_stress.pt"
        torch.save(pred.cpu(), pred_path)

        r2, rrmse, rmae = evaluate_metrics(pred, d_hf)

        print("=" * 72)
        print("[QSCGAN result]")
        print(f"pred shape: {tuple(pred.shape)}")
        print(f"target shape: {tuple(d_hf.shape)}")
        print(f"R2={r2:.4f}, RRMSE={rrmse:.4f}, RMAE={rmae:.4f}")
        print(f"saved pred: {pred_path}")
        print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="QSCGAN stress-field reproduction under unified I/O")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", default="both", choices=["offline", "online", "both"])
    parser.add_argument("--group-id", default="1")

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--nz", type=int, default=256)

    parser.add_argument("--offline_epochs", type=int, default=300)
    parser.add_argument("--online_epochs", type=int, default=300)
    parser.add_argument("--offline_lr_g", type=float, default=1e-4)
    parser.add_argument("--offline_lr_d", type=float, default=1e-4)
    parser.add_argument("--online_lr_g", type=float, default=5e-5)
    parser.add_argument("--online_lr_d", type=float, default=5e-5)

    parser.add_argument("--adv_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=50.0)

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
