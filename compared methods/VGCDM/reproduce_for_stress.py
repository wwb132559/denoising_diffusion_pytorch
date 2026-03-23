import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from model.diffusion.Unet1D import Unet1D_crossatt
from model.diffusion.diffusion import GaussianDiffusion1D


def load_guidance_module(repo_root: Path):
    sys.path.insert(0, str(repo_root / "denoising_diffusion_pytorch"))
    guidance_path = repo_root / "denoising_diffusion_pytorch" / "guidance_diffusion_all step_trans.py"
    spec = importlib.util.spec_from_file_location("guidance_trans", str(guidance_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VGCDMStressAdapter(nn.Module):
    def __init__(
        self,
        *,
        height=24,
        width=88,
        channels=1,
        cond_dim=64,
        dim=64,
        num_layers=3,
        timesteps=1000,
        sampling_timesteps=100,
        objective="pred_v",
        beta_schedule="cosine",
        device="cuda:0",
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.seq_length = height * width

        self.denoise_model = Unet1D_crossatt(
            dim=dim,
            num_layers=num_layers,
            dim_mults=(1, 2, 4, 8),
            context_dim=cond_dim,
            channels=channels,
            use_crossatt=True,
        )

        self.diffusion = GaussianDiffusion1D(
            self.denoise_model,
            seq_length=self.seq_length,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            beta_schedule=beta_schedule,
            auto_normalize=False,
        )

        self.num_timesteps = self.diffusion.num_timesteps
        self.sampling_timesteps = self.diffusion.sampling_timesteps
        self.is_ddim_sampling = self.diffusion.is_ddim_sampling

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _to_seq(self, img_2d):
        b, c, h, w = img_2d.shape
        if c != self.channels or h != self.height or w != self.width:
            raise ValueError(
                f"Expected [B,{self.channels},{self.height},{self.width}], got {tuple(img_2d.shape)}"
            )
        return img_2d.reshape(b, c, self.seq_length)

    def _to_map(self, seq_1d):
        b, c, l = seq_1d.shape
        if c != self.channels or l != self.seq_length:
            raise ValueError(
                f"Expected [B,{self.channels},{self.seq_length}], got {tuple(seq_1d.shape)}"
            )
        return seq_1d.reshape(b, c, self.height, self.width)

    @staticmethod
    def _to_context(classes):
        if classes.ndim != 2:
            raise ValueError(f"Expected classes [B,64], got {tuple(classes.shape)}")
        return classes.unsqueeze(1)

    def _extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def _q_sample(self, x_start, t, noise):
        return (
            self._extract(self.diffusion.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _p_losses_with_context(self, x_start, t, context, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self._q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoise_model(x_noisy, t, context=context)

        if self.diffusion.objective == "pred_noise":
            target = noise
        elif self.diffusion.objective == "pred_x0":
            target = x_start
        elif self.diffusion.objective == "pred_v":
            target = self.diffusion.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"Unsupported objective: {self.diffusion.objective}")

        loss = F.smooth_l1_loss(model_out, target, reduction="none")
        loss = loss.mean(dim=(1, 2))
        loss = loss * self._extract(self.diffusion.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, classes, **kwargs):
        _ = args, kwargs
        seq = self._to_seq(img)
        context = self._to_context(classes)
        t = torch.randint(0, self.num_timesteps, (seq.shape[0],), device=seq.device).long()
        seq = seq * 2 - 1
        return self._p_losses_with_context(seq, t, context=context)

    @torch.no_grad()
    def sample(self, classes, cond_scale=1.0, rescaled_phi=0.0):
        _ = cond_scale, rescaled_phi
        context = self._to_context(classes)
        shape = (classes.shape[0], self.channels, self.seq_length)
        seq = self.diffusion.p_sample_loop(shape=shape, cond=context)
        return self._to_map(seq)


def build_data(args, repo_root: Path, device: str):
    from stress_data_utils import load_real_stress_data, make_synthetic_stress_data, normalize_01

    if args.use_synthetic_data:
        d_lf, d_hf, classes_emb = make_synthetic_stress_data(
            num_samples=args.synthetic_num_samples,
            height=args.synthetic_height,
            width=args.synthetic_width,
            cond_dim=args.synthetic_cond_dim,
            seed=args.synthetic_seed,
            device=device
        )
    else:
        d_lf, d_hf, classes_emb, _, _, _ = load_real_stress_data(
            group_id=args.group_id,
            device=device
        )

    d_lf_g, _, _ = normalize_01(d_lf)
    d_hf_g, _, _ = normalize_01(d_hf)
    return d_lf, d_hf, d_lf_g, d_hf_g, classes_emb


def run_pipeline(args):
    repo_root = Path(__file__).resolve().parents[2]
    guidance_mod = load_guidance_module(repo_root)

    # Avoid interactive windows blocking CLI runs.
    if hasattr(guidance_mod, "plt") and hasattr(guidance_mod.plt, "show"):
        guidance_mod.plt.show = lambda *x, **y: None

    Dataset = guidance_mod.Dataset
    Trainer = guidance_mod.Trainer

    device = args.device if torch.cuda.is_available() else "cpu"
    d_lf, d_hf, d_lf_g, d_hf_g, classes_emb = build_data(args, repo_root, device)

    if args.offline_steps < 10:
        raise ValueError("offline_steps must be >= 10 for the shared warmup scheduler.")
    if args.online_steps < 10:
        raise ValueError("online_steps must be >= 10 for the shared warmup scheduler.")

    model = VGCDMStressAdapter(
        height=d_lf_g.shape[2],
        width=d_lf_g.shape[3],
        channels=d_lf_g.shape[1],
        cond_dim=classes_emb.shape[1],
        dim=args.dim,
        num_layers=args.num_layers,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective="pred_v",
        beta_schedule="cosine",
        device=device,
    )

    lf_dataset = Dataset(d_lf_g)
    hf_dataset = Dataset(d_hf_g)

    offline_trainer = Trainer(
        model,
        dataset=lf_dataset,
        train_batch_size=args.batch_size,
        train_lr=args.offline_lr,
        train_num_steps=args.offline_steps,
        save_and_sample_every=args.offline_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        classes=classes_emb,
    )

    if args.stage in ("offline", "both"):
        offline_trainer.train(type=args.offline_tag)

    if args.stage in ("online", "both"):
        online_trainer = Trainer(
            model,
            dataset=hf_dataset,
            train_batch_size=args.batch_size,
            train_lr=args.online_lr,
            train_num_steps=args.online_steps,
            save_and_sample_every=args.online_steps,
            gradient_accumulate_every=2,
            ema_decay=0.995,
            amp=False,
            classes=classes_emb,
        )

        online_trainer.load(args.offline_tag)

        if args.freeze_encoder:
            for name, param in model.named_parameters():
                if "ups" not in name:
                    param.requires_grad = False

        online_trainer.fine(type=args.online_tag)

    if args.do_sample:
        with torch.no_grad():
            pred = model.sample(classes=classes_emb, cond_scale=args.cond_scale)
        out_dir = Path(__file__).resolve().parent / "results_stress"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vgcdm_pred_stress.pt"
        torch.save(pred.cpu(), out_path)
        print(f"saved: {out_path}")
        print(f"output shape: {tuple(pred.shape)}")
        print(f"target shape: {tuple(d_hf.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description="VGCDM stress-field reproduction under unified training framework")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", default="both", choices=["offline", "online", "both"])
    parser.add_argument("--group-id", default="1")

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--offline_steps", type=int, default=300)
    parser.add_argument("--online_steps", type=int, default=300)
    parser.add_argument("--offline_lr", type=float, default=1e-4)
    parser.add_argument("--online_lr", type=float, default=5e-5)

    parser.add_argument("--offline_tag", default="vgcdm_offline")
    parser.add_argument("--online_tag", default="vgcdm_online")
    parser.add_argument("--freeze_encoder", action="store_true")

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
