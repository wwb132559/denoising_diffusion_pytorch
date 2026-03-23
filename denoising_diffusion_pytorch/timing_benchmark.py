"""
timing_benchmark.py
-------------------
生成论文可核验的推断时延-精度表：
  - 硬件信息（CPU / GPU / 显存）
  - DDIM steps = 10 / 20 / 50 / 100
  - 每种设置重复 30 次，统计 mean ± std
  - 精度指标：R2 / RRMSE / RMAE（沿用当前项目口径）

输出文件：
  - ./results/ddim_latency_accuracy_table.csv
  - ./results/ddim_latency_accuracy_table.md
"""

import argparse
import contextlib
import csv
import io
import importlib.util
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from stress_data_utils import load_real_stress_data, make_synthetic_stress_data, normalize_01

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_fname = os.path.join(os.path.dirname(__file__), "guidance_diffusion_all step_trans.py")
_spec = importlib.util.spec_from_file_location("guidance_mod", _fname)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Unet = _mod.Unet
GaussianDiffusion = _mod.GaussianDiffusion
Trainer = _mod.Trainer


def denominator(target):
    target_mean = torch.mean(target)
    tot = torch.sum((target - target_mean) ** 2)
    return torch.sqrt(torch.mean(tot))


def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    return 1 - ss_res / ss_tot


def rmae(output, target):
    de = denominator(target)
    return torch.abs(torch.max(target - output)) / de


def rrmse(output, target):
    mse = nn.MSELoss()
    return torch.sqrt(mse(target, output))


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt_pm(mean_val, std_val, digits=4):
    return f"{mean_val:.{digits}f}±{std_val:.{digits}f}"


def get_cpu_name():
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore"
        ).strip()
        if out:
            return out
    except Exception:
        pass

    cpu_name = platform.processor().strip()
    if cpu_name:
        return cpu_name

    try:
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "name"],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        lines = [line.strip() for line in out.splitlines() if line.strip() and line.strip().lower() != "name"]
        if lines:
            return lines[0]
    except Exception:
        pass

    return platform.uname().processor or "Unknown CPU"


def get_hardware_info(device):
    info = {
        "cpu": get_cpu_name(),
        "gpu": "N/A",
        "gpu_mem_gb": "N/A"
    }

    if torch.cuda.is_available() and "cuda" in device:
        gpu_idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(gpu_idx)
        info["gpu"] = prop.name
        info["gpu_mem_gb"] = f"{prop.total_memory / (1024 ** 3):.2f}"

    return info


def build_model(classes_emb, channels, height, width, device, sampling_steps=100):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        classes_emb=classes_emb,
        cond_drop_prob=0.5,
        channels=channels,
        resnet_block_groups=4,
        attn_dim_head=32,
        attn_heads=4
    )
    diffusion = GaussianDiffusion(
        model,
        height=height,
        width=width,
        timesteps=1000,
        sampling_timesteps=sampling_steps,
        objective='pred_v'
    ).to(device)
    return model, diffusion


def load_or_train_online_checkpoint(
    diffusion,
    classes_emb,
    lf_data,
    hf_data,
    ckpt_type,
    train_num_steps,
    device
):
    print(f"加载权重: {ckpt_type}")
    trainer = Trainer(
        diffusion,
        dataset=_mod.Dataset(hf_data),
        train_batch_size=classes_emb.shape[0],
        train_lr=5e-5,
        train_num_steps=train_num_steps,
        save_and_sample_every=train_num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        classes=classes_emb
    )

    try:
        trainer.load(ckpt_type)
        print(f"已加载 results/model-all step-trans-{ckpt_type}.pt")
        return
    except Exception as exc:
        print(f"未找到 {ckpt_type} 权重，开始自动训练（一次性）: {exc}")

    offline = Trainer(
        diffusion,
        dataset=_mod.Dataset(lf_data),
        train_batch_size=classes_emb.shape[0],
        train_lr=1e-4,
        train_num_steps=train_num_steps,
        save_and_sample_every=train_num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        classes=classes_emb
    )
    offline.train(type='bench_offline')

    trainer.load('bench_offline')
    for name, param in diffusion.model.named_parameters():
        if 'ups' not in name:
            param.requires_grad = False
    trainer.fine(type=ckpt_type)


def set_ddim_steps(diffusion, steps):
    diffusion.sampling_timesteps = int(steps)
    diffusion.is_ddim_sampling = diffusion.sampling_timesteps < diffusion.num_timesteps


def run_ddim_latency_accuracy_table(
    diffusion,
    classes_emb,
    d_target,
    min_t,
    max_t,
    ddim_steps,
    repeats,
    cond_scale,
    warmup,
    device
):
    records = []
    batch_size = classes_emb.shape[0]

    print("=" * 72)
    print(f"开始统计：DDIM steps={ddim_steps}, repeats={repeats}, warmup={warmup}")

    for steps in ddim_steps:
        set_ddim_steps(diffusion, steps)
        print(f"\n[DDIM={steps}] 预热 {warmup} 次 ...")

        for _ in range(warmup):
            with torch.no_grad():
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _ = diffusion.sample(classes=classes_emb, cond_scale=cond_scale)

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()

        time_runs = []
        r2_runs = []
        rrmse_runs = []
        rmae_runs = []

        print(f"[DDIM={steps}] 正式重复 {repeats} 次 ...")
        for i in range(repeats):
            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    sampled = diffusion.sample(classes=classes_emb, cond_scale=cond_scale)

            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - t0
            pred = sampled * (max_t - min_t) + min_t

            r2_val = r2_score(pred[:, 0, :, :], d_target[:, 0, :, :]).item()
            rrmse_val = rrmse(pred, d_target).item()
            rmae_val = rmae(pred, d_target).item()

            time_runs.append(elapsed)
            r2_runs.append(r2_val)
            rrmse_runs.append(rrmse_val)
            rmae_runs.append(rmae_val)

            print(
                f"  run {i + 1:02d}/{repeats} | time={elapsed:.3f}s | "
                f"R2={r2_val:.4f} RRMSE={rrmse_val:.4f} RMAE={rmae_val:.4f}"
            )

        t_mean, t_std = mean_std(time_runs)
        ps_mean, ps_std = mean_std([t / batch_size for t in time_runs])
        r2_mean, r2_std = mean_std(r2_runs)
        rrmse_mean, rrmse_std = mean_std(rrmse_runs)
        rmae_mean, rmae_std = mean_std(rmae_runs)

        records.append({
            "ddim_steps": steps,
            "batch_size": batch_size,
            "time_batch_mean_s": t_mean,
            "time_batch_std_s": t_std,
            "time_sample_mean_s": ps_mean,
            "time_sample_std_s": ps_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "rrmse_mean": rrmse_mean,
            "rrmse_std": rrmse_std,
            "rmae_mean": rmae_mean,
            "rmae_std": rmae_std,
        })

        print(
            f"[DDIM={steps}] 汇总：batch {fmt_pm(t_mean, t_std, 3)} s, "
            f"sample {fmt_pm(ps_mean, ps_std, 4)} s, "
            f"R2 {fmt_pm(r2_mean, r2_std)}, "
            f"RRMSE {fmt_pm(rrmse_mean, rrmse_std)}, "
            f"RMAE {fmt_pm(rmae_mean, rmae_std)}"
        )

    return records


def save_csv(records, out_csv, hardware):
    fieldnames = [
        "cpu", "gpu", "gpu_mem_gb", "batch_size", "ddim_steps", "repeats",
        "time_batch_mean_s", "time_batch_std_s",
        "time_sample_mean_s", "time_sample_std_s",
        "r2_mean", "r2_std", "rrmse_mean", "rrmse_std", "rmae_mean", "rmae_std"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {
                "cpu": hardware["cpu"],
                "gpu": hardware["gpu"],
                "gpu_mem_gb": hardware["gpu_mem_gb"],
                "batch_size": rec["batch_size"],
                "ddim_steps": rec["ddim_steps"],
                "repeats": 30,
                "time_batch_mean_s": rec["time_batch_mean_s"],
                "time_batch_std_s": rec["time_batch_std_s"],
                "time_sample_mean_s": rec["time_sample_mean_s"],
                "time_sample_std_s": rec["time_sample_std_s"],
                "r2_mean": rec["r2_mean"],
                "r2_std": rec["r2_std"],
                "rrmse_mean": rec["rrmse_mean"],
                "rrmse_std": rec["rrmse_std"],
                "rmae_mean": rec["rmae_mean"],
                "rmae_std": rec["rmae_std"],
            }
            writer.writerow(row)


def save_markdown(records, out_md, hardware, repeats, cond_scale):
    lines = []
    lines.append("# DDIM 推断时延与精度可核验表")
    lines.append("")
    lines.append("## 硬件与实验设置")
    lines.append("")
    lines.append(f"- CPU: {hardware['cpu']}")
    lines.append(f"- GPU: {hardware['gpu']}")
    lines.append(f"- 显存: {hardware['gpu_mem_gb']} GB")
    lines.append(f"- Batch size: {records[0]['batch_size']}")
    lines.append(f"- 重复次数: {repeats} 次/设置")
    lines.append(f"- cond_scale: {cond_scale}")
    lines.append("")
    lines.append("## 表：DDIM步数-时延-精度（mean±std）")
    lines.append("")
    lines.append("| DDIM steps | 推断时间 / batch (s) | 推断时间 / sample (s) | R2 | RRMSE | RMAE |")
    lines.append("|---:|---:|---:|---:|---:|---:|")

    for rec in records:
        lines.append(
            f"| {rec['ddim_steps']} | "
            f"{fmt_pm(rec['time_batch_mean_s'], rec['time_batch_std_s'], 3)} | "
            f"{fmt_pm(rec['time_sample_mean_s'], rec['time_sample_std_s'], 4)} | "
            f"{fmt_pm(rec['r2_mean'], rec['r2_std'])} | "
            f"{fmt_pm(rec['rrmse_mean'], rec['rrmse_std'])} | "
            f"{fmt_pm(rec['rmae_mean'], rec['rmae_std'])} |"
        )

    best = min(records, key=lambda x: x["time_batch_mean_s"])
    lines.append("")
    lines.append("## 结论建议")
    lines.append("")
    lines.append(
        f"- 速度最快设置：DDIM {best['ddim_steps']}（{best['time_batch_mean_s']:.3f}s/batch）。"
    )
    lines.append(
        "- 若论文最终采用 DDIM=100，请强调其在精度（R2更高、RRMSE/RMAE更低）与时延之间的折中最优。"
    )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="DDIM 时延与精度表生成")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cond_scale", type=float, default=8.0)
    parser.add_argument("--ddim_steps", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--ckpt_type", default="bench_online")
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--group-id", default="1")
    parser.add_argument("--use-synthetic-data", action="store_true")
    parser.add_argument("--synthetic-seed", type=int, default=20260320)
    parser.add_argument("--synthetic-num-samples", type=int, default=24)
    parser.add_argument("--synthetic-height", type=int, default=24)
    parser.add_argument("--synthetic-width", type=int, default=88)
    parser.add_argument("--synthetic-cond-dim", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("加载数据 ...")
    if args.use_synthetic_data:
        d_lf, d_target, classes_emb = make_synthetic_stress_data(
            num_samples=args.synthetic_num_samples,
            height=args.synthetic_height,
            width=args.synthetic_width,
            cond_dim=args.synthetic_cond_dim,
            seed=args.synthetic_seed,
            device=device
        )
    else:
        d_lf, d_target, classes_emb, _, _, _ = load_real_stress_data(
            group_id=args.group_id,
            device=device
        )

    d_lf_g, _, _ = normalize_01(d_lf)
    d_target_g, min_t, max_t = normalize_01(d_target)

    channels = d_lf_g.shape[1]
    height = d_lf_g.shape[2]
    width = d_lf_g.shape[3]

    print(f"LF data   : {tuple(d_lf_g.shape)}")
    print(f"HF data   : {tuple(d_target_g.shape)}")
    print(f"classes   : {tuple(classes_emb.shape)}")
    print(f"device    : {device}")
    print(f"data mode : {'synthetic benchmark' if args.use_synthetic_data else 'real experiment data'}")

    hardware = get_hardware_info(device)
    print(f"CPU       : {hardware['cpu']}")
    print(f"GPU       : {hardware['gpu']}")
    print(f"显存(GB)  : {hardware['gpu_mem_gb']}")

    model, diffusion = build_model(classes_emb, channels, height, width, device, sampling_steps=100)

    load_or_train_online_checkpoint(
        diffusion=diffusion,
        classes_emb=classes_emb,
        lf_data=d_lf_g,
        hf_data=d_target_g,
        ckpt_type=args.ckpt_type,
        train_num_steps=args.train_steps,
        device=device
    )

    records = run_ddim_latency_accuracy_table(
        diffusion=diffusion,
        classes_emb=classes_emb,
        d_target=d_target,
        min_t=min_t,
        max_t=max_t,
        ddim_steps=args.ddim_steps,
        repeats=args.repeats,
        cond_scale=args.cond_scale,
        warmup=args.warmup,
        device=device
    )

    out_dir = Path("./results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ddim_latency_accuracy_table.csv"
    out_md = out_dir / "ddim_latency_accuracy_table.md"

    save_csv(records, out_csv, hardware)
    save_markdown(records, out_md, hardware, args.repeats, args.cond_scale)

    print("\n" + "=" * 72)
    print("结果已写出：")
    print(f"- {out_csv}")
    print(f"- {out_md}")
    print("=" * 72)


if __name__ == "__main__":
    main()
