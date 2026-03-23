from pathlib import Path

import torch


DEFAULT_NUM_SAMPLES = 24
DEFAULT_HEIGHT = 24
DEFAULT_WIDTH = 88
DEFAULT_COND_DIM = 64


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    return script_dir() / "Data"


def default_group_paths(group_id: str):
    root = data_dir()
    return {
        "source": root / f"Data_{group_id}_trans.pt",
        "target": root / f"Data_{group_id}_t_trans.pt",
        "classes": root / f"classes_emb_trans_{group_id}.64.pt",
    }


def normalize_01(x: torch.Tensor):
    x_max = x.max()
    x_min = x.min()
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    return x_norm, x_min, x_max


def make_synthetic_stress_data(
        *,
        num_samples=DEFAULT_NUM_SAMPLES,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        cond_dim=DEFAULT_COND_DIM,
        seed=20260320,
        device="cpu"
):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    classes = torch.randn(num_samples, cond_dim, generator=gen)

    x_axis = torch.linspace(0.0, 1.0, width)
    y_axis = torch.linspace(0.0, 1.0, height)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")

    source = []
    target = []
    for idx in range(num_samples):
        cls = classes[idx]
        amp = 0.6 + 0.2 * torch.tanh(cls[0])
        phase = torch.tanh(cls[1]) * 3.1415926
        slope = 0.2 * torch.tanh(cls[2])
        ridge = torch.exp(-((xx - (0.2 + 0.6 * torch.sigmoid(cls[3]))) ** 2) / 0.02)
        wave = torch.sin((2.0 + torch.sigmoid(cls[4]) * 4.0) * 3.1415926 * xx + phase)
        field = amp * wave * (0.5 + yy) + slope * yy + 0.15 * ridge
        field = field + 0.03 * torch.randn(height, width, generator=gen)
        source.append(field)

        target_field = 1.1 * field + 0.08 * torch.cos(2.0 * 3.1415926 * yy) + 0.02 * torch.randn(height, width, generator=gen)
        target.append(target_field)

    source = torch.stack(source, dim=0).unsqueeze(1).to(device).float()
    target = torch.stack(target, dim=0).unsqueeze(1).to(device).float()
    classes = classes.to(device).float()
    return source, target, classes


def load_real_stress_data(*, group_id: str, device="cpu", source_path=None, target_path=None, classes_path=None):
    defaults = default_group_paths(group_id)
    source_path = Path(source_path) if source_path else defaults["source"]
    target_path = Path(target_path) if target_path else defaults["target"]
    classes_path = Path(classes_path) if classes_path else defaults["classes"]

    for path in (source_path, target_path, classes_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    source = torch.load(str(source_path), map_location=device).float().unsqueeze(1)
    target = torch.load(str(target_path), map_location=device).float().unsqueeze(1)
    classes = torch.load(str(classes_path), map_location=device).float()
    return source, target, classes, source_path, target_path, classes_path
