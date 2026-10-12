"""Train the Auto slider-residual network on FiveK and export ONNX.

The seed (`src/auto.rs`) picks a base slider vector per photo; this network
predicts a bounded correction on top of it. Training renders through the
differentiable replica in `render.py` and matches the expert retouch, so the
network learns only what the slider family can actually represent.

    .venv-train/bin/python tools/fivek/train.py --epochs 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from render import SLIDERS, transform

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "datasets" / "fivek" / "pairs"

# How far the network may move each seeded slider. Deliberately tight: the
# FiveK inputs are flat and dark, so an unconstrained model learns a large
# "darken + saturate" correction that wrecks already-good images out of
# distribution. These caps keep it a polish rather than an overhaul.
RESIDUAL_SCALE = {
    "wb_temp": 16.0,
    "wb_tint": 16.0,
    "levels_black": 32.0,
    "levels_gamma": 0.6,
    "levels_white": 32.0,
    "contrast": 20.0,
    "blacks": 24.0,
    "shadows": 24.0,
    "highlights": 24.0,
    "whites": 24.0,
    "brightness": 28.0,
    "saturation": 14.0,
    "vibrance": 12.0,
    "exposure": 0.32,
}

# Rough per-slider magnitude, used to hand the seed to the network normalised.
SEED_NORM = {
    "wb_temp": 50.0,
    "wb_tint": 50.0,
    "levels_black": 128.0,
    "levels_gamma": 1.5,
    "levels_white": 128.0,
    "contrast": 50.0,
    "blacks": 60.0,
    "shadows": 60.0,
    "highlights": 60.0,
    "whites": 60.0,
    "brightness": 70.0,
    "saturation": 35.0,
    "vibrance": 30.0,
    "exposure": 0.8,
}

LIMITS = {
    "wb_temp": (-100.0, 100.0),
    "wb_tint": (-100.0, 100.0),
    "levels_black": (0.0, 254.0),
    "levels_gamma": (0.1, 5.0),
    "levels_white": (1.0, 255.0),
    "contrast": (-100.0, 100.0),
    "blacks": (-100.0, 100.0),
    "shadows": (-100.0, 100.0),
    "highlights": (-100.0, 100.0),
    "whites": (-100.0, 100.0),
    "brightness": (-100.0, 100.0),
    "saturation": (-100.0, 100.0),
    "vibrance": (-100.0, 100.0),
    "exposure": (-3.0, 3.0),
}

SCALE = torch.tensor([RESIDUAL_SCALE[name] for name in SLIDERS])
NORM = torch.tensor([SEED_NORM[name] for name in SLIDERS])
LOW = torch.tensor([LIMITS[name][0] for name in SLIDERS])
HIGH = torch.tensor([LIMITS[name][1] for name in SLIDERS])


def load_seeds(path: Path) -> dict[str, np.ndarray]:
    seeds: dict[str, np.ndarray] = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            identifier = row.pop("id")
            seeds[identifier] = np.array(
                [float(row[name]) for name in SLIDERS], dtype=np.float32
            )
    return seeds


def to_sliders(vector: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: vector[:, i].reshape(-1, 1, 1, 1) for i, name in enumerate(SLIDERS)}


class Pairs(torch.utils.data.Dataset):
    def __init__(self, pairs: Path, ids: list[str], seeds: dict[str, np.ndarray], size: int):
        self.pairs = pairs
        self.ids = ids
        self.seeds = seeds
        self.size = size

    def __len__(self) -> int:
        return len(self.ids)

    def _image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)  # (H, W, 3)

    def __getitem__(self, index: int):
        identifier = self.ids[index]
        return (
            self._image(self.pairs / f"{identifier}_in.jpg"),
            self._image(self.pairs / f"{identifier}_gt.jpg"),
            torch.from_numpy(self.seeds[identifier]),
        )


class ResidualNet(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.body = nn.Sequential(
            block(3, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(width * 4 + len(SLIDERS), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(SLIDERS)),
        )
        # Start exactly at the seed: learn the correction only.
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, image: torch.Tensor, seed_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.body(image), seed_norm], dim=1))


def apply_residual(seed: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    scaled = seed + residual * SCALE.to(residual)
    return torch.clamp(scaled, LOW.to(residual), HIGH.to(residual))


def render_with(input_hwc, seed, residual):
    return transform(input_hwc, to_sliders(apply_residual(seed, residual)))


def run_epochs(model, loader, optimizer, device, epochs, log_every, reg):
    model.train()
    step = 0
    for epoch in range(epochs):
        total, count = 0.0, 0
        for input_hwc, target_hwc, seed in loader:
            input_hwc = input_hwc.to(device)
            target_hwc = target_hwc.to(device)
            seed = seed.to(device)
            seed_norm = seed / NORM.to(seed)
            optimizer.zero_grad(set_to_none=True)
            residual = model(input_hwc.permute(0, 3, 1, 2).contiguous(), seed_norm)
            rendered = render_with(input_hwc, seed, residual)
            # L1 against the expert, plus a pull toward zero so the model only
            # corrects where it clearly helps (its inputs are out of domain).
            loss = (rendered - target_hwc).abs().mean() + reg * (residual**2).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * input_hwc.shape[0]
            count += input_hwc.shape[0]
            step += 1
            if log_every and step % log_every == 0:
                print(f"epoch {epoch} step {step} loss {total / count:.4f}", flush=True)
        print(f"epoch {epoch} done, mean loss {total / max(count, 1):.4f}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    seed_mae, learned_mae, count = 0.0, 0.0, 0
    for input_hwc, target_hwc, seed in loader:
        input_hwc = input_hwc.to(device)
        target_hwc = target_hwc.to(device)
        seed = seed.to(device)
        residual = model(input_hwc.permute(0, 3, 1, 2).contiguous(), seed / NORM.to(seed))
        seed_render = render_with(input_hwc, seed, torch.zeros_like(residual))
        learned_render = render_with(input_hwc, seed, residual)
        seed_mae += float((seed_render - target_hwc).abs().sum())
        learned_mae += float((learned_render - target_hwc).abs().sum())
        count += target_hwc.numel()
    return seed_mae / count, learned_mae / count


@torch.no_grad()
def save_strip(model, pairs, ids, seeds, device, size, out: Path):
    """in | seed | learned | target, side by side, for eyeballing."""
    model.eval()
    rows = []
    for identifier in ids:
        input_image = Image.open(pairs / f"{identifier}_in.jpg").convert("RGB").resize((size, size))
        target_image = Image.open(pairs / f"{identifier}_gt.jpg").convert("RGB").resize((size, size))
        input_hwc = torch.from_numpy(np.asarray(input_image, dtype=np.float32) / 255.0)[None].to(device)
        seed = torch.from_numpy(seeds[identifier])[None].to(device)
        residual = model(input_hwc.permute(0, 3, 1, 2).contiguous(), seed / NORM.to(seed))
        seed_render = render_with(input_hwc, seed, torch.zeros_like(residual))[0]
        learned = render_with(input_hwc, seed, residual)[0]
        to_pil = lambda t: Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
        rows.append((input_image, to_pil(seed_render), to_pil(learned), target_image))
    width, height = size * 4 + 3 * 8, size * len(rows)
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    for row, images in enumerate(rows):
        for col, image in enumerate(images):
            canvas.paste(image, (col * (size + 8), row * size))
    canvas.save(out)
    print(f"wrote {out}")


def make_tract_compatible(path: Path) -> None:
    """`tract` cannot parse the opset-18 `ReduceMean` (axes as an input), so fold
    the axes back into an attribute and pin the opset to 13."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path)
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    nodes = []
    for node in model.graph.node:
        attrs = {a.name: a for a in node.attribute}
        if node.op_type == "ReduceMean" and len(node.input) == 2:
            axes = [int(a) for a in initializers[node.input[1]].tolist()]
            keepdims = int(attrs["keepdims"].i) if "keepdims" in attrs else 1
            nodes.append(
                helper.make_node(
                    "ReduceMean", [node.input[0]], list(node.output), axes=axes, keepdims=keepdims
                )
            )
        elif node.op_type == "Reshape":
            nodes.append(helper.make_node("Reshape", list(node.input), list(node.output)))
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.opset_import[0].version = 13
    # `onnx.save` inlines the weights; drop the exporter's external-data file.
    onnx.save(model, path)
    external = Path(str(path) + ".data")
    if external.exists():
        external.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=0.05, help="residual L2 penalty")
    parser.add_argument("--limit", type=int, default=0, help="use only N pairs (debug)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "auto_residual.onnx")
    parser.add_argument("--strip", type=Path, default=None, help="write a before/after strip")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    seeds = load_seeds(args.pairs / "seeds.csv")
    ids = [line.strip() for line in (args.pairs / "index.txt").read_text().split() if line.strip()]
    ids = [i for i in ids if i in seeds]
    if args.limit:
        ids = ids[: args.limit]
    validation = ids[::14]
    seen = set(validation)
    training = [i for i in ids if i not in seen]
    print(f"{len(training)} train / {len(validation)} validation pairs")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    common = {"num_workers": args.workers, "persistent_workers": args.workers > 0}
    train_loader = torch.utils.data.DataLoader(
        Pairs(args.pairs, training, seeds, args.size),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = torch.utils.data.DataLoader(
        Pairs(args.pairs, validation, seeds, args.size),
        batch_size=args.batch,
        shuffle=False,
        **common,
    )

    model = ResidualNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_epochs(model, train_loader, optimizer, device, args.epochs, args.log_every, args.reg)

    before, after = evaluate(model, val_loader, device)
    print(f"validation MAE: seed {before:.4f} -> learned {after:.4f}")

    if args.strip:
        save_strip(model, args.pairs, validation[:6], seeds, device, args.size, args.strip)

    model.eval().cpu()
    dummy_image = torch.zeros(1, 3, args.size, args.size)
    dummy_seed = torch.zeros(1, len(SLIDERS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_image, dummy_seed),
        args.out,
        input_names=["image", "seed"],
        output_names=["residual"],
        opset_version=18,
    )
    make_tract_compatible(args.out)
    torch.save(model.state_dict(), args.out.with_suffix(".pt"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
