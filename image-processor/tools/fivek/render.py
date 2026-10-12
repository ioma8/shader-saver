"""Differentiable torch replica of the editor's auto-slider transform.

Mirrors `src/shaders.wgsl` (`contrast_pass` then `tonal_pass`) and the Rust
replica in `src/auto/pipeline.rs`. The training script differentiates through
this, so it must agree with the GPU shader to high precision — otherwise every
learned slider is biased by the difference.

Shapes: `rgb` is `(..., 3)` (channels last). Sliders may be scalars (single
image) or `(N, 1, 1, 1)` (batch), and every intermediate keeps its last axis so
the two broadcast cleanly.

Check it against the Rust golden grid:

    cargo test --release golden_probe            # writes /tmp/auto_golden.txt
    .venv-train/bin/python tools/fivek/render.py --check /tmp/auto_golden.txt
"""

from __future__ import annotations

import argparse

import torch

LUMA = (0.2126, 0.7152, 0.0722)
_LUMA = torch.tensor(LUMA)

# Slider keys the transform reads, in a stable order for the training vector.
SLIDERS = [
    "wb_temp",
    "wb_tint",
    "levels_black",
    "levels_gamma",
    "levels_white",
    "contrast",
    "blacks",
    "shadows",
    "highlights",
    "whites",
    "brightness",
    "saturation",
    "vibrance",
    "exposure",
]

DEFAULT = {name: 0.0 for name in SLIDERS}
DEFAULT["levels_white"] = 255.0
DEFAULT["levels_gamma"] = 1.0


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma, keeping the last axis so it stays rank-consistent."""
    return (rgb * _LUMA.to(rgb)).sum(-1, keepdim=True)


def transform(rgb: torch.Tensor, s: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply the transform to gamma-encoded rgb in [0, 1], shape (..., 3)."""
    x = rgb

    # contrast_pass: exposure
    x = x * torch.exp2(s["exposure"] * 0.4545)

    # white balance (normalised to unit luminance, then applied in gamma space)
    t = s["wb_temp"] * 0.01
    g = s["wb_tint"] * 0.01
    red = 1.0 + 0.30 * t + 0.10 * g
    green = 1.0 - 0.20 * g
    blue = 1.0 - 0.30 * t + 0.10 * g
    norm = red * 0.2126 + green * 0.7152 + blue * 0.0722
    red = torch.clamp(red / norm, min=0.0) ** 0.4545
    green = torch.clamp(green / norm, min=0.0) ** 0.4545
    blue = torch.clamp(blue / norm, min=0.0) ** 0.4545
    # `torch.cat` needs at least one dimension; a scalar slider becomes shape (3).
    red, green, blue = (v.reshape(1) if v.dim() == 0 else v for v in (red, green, blue))
    x = x * torch.cat([red, green, blue], dim=-1)

    # levels + gamma
    lum = _luma(x)
    black = s["levels_black"] / 255.0
    white = s["levels_white"] / 255.0
    span = torch.clamp(white - black, min=0.001)
    l = torch.clamp((lum - black) / span, 0.0, 1.0)
    # pow(0, exponent<1) has an infinite derivative; clamp the base for the
    # gradient but keep the exact value (0) where the base is essentially zero.
    exponent = 1.0 / torch.clamp(s["levels_gamma"], min=0.01)
    base = torch.clamp(l, min=1e-7)
    l = torch.where(l > 1e-7, base**exponent, l)

    # contrast (normalised logistic S-curve); below 0.1% it is the identity, and
    # the guard also keeps k away from zero so the division stays finite.
    cs = s["contrast"] * 0.01
    active = torch.abs(cs) > 0.001
    k = torch.where(active, torch.abs(cs) * 10.0, torch.ones_like(cs))
    s0 = 1.0 / (1.0 + torch.exp(0.5 * k))
    s1 = 1.0 / (1.0 + torch.exp(-0.5 * k))
    positive = (1.0 / (1.0 + torch.exp(-k * (l - 0.5))) - s0) / (s1 - s0)
    u = torch.clamp(s0 + l * (s1 - s0), 1e-5, 1.0 - 1e-5)
    negative = 0.5 - torch.log(1.0 / u - 1.0) / k
    l = torch.where(active, torch.where(cs > 0.0, positive, negative), l)

    scale = _safe_div(l, lum)
    x = torch.clamp(x * scale, 0.0, 1.0)

    # tonal_pass: brightness
    lum = _luma(x)
    bias = 0.5 + s["brightness"] * 0.0025
    lum_b = lum / ((1.0 / bias - 2.0) * (1.0 - lum) + 1.0)
    inv = 1.0 - lum_b

    delta = (
        s["blacks"] * 0.01 * 0.25 * inv**6
        + s["shadows"] * 0.01 * 0.35 * (9.4815 * lum_b * inv**3)
        + s["highlights"] * 0.01 * 0.35 * (9.4815 * lum_b**3 * inv)
        + s["whites"] * 0.01 * 0.25 * lum_b**6
    )
    new_lum = torch.clamp(lum_b + delta, 0.0, 1.0)
    x = torch.clamp(x * _safe_div(new_lum, lum), 0.0, 1.0)

    # vibrance + saturation
    mid = _luma(x)
    chroma = torch.sqrt(((x - mid) ** 2).sum(-1, keepdim=True) + 1e-12)
    sat = 1.0 + s["saturation"] * 0.01
    vib = 1.0 + s["vibrance"] * 0.01 * (1.0 - torch.clamp(chroma * 2.0, 0.0, 1.0))
    return torch.clamp(mid + (x - mid) * sat * vib, 0.0, 1.0)


def _safe_div(numerator: torch.Tensor, lum: torch.Tensor) -> torch.Tensor:
    """`numerator / lum`, but 1.0 where lum is (near) zero, as the shader does."""
    safe = torch.where(lum < 0.001, torch.ones_like(lum), lum)
    return torch.where(lum < 0.001, torch.ones_like(numerator), numerator / safe)


def _t(uniform: float, device="cpu") -> torch.Tensor:
    return torch.tensor(uniform, dtype=torch.float32, device=device)


def scalar_state(values: dict[str, float], device="cpu") -> dict[str, torch.Tensor]:
    merged = dict(DEFAULT)
    merged.update(values)
    return {name: _t(merged[name], device) for name in SLIDERS}


def check(golden_path: str) -> float:
    slider_sets = {
        0: {},
        1: {
            "wb_temp": 22.0,
            "wb_tint": -11.0,
            "levels_black": 12.0,
            "levels_gamma": 1.7,
            "levels_white": 240.0,
            "contrast": 24.0,
            "shadows": 18.0,
            "highlights": -12.0,
            "vibrance": 16.0,
        },
    }
    worst = 0.0
    with open(golden_path) as handle:
        for line in handle:
            index, r, c, out_r, out_g, out_b = line.split()
            rgb = torch.tensor(
                [[int(r) / 8.0, int(c) / 8.0, ((int(r) + int(c)) % 9) / 8.0]],
                dtype=torch.float32,
            )
            expected = torch.tensor([[float(out_r), float(out_g), float(out_b)]])
            got = transform(rgb, scalar_state(slider_sets[int(index)]))
            if not torch.isfinite(got).all():
                raise SystemExit(f"non-finite output for golden line: {line.strip()}")
            worst = max(worst, float((got - expected).abs().max()))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", help="path to the Rust golden file")
    args = parser.parse_args()
    if args.check:
        worst = check(args.check)
        print(f"max abs difference vs golden: {worst:.3e}")
        if worst > 1e-5:
            raise SystemExit("renderer diverges from the shader replica")


if __name__ == "__main__":
    main()
