#!/usr/bin/env python3
"""Export CanonCGT's LUT predictors without the unsupported 5-D grid sampler."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml


class LutOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def estimate(estimator, image, condition):
        resized = estimator.downsample(image)
        tokens = estimator.feature_extractor(resized, condition)
        identity = estimator.identity[None].repeat(len(image), 1, 1, 1, 1)
        return estimator.generate_lut(identity, tokens, condition)

    @staticmethod
    def render(image, lut):
        # LUT is [1, C, B, G, R]. Keep this as primitive tensor operations so
        # ONNX runtimes do not need PyTorch's unsupported 5-D grid_sample.
        n = lut.shape[-1]
        coordinates = image.clamp(0, 1) * (n - 1)
        lower = coordinates.floor().long()
        upper = (lower + 1).clamp(max=n - 1)
        fraction = coordinates - lower
        flat = lut.reshape(1, 3, -1)

        def corner(r, g, b):
            index = b * n * n + g * n + r
            return flat.gather(2, index.reshape(1, 1, -1).expand(1, 3, -1)).reshape_as(image)

        r0, g0, b0 = lower[:, 0], lower[:, 1], lower[:, 2]
        r1, g1, b1 = upper[:, 0], upper[:, 1], upper[:, 2]
        wr, wg, wb = fraction[:, 0:1], fraction[:, 1:2], fraction[:, 2:3]
        c00 = corner(r0, g0, b0) * (1 - wr) + corner(r1, g0, b0) * wr
        c01 = corner(r0, g0, b1) * (1 - wr) + corner(r1, g0, b1) * wr
        c10 = corner(r0, g1, b0) * (1 - wr) + corner(r1, g1, b0) * wr
        c11 = corner(r0, g1, b1) * (1 - wr) + corner(r1, g1, b1) * wr
        c0 = c00 * (1 - wg) + c10 * wg
        c1 = c01 * (1 - wg) + c11 * wg
        return c0 * (1 - wb) + c1 * wb

    def forward(self, image, reference):
        source_style = self.model.Embedding_Net(image)
        target_style = self.model.Embedding_Net(reference)
        canonical = self.estimate(self.model.Canonicalizer, image, source_style)
        restyle = self.estimate(self.model.Restyler, self.render(image, canonical), target_style)
        return canonical, restyle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="CanonCGT checkout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from models.networks.SSL_training import CanonCGT_SSL

    config_path = args.source / "configs/Stage3_SSL_training_Flickr2K_PPR10K_LSDIR.yaml"
    config = SimpleNamespace(**yaml.safe_load(config_path.read_text()))
    model = CanonCGT_SSL(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    wrapper = LutOnly(model.eval())
    sample = torch.rand(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        canonical, restyle = wrapper(sample, sample)
        native = model(sample, sample)
        assert torch.max(torch.abs(canonical - native["canonicalize_LUT"])) < 1e-5
        assert torch.max(torch.abs(restyle - native["restylize_LUT"])) < 1e-4
        torch.onnx.export(
            wrapper,
            (sample, sample),
            args.output,
            input_names=["img", "ref"],
            output_names=["canonicalize_lut", "restylize_lut"],
            opset_version=17,
            dynamo=False,
        )


if __name__ == "__main__":
    main()
