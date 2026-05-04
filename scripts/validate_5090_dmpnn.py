#!/usr/bin/env python
"""Smoke tests for the single-GPU RTX 5090 D-MPNN environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from molpal.models import model as build_model
from molpal.models.mpnn.predict import predict as predict_mpnn


SMILES = [
    "CC",
    "CCC",
    "CCO",
    "CCN",
    "c1ccccc1",
    "CC(=O)O",
    "CCCl",
    "CCBr",
]
SCORES = [0.12, 0.25, 0.18, 0.05, 0.45, 0.2, 0.09, 0.14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true", help="run the smoke test on CPU")
    parser.add_argument("--skip-cli", action="store_true", help="skip the tiny `molpal run` check")
    parser.add_argument("--epochs", type=int, default=2)
    return parser.parse_args()


def verify_torch(allow_cpu: bool) -> None:
    print(f"[torch] version={torch.__version__}")
    print(f"[torch] cuda={torch.version.cuda}")
    print(f"[torch] cuda_available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA is not available. Re-run with --allow-cpu only for local dry runs.")
        print("[torch] running in CPU fallback mode")
        return

    print(f"[torch] device={torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
    lhs = torch.randn((128, 128), device=device)
    rhs = torch.randn((128, 128), device=device)
    result = lhs @ rhs
    print(f"[torch] matmul_ok shape={tuple(result.shape)} dtype={result.dtype}")


def tiny_dmpnn_train(allow_cpu: bool, epochs: int) -> None:
    use_gpu = torch.cuda.is_available() and not allow_cpu
    dmpn = build_model(
        "dmpn",
        conf_method="mve",
        test_batch_size=4,
        batch_size=4,
        epochs=epochs,
        ncpu=0,
        precision=32,
        model_seed=7,
    )
    y = np.asarray(SCORES, dtype=np.float32)
    dmpn.train(SMILES, y, retrain=True)

    preds = predict_mpnn(
        dmpn.model.model,
        SMILES[:4],
        batch_size=4,
        ncpu=0,
        uncertainty="mve",
        scaler=dmpn.model.scaler,
        use_gpu=use_gpu,
        disable=True,
    )
    means, variances = preds[:, 0], preds[:, 1]
    print(f"[dmpn] means_shape={means.shape} vars_shape={variances.shape}")


def write_tiny_cli_inputs(root: Path) -> Path:
    library_path = root / "library.csv"
    library_path.write_text("smiles\n" + "\n".join(SMILES) + "\n", encoding="ascii")

    lookup_path = root / "scores.csv"
    lookup_lines = ["smiles,score"] + [f"{smi},{score}" for smi, score in zip(SMILES, SCORES)]
    lookup_path.write_text("\n".join(lookup_lines) + "\n", encoding="ascii")

    objective_path = root / "lookup.ini"
    objective_path.write_text(
        f"path = {lookup_path}\nsmiles-col = 0\nscore-col = 1\n", encoding="ascii"
    )

    config_path = root / "tiny.ini"
    config_path.write_text(
        "\n".join(
            [
                "[general]",
                f"output-dir = {root / 'output'}",
                "ncpu = 0",
                "--write-final",
                "--retrain-from-scratch",
                "",
                "[pool]",
                f"library = {library_path}",
                "invalid-idxs = []",
                "",
                "[encoder]",
                "fingerprint = pair",
                "length = 256",
                "radius = 2",
                "",
                "[acquisition]",
                "metric = greedy",
                "init-size = 2",
                "batch-sizes = 1",
                "model = dmpn",
                "conf-method = mve",
                "test-batch-size = 4",
                "",
                "[objective]",
                "objective = lookup",
                f"objective-config = {objective_path}",
                "--minimize",
                "",
                "[stopping]",
                "top-k = 1",
                "window-size = 2",
                "delta = 0.0",
                "max-iters = 1",
                "",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    return config_path


def tiny_cli_run() -> None:
    with tempfile.TemporaryDirectory(prefix="molpal_5090_validate_") as tmpdir:
        tmp_path = Path(tmpdir)
        config_path = write_tiny_cli_inputs(tmp_path)

        env = dict(os.environ)
        env.pop("ip_head", None)
        completed = subprocess.run(
            [sys.executable, "-m", "molpal.cli.main", "run", "--config", str(config_path)],
            cwd=tmp_path,
            env=env,
            check=True,
        )
        print(f"[cli] returncode={completed.returncode}")


def main() -> None:
    args = parse_args()
    verify_torch(args.allow_cpu)
    tiny_dmpnn_train(args.allow_cpu, args.epochs)
    if not args.skip_cli:
        tiny_cli_run()
    print("[done] validation completed")


if __name__ == "__main__":
    main()
