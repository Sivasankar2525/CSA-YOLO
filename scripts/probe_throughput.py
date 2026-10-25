"""Short training probe to measure per-epoch wall time on this machine.

Informs the local-MPS vs cloud-GPU decision (Week 4) and doubles as the
does-it-train verification for the restored ICSD-YOLO detector.

Usage:
    python scripts/probe_throughput.py --model yolo11n --epochs 2 --device mps
    python scripts/probe_throughput.py --model icsd --epochs 2 --device mps
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# The restored ICSD focus blocks use adaptive pooling with non-divisible
# sizes, unimplemented on MPS (pytorch#96056); fall back to CPU for that op.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csa_yolo.utils.logging_utils import setup_logger  # noqa: E402
from csa_yolo.utils.reproducibility import set_seed  # noqa: E402

logger = logging.getLogger("probe_throughput")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["yolo11n", "icsd"], required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "datasets/csa_ppe_v1_clean/data.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger("INFO")
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    if args.model == "yolo11n":
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # pretrained baseline path
    else:
        from training.icsd import load_icsd_yolo

        model = load_icsd_yolo(scale="n", arch="x")  # from scratch

    run_name = f"probe_{args.model}_{args.device}"
    start = time.perf_counter()
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.out),
        name=run_name,
        exist_ok=True,
        verbose=True,
        val=True,
        plots=False,
    )
    wall = time.perf_counter() - start

    record = {
        "model": args.model,
        "device": args.device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "wall_seconds": round(wall, 1),
        "seconds_per_epoch": round(wall / args.epochs, 1),
    }
    out_file = args.out / f"{run_name}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2) + "\n")
    logger.info("Probe result: %s", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
