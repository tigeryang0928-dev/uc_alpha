"""CLI entry: XGBoost + SHAP training (loads xgb_shap/config.yaml by default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "xgb_shap"))

from train_config import load_train_config  # noqa: E402
from train_xgboost_shap import run  # noqa: E402


def main() -> None:
    default_config = str(ROOT / "xgb_shap" / "config.yaml")

    p = argparse.ArgumentParser(
        description="XGBoost + SHAP using config.yaml and backtest_2 Handler data"
    )
    p.add_argument("--config", default=default_config, help="YAML settings file")
    p.add_argument("--data-dir", default=None, help="Override config data.handler_dir")
    p.add_argument("--start-date", default=None, help="Override config data.start_date")
    p.add_argument(
        "--target",
        choices=("intraday", "overnight", "day"),
        default=None,
        help="Override config target",
    )
    p.add_argument("--train-frac", type=float, default=None, help="Override config splits.train_frac")
    p.add_argument("--val-frac", type=float, default=None, help="Override config splits.val_frac")
    p.add_argument("--out-dir", default=None, help="Override config output.dir")
    p.add_argument("--shap-sample", type=int, default=None, help="Override config shap.sample")
    p.add_argument("--seed", type=int, default=None, help="Override config random_state")
    args = p.parse_args()

    cfg = load_train_config(args.config)
    if args.data_dir is not None:
        cfg.data.handler_dir = args.data_dir
    if args.start_date is not None:
        cfg.data.start_date = args.start_date if args.start_date else None
    if args.target is not None:
        cfg.target = args.target
    if args.train_frac is not None:
        cfg.splits.train_frac = args.train_frac
    if args.val_frac is not None:
        cfg.splits.val_frac = args.val_frac
    if args.out_dir is not None:
        cfg.output.dir = args.out_dir
    if args.shap_sample is not None:
        cfg.shap.sample = args.shap_sample
    if args.seed is not None:
        cfg.random_state = args.seed

    run(cfg)


if __name__ == "__main__":
    main()
