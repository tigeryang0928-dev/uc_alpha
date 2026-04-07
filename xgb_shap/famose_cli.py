"""CLI alias for FAMOSE (same as repo-root famose_verify.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "xgb_shap"))
sys.path.insert(0, str(ROOT / "backtest_2"))

from famose.runner import run_famose  # noqa: E402
from train_config import load_train_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="FAMOSE DSL feature discovery")
    p.add_argument("--config", default=str(ROOT / "xgb_shap" / "config.yaml"))
    p.add_argument(
        "--famose-config",
        default=str(ROOT / "xgb_shap" / "famose" / "config.yaml"),
        help="FAMOSE-only YAML",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_train_config(args.config, famose_config_path=args.famose_config)
    run_famose(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
