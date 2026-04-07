"""
Verify FAMOSE (DSL feature discovery + XGBoost validation) without running SHAP training.

This entry point only loads Handler data, stacks the panel like training, runs the FAMOSE
loop (or a single --dry-run step), and writes artifacts under ``output.dir``:

- ``famose_runs_<timestamp>/famose_trace.jsonl`` — one JSON object per step
- ``famose_runs_<timestamp>/famose_summary.json`` — accepted features + mRMR order
- ``famose_runs_<timestamp>/discovered_features.yaml`` — pasteable ``features:`` list

Examples (from repository root)::

    python famose_verify.py --dry-run
    python famose_verify.py --config xgb_shap/config.yaml --show-trace 20

FAMOSE hyperparameters and LLM settings live in ``xgb_shap/famose/config.yaml`` by default
(override with ``--famose-config``). Training data / target / seed ``features`` still come from
``--config`` (main ``xgb_shap/config.yaml``).

LLM keys: set ``OPENAI_API_KEY`` for ``llm_provider: openai`` (``llm_base_url`` + ``/chat/completions``),
or ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` for ``llm_provider: gemini`` (Google AI Studio REST).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "xgb_shap"))
sys.path.insert(0, str(ROOT / "backtest_2"))

from famose.runner import run_famose  # noqa: E402
from train_config import load_train_config  # noqa: E402

_DEFAULT_FAMOSE_YAML = ROOT / "xgb_shap" / "famose" / "config.yaml"


def _print_trace_tail(log_path: Path, n: int) -> None:
    if not log_path.is_file():
        print(f"No trace file at {log_path}")
        return
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    tail = lines[-n:] if n > 0 else lines
    print(f"--- last {len(tail)} line(s) of {log_path.name} ---")
    for line in tail:
        try:
            obj = json.loads(line)
            print(json.dumps(obj, indent=2, ensure_ascii=False)[:2000])
        except json.JSONDecodeError:
            print(line[:2000])
        print("-" * 40)


def main() -> None:
    default_config = str(ROOT / "xgb_shap" / "config.yaml")
    p = argparse.ArgumentParser(
        description="Verify FAMOSE feature discovery (no SHAP full train). See module docstring."
    )
    p.add_argument("--config", default=default_config, help="Main training YAML (data, target, features)")
    p.add_argument(
        "--famose-config",
        default=str(_DEFAULT_FAMOSE_YAML),
        help="FAMOSE-only YAML (default: xgb_shap/famose/config.yaml)",
    )
    p.add_argument("--dry-run", action="store_true", help="No LLM; one canned proposal + mRMR export")
    p.add_argument(
        "--show-trace",
        type=int,
        metavar="N",
        default=0,
        help="After run, print last N JSONL records from famose_trace.jsonl",
    )
    p.add_argument("--data-dir", default=None, help="Override config data.handler_dir")
    p.add_argument("--start-date", default=None, help="Override config data.start_date")
    p.add_argument(
        "--target",
        choices=("intraday", "winsorize_intraday", "overnight", "day"),
        default=None,
        help="Override config target",
    )
    p.add_argument("--out-dir", default=None, help="Override config output.dir")
    p.add_argument("--seed", type=int, default=None, help="Override config random_state")
    args = p.parse_args()

    cfg = load_train_config(args.config, famose_config_path=args.famose_config)
    if args.data_dir is not None:
        cfg.data.handler_dir = args.data_dir
    if args.start_date is not None:
        cfg.data.start_date = args.start_date if args.start_date else None
    if args.target is not None:
        cfg.target = args.target
    if args.out_dir is not None:
        cfg.output.dir = args.out_dir
    if args.seed is not None:
        cfg.random_state = args.seed

    print("FAMOSE verify: loading panel (Handler + target + seed features from config)...")
    out_dir = run_famose(cfg, dry_run=args.dry_run)

    if args.show_trace > 0:
        _print_trace_tail(out_dir / "famose_trace.jsonl", args.show_trace)

    summary_path = out_dir / "famose_summary.json"
    if summary_path.is_file():
        print(f"Summary: {summary_path}")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"  seeds={data.get('seed_count')} accepted={data.get('accepted_count')} "
            f"mRMR order length={len(data.get('mrmr_order') or [])}"
        )


if __name__ == "__main__":
    main()
