#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List FAMOSE expressions by rel_improve threshold."
    )
    parser.add_argument(
        "trace_file",
        nargs="?",
        default="xgb_shap/output/famose_runs_2026-04-07_14-59-47/famose_trace.jsonl",
        help="Path to famose_trace.jsonl",
    )
    parser.add_argument(
        "--min-rel-improve",
        type=float,
        default=0.0001,
        help="Minimum rel_improve to keep (default: 0.0001)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace_file)

    if not trace_path.exists():
        print(f"File not found: {trace_path}")
        return 1

    hits = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rel = record.get("rel_improve")
            expr = record.get("expr")
            if rel is None or expr is None:
                continue
            if rel >= args.min_rel_improve:
                hits.append(
                    {
                        "line": line_no,
                        "round": record.get("round"),
                        "step": record.get("step"),
                        "agent": record.get("agent"),
                        "name": record.get("name"),
                        "rel_improve": rel,
                        "expr": expr,
                    }
                )

    hits.sort(key=lambda x: x["rel_improve"], reverse=True)

    print(f"trace_file: {trace_path}")
    print(f"min_rel_improve: {args.min_rel_improve}")
    print(f"matches: {len(hits)}")
    print("-" * 100)

    for i, item in enumerate(hits, start=1):
        print(
            f"[{i}] line={item['line']} round={item['round']} step={item['step']} "
            f"agent={item['agent']} rel_improve={item['rel_improve']:.12f} "
            f"name={item['name']}"
        )
        print(f"    expr: {item['expr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
