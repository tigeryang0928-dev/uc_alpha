"""FAMOSE-style outer rounds: LLM proposes DSL features, XGBoost validates, mRMR orders output."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from train_config import TrainConfig

from famose.context import (
    build_panel_context,
    maybe_subsample_by_date,
    stack_from_specs,
    train_val_masks,
)
from famose.evaluate import score_candidate
from famose.expr_validate import allowed_names_for_handler, validate_feature_expr
from famose.llm import (
    GEMINI_429_MAX_CONSECUTIVE,
    famose_chat_completion,
    parse_proposal_json,
)
from famose.cursor_cli import resolve_workspace
from famose.mrmr import mrmr_select_features

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe_name(name: str, used: set[str]) -> str:
    base = "".join(c if c.isalnum() or c == "_" else "_" for c in name.strip())[:80]
    if not base:
        base = "famose_feat"
    out = base
    i = 0
    while out in used:
        i += 1
        out = f"{base}_{i}"
    return out


def run_famose(cfg: TrainConfig, *, api_key: str | None = None, dry_run: bool = False) -> Path:
    fc = cfg.famose
    ctx = build_panel_context(cfg)
    handler_keys = list(ctx.handler.cash_list())
    allow = allowed_names_for_handler(handler_keys, ctx.funcs_methods.keys())

    base_specs = list(ctx.seed_specs)
    accepted: list[tuple[str, str]] = []

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(cfg.resolved_output_dir()) / f"{fc.run_subdir}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "famose_trace.jsonl"

    def log(obj: dict) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    rng = np.random.default_rng(cfg.random_state)
    xgb_kw = dict(cfg.xgboost)
    names_used = {n for n, _ in base_specs + accepted}

    stagnant = 0
    for r in range(fc.max_rounds):
        round_best: tuple[str, str] | None = None
        round_best_rmse = float("inf")
        round_best_rel = 0.0
        gemini_429_step_streak = 0

        for step in range(fc.max_steps_per_round):
            if dry_run and r == 0 and step == 0:
                proposal = {
                    "feature_name": "famose_cs_rank_turnover",
                    "expr": "cs_rank(Turnover)",
                    "rationale": "dry-run canned feature",
                }
            else:
                meta = {
                    "handler_columns": handler_keys[:200],
                    "dsl_functions": sorted(ctx.funcs_methods.keys()),
                    "target": cfg.target,
                    "seed_and_accepted": [f"{n} = {e}" for n, e in base_specs + accepted],
                }
                sys_msg = (
                    "You are a quant researcher. Propose ONE new factor as a Python expression "
                    "evaluated as eval(expr, funcs_methods, handler) in this codebase. "
                    "Use only handler column names (raw data fields) and registered DSL function "
                    "names from metadata. No attributes, no subscripts, no imports, no lambdas. "
                    "Reply with a single JSON object only: "
                    '{"feature_name": "snake_case", "expr": "...", "rationale": "short"}.'
                )
                user_msg = json.dumps(meta, indent=2)
                try:
                    prov_l = fc.llm_provider.strip().lower()
                    raw = famose_chat_completion(
                        [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        provider=fc.llm_provider,
                        model=fc.llm_model,
                        temperature=fc.llm_temperature,
                        base_url=fc.llm_base_url,
                        api_key=api_key,
                        cursor_workspace=resolve_workspace(fc.llm_cursor_workspace, _REPO_ROOT)
                        if prov_l == "cursor"
                        else None,
                        cursor_cli=fc.llm_cursor_cli,
                        cursor_extra_args=fc.llm_cursor_extra_args,
                        cursor_timeout_sec=fc.llm_cursor_timeout_sec,
                    )
                except Exception as e:
                    err_s = str(e)
                    log({"round": r, "step": step, "error": err_s})
                    prov = fc.llm_provider.strip().lower()
                    is_gem = prov in ("gemini", "google", "google_gemini")
                    is_429 = "429" in err_s or "RESOURCE_EXHAUSTED" in err_s
                    if is_gem and is_429:
                        gemini_429_step_streak += 1
                        log(
                            {
                                "round": r,
                                "step": step,
                                "gemini_429_step_streak": gemini_429_step_streak,
                            }
                        )
                        if gemini_429_step_streak >= GEMINI_429_MAX_CONSECUTIVE:
                            log(
                                {
                                    "round": r,
                                    "step": step,
                                    "gemini_429_streak_break": True,
                                    "message": f"{GEMINI_429_MAX_CONSECUTIVE} consecutive Gemini 429 step failures",
                                }
                            )
                            break
                        continue
                    gemini_429_step_streak = 0
                    continue
                gemini_429_step_streak = 0
                proposal = parse_proposal_json(raw) or {}
                proposal["_raw_head"] = raw[:2000]

            name = str(proposal.get("feature_name") or "").strip()
            expr = str(proposal.get("expr") or "").strip()
            err = validate_feature_expr(expr, allow)
            if err:
                row = {"round": r, "step": step, "reject": err, "expr": expr}
                if not expr and proposal.get("_raw_head"):
                    row["llm_raw_head"] = str(proposal["_raw_head"])[:2000]
                log(row)
                continue

            fname = _safe_name(name or "famose_feat", names_used)
            names_used.add(fname)
            try:
                Xc, yc = stack_from_specs(ctx, base_specs + accepted + [(fname, expr)])
                Xc, yc = maybe_subsample_by_date(Xc, yc, fc.date_sample_frac, rng)
                tr2, va2 = train_val_masks(Xc.index, cfg)
                cols_base = [c for c in Xc.columns if c != fname]
                X_tr_b = Xc.loc[tr2, cols_base]
                X_va_b = Xc.loc[va2, cols_base]
                cand_tr = Xc.loc[tr2, fname]
                cand_va = Xc.loc[va2, fname]
            except Exception as e:
                log({"round": r, "step": step, "eval_error": str(e), "expr": expr})
                continue

            rmse_w, rmse_wo, rel = score_candidate(
                X_tr_b,
                yc.loc[tr2],
                X_va_b,
                yc.loc[va2],
                fname,
                cand_tr,
                cand_va,
                xgb_kw,
                cfg.random_state + r * 100 + step,
            )
            log(
                {
                    "round": r,
                    "step": step,
                    "name": fname,
                    "expr": expr,
                    "rmse_with": rmse_w,
                    "rmse_without": rmse_wo,
                    "rel_improve": rel,
                }
            )

            if rel >= fc.min_rel_improvement and rmse_w < round_best_rmse:
                round_best_rmse = rmse_w
                round_best_rel = rel
                round_best = (fname, expr)

            if dry_run and r == 0:
                break

        if round_best is not None:
            accepted.append(round_best)
            stagnant = 0
            log({"round_end": r, "accepted": list(round_best), "rel_improve": round_best_rel})
        else:
            stagnant += 1
            log({"round_end": r, "accepted": None, "stagnant": stagnant})
            if stagnant >= fc.early_stop_rounds_no_gain:
                break

        if dry_run:
            break

    all_specs = base_specs + accepted
    spec_by_name = dict(all_specs)
    X_sel, y_sel = stack_from_specs(ctx, all_specs)
    X_sel, y_sel = maybe_subsample_by_date(X_sel, y_sel, fc.date_sample_frac, rng)
    tr_m, _ = train_val_masks(X_sel.index, cfg)
    X_tr_only = X_sel.loc[tr_m]
    y_tr_only = y_sel.loc[tr_m]
    order = mrmr_select_features(
        X_tr_only, y_tr_only, fc.mrmr_max_features, random_state=cfg.random_state
    )
    mrmr_specs = [(n, spec_by_name[n]) for n in order if n in spec_by_name]

    out_yaml = out_dir / "discovered_features.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"features": [{"name": n, "expr": e} for n, e in mrmr_specs]},
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    summary = {
        "seed_count": len(base_specs),
        "accepted_count": len(accepted),
        "accepted": [{"name": n, "expr": e} for n, e in accepted],
        "mrmr_order": order,
    }
    (out_dir / "famose_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"FAMOSE run directory: {out_dir}")
    print(f"Wrote {out_yaml}")
    return out_dir
