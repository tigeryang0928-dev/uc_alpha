# uc_alpha

Cross-sectional equity research: **TEJ-style panel data** → factor expressions → **XGBoost** training with **SHAP** (including conditional analysis), plus optional **FAMOSE** DSL feature discovery.

## Where to look

| Doc | Purpose |
|-----|---------|
| [`SYSTEMS.md`](SYSTEMS.md) | Map of subsystems, paths, and when to change what |
| [`AGENTS.md`](AGENTS.md) | Workspace conventions (SHAP monotonicity = Spearman, run folders, FAMOSE notes) |

## Quick entry points

- Training: `demo.py` → `xgb_shap/train_xgboost_shap.py`; config in `xgb_shap/config.yaml` and `xgb_shap/train_config.py`.
- FAMOSE: `famose_verify.py` (e.g. `--dry-run` without `OPENAI_API_KEY`); package under `xgb_shap/famose/`. Copy `xgb_shap/famose/config.example.yaml` to `config.yaml` locally (gitignored).

Data under `backtest_2/data/` and run outputs under `xgb_shap/output/` are not tracked; see `.gitignore`.
