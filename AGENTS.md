## Learned User Preferences

- When changing a subsystem, prefer the named systems and paths in `SYSTEMS.md` so edits stay scoped to the right area.
- SHAP “monotonicity” in this repo means Spearman correlation between raw feature values and that feature’s SHAP, not Pearson linearity.
- Run folder timestamps are treated as the experiment id for saved config snapshots and outputs.

## Learned Workspace Facts

- Main training entry is `xgb_shap/train_xgboost_shap.py`; defaults and overrides live in `xgb_shap/config.yaml` and `xgb_shap/train_config.py`.
- Conditional and discretization scans are implemented in `xgb_shap/shap_conditional.py`; optional inner-quantile band scans use `discretization_scan_band_pairs` (each pair needs 0 < q_lo < q_hi < 1).
- Each run writes under `xgb_shap/output/<YYYY-mm-dd_HH-MM-SS>/`; the effective config snapshot is `config_<same_timestamp>.yaml` in that folder.
- Root `SYSTEMS.md` maps conversational system names (e.g. ML training, Handler / data bridge) to paths and responsibilities.
- `.gitignore` excludes `backtest_2/data/`, `xgb_shap/output/`, and Python bytecode (`__pycache__/`, `*.py[cod]`).
- Remote GitHub repo for this project is `tigeryang0928-dev/uc_alpha`; HTTPS is often used when SSH to GitHub fails on this machine.
- Primary feature branch used in recent work is `jonathan_branch` (the misspelled `jonathon_branch` was removed from remote).
