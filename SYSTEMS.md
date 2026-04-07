# System map (`uc_alpha`)

Use this file when you want changes in a **specific area** of the project. In chat you can say: *“Update **ML training** per `SYSTEMS.md`”* or *“Change **Handler / data** only.”*

Paths are relative to the repository root.

---

## Quick lookup

| System name (say this) | Primary locations | What it does |
|------------------------|-------------------|--------------|
| **Entry / CLI** | `demo.py` | Runs the pipeline; parses args and calls training. |
| **ML training (XGBoost + SHAP)** | `xgb_shap/train_xgboost_shap.py` | End-to-end train, metrics, SHAP, writes `xgb_shap/output/`. |
| **Training config** | `xgb_shap/config.yaml`, `xgb_shap/train_config.py` | YAML settings and typed load/save of config. |
| **Panel ML** | `xgb_shap/panel_ml.py` | Panel stacking, row/date masks used by training. |
| **Conditional SHAP** | `xgb_shap/shap_conditional.py` | Conditional monotonic SHAP analysis and plots. |
| **Handler / data bridge** | `backtest_2/utils/data_bridge.py` | Loads TEJ-style panel data (`Handler`) for features and labels. |
| **Factors / feature expressions** | `backtest_2/method.py` | `funcs_methods` and expressions consumed by training. |
| **Evaluation / factor analytics** | `backtest_2/utils/evaluate.py` | `factors_analyze` and related backtest-style evaluation. |
| **Data assets** | `backtest_2/data/` *(gitignored)* | Parquet inputs; not in Git—document expected filenames here if you add new ones. |
| **Run outputs** | `xgb_shap/output/` *(gitignored)* | Timestamped runs (CSVs, HTML, JSON, plots). |
| **Notebook (exploratory)** | `backtest_2/因子分析.ipynb` | Jupyter factor analysis. |
| **Data sanity check** | `data_print.py` | Small script to peek at a parquet (example: `turnover`). |
| **Dependencies** | `requirements.txt` | Python packages. |
| **FAMOSE (feature discovery)** | `xgb_shap/famose/` (`config.yaml` + package), `famose_verify.py`, `xgb_shap/famose_cli.py` | FAMOSE settings in `famose/config.yaml`; main `config.yaml` supplies data/target/seed features. Writes `output.dir/famose_runs_<ts>/`. |

---

## Detail (for agents and humans)

### Entry / CLI — `demo.py`

- **Change when:** CLI flags, default paths, wiring into `xgb_shap` without touching core training logic.
- **Depends on:** `xgb_shap/train_xgboost_shap.py`, `xgb_shap/train_config.py`.

### FAMOSE — `famose_verify.py` + `xgb_shap/famose/`

- **Change when:** Automated DSL feature proposals, validation metric, mRMR export, or LLM prompt/tools.
- **Config:** `xgb_shap/famose/config.yaml` (FAMOSE-only). Main `xgb_shap/config.yaml` still provides data, target, splits, XGBoost params, and seed `features`. `load_train_config(..., famose_config_path=...)` merges them.
- **Depends on:** `train_config.py` (`FamoseConfig`, `load_famose_config`), `train_xgboost_shap.py` (panel + target builders), `panel_ml.py`, `backtest_2/method.py`, `Handler`.
- **Verify:** `python famose_verify.py --dry-run` (no API key). Full loop needs `OPENAI_API_KEY`. Outputs are under `xgb_shap/output/famose_runs_<timestamp>/` by default.

### ML training (XGBoost + SHAP) — `xgb_shap/train_xgboost_shap.py`

- **Change when:** Model fit, target definition, feature loop, SHAP integration, saving artifacts.
- **Pulls from:** `backtest_2/method.py`, `backtest_2/utils/data_bridge.py`, `backtest_2/utils/evaluate.py`, `panel_ml.py`, `shap_conditional.py`, `train_config.py`.

### Training config — `xgb_shap/config.yaml` + `xgb_shap/train_config.py`

- **Change when:** New hyperparameters, data paths, stock lists, split ratios, SHAP sample size, output directory schema.
- **Keep in sync:** Any new fields must be parsed in `train_config.py` and used in `train_xgboost_shap.py` (and optionally `demo.py` overrides).

### Panel ML — `xgb_shap/panel_ml.py`

- **Change when:** How the panel is stacked, masks by date/row, train/val/test construction.

### Conditional SHAP — `xgb_shap/shap_conditional.py`

- **Change when:** Monotonicity reports, plots, JSON export of conditional SHAP summaries.

### Handler / data bridge — `backtest_2/utils/data_bridge.py`

- **Change when:** Reading parquets, symbol/date indexing, exposing columns to `Handler`, IO paths.

### Factors / feature expressions — `backtest_2/method.py`

- **Change when:** Adding or editing cross-sectional or time-series feature functions used by `funcs_methods` and string expressions in config/training.

### Evaluation / factor analytics — `backtest_2/utils/evaluate.py`

- **Change when:** IC, quantile stats, cumulative plots, or other factor-evaluation logic used after features are built.

### Data & outputs (not tracked by Git)

- **`backtest_2/data/`** — Source parquets; referenced by config (`handler_dir`, etc.).
- **`xgb_shap/output/`** — Generated per run; safe to delete locally; do not commit.

---

## Maintenance

- When you add a **new top-level folder** or **major script**, add one row to the **Quick lookup** table and a short **Detail** subsection.
- Prefer stable **system names** in the first column so prompts stay consistent over time.
