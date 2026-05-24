# AutoML Studio

**End-to-end automated machine learning platform** with 80+ API endpoints covering data profiling, cleaning, transformation, model training, hyperparameter optimization, explainability, and deployment.

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd AutoML
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY

# 3. Run
python app.py
# Open http://localhost:7860
```

### Docker

```bash
docker build -t automl-studio .
docker run -p 7860:7860 --env-file .env automl-studio
```

## Architecture

```
app.py                  # Flask application (routes, middleware, startup)
session_manager.py      # Thread-safe TTL session store (cachetools)
logging_config.py       # Structured JSON logging with request IDs

ml_engine/
├── pipeline.py         # Orchestrator — PipelineManager & PipelineSession
├── profiler.py         # Dataset profiling & statistics
├── transformer.py      # Data transformation (encoding, scaling, imputation)
├── trainer.py          # Model training & evaluation
├── hyperopt_engine.py  # Hyperparameter optimization (Optuna)
├── search_spaces.py    # Centralised hyperparameter grids (single source of truth)
├── explainer.py        # SHAP, PDP, counterfactuals
├── auto_eda.py         # Automated exploratory data analysis
└── ...                 # 25+ additional ML engine modules

tests/
├── conftest.py         # Shared fixtures
├── test_transformer.py # Data leakage prevention tests
└── test_session_manager.py  # Session TTL & thread-safety tests
```

## Key Features

| Category | Features |
|----------|----------|
| **Data** | Upload CSV/Excel/JSON/Parquet, auto-profiling, cleaning suggestions, feature engineering studio |
| **Training** | 10+ algorithms (RF, XGB, LightGBM, CatBoost, DL), auto hyperparameter tuning, stacking ensemble |
| **Explainability** | SHAP, partial dependence, counterfactuals, prediction autopsy, model cards |
| **Advanced** | Federated learning, causal inference, conformal prediction, adversarial testing |
| **Ops** | Experiment tracking, model versioning, drift detection, health monitoring |

## Security

- **CORS** — Restricted to `ALLOWED_ORIGINS` (configurable via env var)
- **Rate Limiting** — 200 req/hour default; stricter on upload (20/min), train (5/min), chat (10/min)
- **File Validation** — Extension check + magic-byte content validation
- **Sessions** — Full 128-bit UUIDs, TTL-based auto-expiry, thread-safe store
- **Upload Limits** — Configurable max file size (default 100 MB)

## Testing

```bash
pytest tests/ -v
```

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `dev-change-me` | Flask secret key (required in production) |
| `ALLOWED_ORIGINS` | `http://localhost:7860` | CORS allowed origins (comma-separated) |
| `MAX_UPLOAD_MB` | `100` | Maximum upload file size in MB |
| `SESSION_TTL` | `7200` | Session time-to-live in seconds |
| `GEMINI_API_KEY` | — | Google Gemini API key for LLM features |
| `OPENAI_API_KEY` | — | OpenAI API key (alternative LLM) |

## License

MIT