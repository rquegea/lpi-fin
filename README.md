# LPI Finance — Validación con Datos Reales

Validates the LPI (Latent Propensity Index) clustering algorithm on real equity market data, predicting divergence between implied and realized volatility.

## Decision criteria

| AUC OOS | Verdict | Action |
|---------|---------|--------|
| < 0.53 | SIN SEÑAL | Close this research line |
| 0.53–0.56 | MARGINAL | Document as inconclusive |
| 0.56–0.60 | DÉBIL | Iterate features and IV proxy |
| 0.60–0.65 | MODERADA | Proceed to economic backtest (Script 05) |
| > 0.65 | FUERTE | Apply Deflated Sharpe + purged combinatorial CV first |

## Quick start

```bash
pip install -r requirements.txt
pytest tests/                         # must pass before proceeding
python scripts/01_download_data.py
python scripts/02_build_panel.py
python scripts/03_run_lpi.py
python scripts/04_diagnostics.py
```

## Project structure

```
src/
  lpi_core.py      # LPI algorithm (finance-free)
  cv.py            # PurgedTimeSeriesSplit with embargo
  features.py      # 7 features, no look-ahead
  target.py        # Binary IV/RV target
  data_loader.py   # yfinance download + Parquet cache
  diagnostics.py   # Shuffle test, ablation, stability
  reporting.py     # Metrics, figures, verdict
scripts/
  01_download_data.py
  02_build_panel.py
  03_run_lpi.py
  04_diagnostics.py
  05_economic_backtest.py  # placeholder
tests/
  test_lpi_core.py        # AUC > 0.95 on synthetic separable data
  test_no_leakage.py      # no look-ahead in any feature
  test_cv_no_overlap.py   # no train/test contamination
```

## Configuration

All parameters in `config.yaml`:
- Universe: 30 liquid S&P 500 tickers
- Period: 2015-01-01 to 2024-12-31
- 5-fold expanding CV, 10-day embargo
- Bootstrap BIC: K ∈ {2..15}, B=20 bootstraps
- IV proxy: `rv_30 * 1.04`
- Target: `rv_5_forward > iv_level * 1.3`
