# LPI Finance — Validación con Datos Reales

## Estado actual del proyecto (abril 2026)

**Fase técnica completada.** El proyecto entra en fase de escritura del paper y comercialización.

| Elemento | Valor |
|---|---|
| Modelo baseline oficial | CBOE 4-features (scripts 09-13) |
| AUC OOS | **0.6253 ± 0.0112** |
| Shuffle test | 0.5005 ± 0.0024 (leakage descartado) |
| K* (Bootstrap BIC) | 4 |
| Backtest (VIX puro, bid-ask 20%) | Sharpe **+1.37**, CAGR +7.5% |
| Backtest (VIX puro, bid-ask 5%) | Sharpe **+3.53**, CAGR +20.7% |
| Break-even de IV markup | Entre 1.2× y 1.5× (VIX/VXN como proxy) |

**Features del modelo final:** `iv_rv_spread_cboe`, `log_range`, `vol_of_vol`, `log_dvol`

**Experimentos descartados:**
- v3 7-features (AUC 0.6015) — maldición de dimensionalidad en GMM
- v3 ensemble 7 semillas (AUC 0.5785) — no mejora vs single seed

**Próximos pasos:** paper académico + producto B2B. No más mejoras técnicas del algoritmo en esta iteración.
Si se quiere mejorar AUC: obtener IV individual por ticker (Polygon.io, Tradier o OptionMetrics), no añadir más features OHLCV.

---

## Resumen del análisis de costes (stress test)

Sharpe ratio según bid-ask spread e IV markup (position_size=5%):

```
                bid_ask 5%  bid_ask 10%  bid_ask 15%  bid_ask 20%
iv_markup 1.0×   +3.53 ✓     +2.82 ✓     +2.10 ✓     +1.37 ✓
iv_markup 1.2×   +0.50 ✓     -0.37       -1.21 ✗     -2.02 ✗
iv_markup 1.5×   -3.70 ✗     -4.53 ✗     -5.29 ✗     -5.97 ✗
iv_markup 2.0×   -8.05 ✗     -8.56 ✗     -9.00 ✗     -9.37 ✗
```

**Interpretación:** El modelo es viable si VIX/VXN como proxy subestima la IV real en menos de ~20-30%.
La hipótesis más optimista (IV individual = VIX puro) da Sharpe > 1 incluso con bid-ask del 20%.
La hipótesis pesimista (IV individual = 1.5× VIX) produce pérdidas sistemáticas.

---

## Algoritmo

**LPI (Latent Propensity Index):**
1. Estandarización con RobustScaler (por fold, sin leakage)
2. Selección de K* con Bootstrap BIC sobre GMM full-covariance
3. CV temporal purged (PurgedTimeSeriesSplit, 5 folds, embargo 10 días)
4. Por fold: GMM → enriquecimiento por cluster → score OOS = Σ P(Cₖ|x) · f̄ₖ

**Target:** `rv_5_forward > iv_cboe × 1.3` (¿será la vol realizada mayor que la implícita por un factor 1.3?)

**IV source:** VIX (tickers S&P 500) / VXN (tickers Nasdaq), en decimal

---

## Reproducir el modelo baseline

```bash
pip install -r requirements.txt
pytest tests/                          # todos deben pasar
python3.11 scripts/09_download_cboe.py
python3.11 scripts/10_build_panel_cboe.py
python3.11 scripts/11_run_lpi_cboe.py
python3.11 scripts/12_diagnostics_cboe.py
python3.11 scripts/13_compare_proxy_vs_cboe.py
python3.11 scripts/05_economic_backtest.py   # backtest con config original
python3.11 scripts/21_cost_sensitivity.py    # stress test de costes
```

---

## Estructura del proyecto

```
src/
  lpi_core.py        # Algoritmo LPI (sin dependencias financieras)
  features_v2.py     # 4 features CBOE + 3 features v3 (descartadas)
  target_v2.py       # Target con IV real (CBOE)
  cboe_data.py       # Descarga y mapeo VIX/VXN por ticker
  cv.py              # PurgedTimeSeriesSplit con embargo
  backtest.py        # Simulación de straddles
  diagnostics.py     # Shuffle test, ablación, estabilidad
  reporting.py       # Métricas, figuras, veredictos

scripts/
  01-04              # Pipeline baseline (proxy IV, 7 features)
  05                 # Backtest económico original
  06-08              # Experimento 4-features (ablación)
  09-13              # Pipeline CBOE IV → MODELO FINAL
  14-18              # Experimento v3 7-features (DESCARTADO)
  19-20              # Backtest honesto con costes realistas
  21                 # Stress test de costes (4×4 combinations)

tests/
  test_lpi_core.py       # AUC > 0.95 en datos sintéticos separables
  test_no_leakage.py     # Sin look-ahead en ninguna feature (14 tests)
  test_cv_no_overlap.py  # Sin contaminación train/test
  test_backtest.py       # Validación del simulador de straddles
  test_cboe_data.py      # Cobertura del mapeo VIX/VXN
  test_lpi_ensemble.py   # Tests del ensemble (DESCARTADO)

config.yaml              # Baseline 7 features (proxy IV)
config_cboe.yaml         # MODELO FINAL: 4 features CBOE
config_v3.yaml           # DESCARTADO: 7 features + ensemble
config_backtest.yaml     # Backtest original (1% pos, 8% bid-ask)
config_backtest_honest.yaml  # Backtest conservador (5% pos, 15% bid-ask, 1.5× IV)
```

---

## Decision criteria (originales)

| AUC OOS | Veredicto | Acción |
|---------|-----------|--------|
| < 0.53 | SIN SEÑAL | Cerrar línea de investigación |
| 0.53–0.56 | MARGINAL | Documentar como inconcluso |
| 0.56–0.60 | DÉBIL | Iterar features y proxy IV |
| 0.60–0.65 | MODERADA | Pasar a backtest económico |
| > 0.65 | FUERTE | Deflated Sharpe + purged combinatorial CV |

**Resultado final: AUC 0.6253 → MODERADA** — señal real, costes son el factor limitante.
