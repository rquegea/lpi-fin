# RESULTS SUMMARY — LPI Finance

Resumen ejecutivo de todos los experimentos realizados.
Fase técnica completada: abril 2026.

---

## Experimentos realizados

| Experimento | Features | Modelo | AUC OOS | K* | Notas |
|---|---|---|---|---|---|
| Proxy IV 7f | 7 (OHLCV + proxy rv30×1.04) | GMM single | 0.6342 ± — | — | Baseline inicial |
| Proxy IV 4f | 4 (ablación: solo deltas +) | GMM single | **0.6406** ± — | — | Mejor con proxy IV |
| CBOE IV 4f | 4 (iv_rv_spread_cboe, log_range, vol_of_vol, log_dvol) | GMM single | **0.6253** ± 0.0112 | 4 | **MODELO FINAL** |
| v3 7f single | 7 (CBOE 4 + skew_60d, kurt_60d, vol_autocorr_5d) | GMM single | 0.6015 ± 0.0149 | 11 | DESCARTADO |
| v3 7f ensemble | 7 (ídem) | GMM × 7 semillas | 0.5785* | 11–14 | DESCARTADO |

*AUC global sobre scores promediados; AUCs individuales por semilla: 0.5915–0.6076.

**Lección del experimento v3:** al pasar de 4 a 7 features, el Bootstrap BIC selecciona K*=11–14 (vs K*=4 con 4 features). El GMM full-covariance con K alto en 7 dimensiones sufre maldición de la dimensionalidad y los nuevos features añaden ruido en lugar de señal. El camino correcto para mejorar el AUC es obtener IV individual por ticker (no features OHLCV adicionales).

---

## Diagnósticos del modelo final (CBOE 4-features)

| Diagnóstico | Resultado | Interpretación |
|---|---|---|
| Shuffle test (12 perm.) | 0.5005 ± 0.0024 | ✓ Sin leakage — señal real |
| p empírico | < 0.001 | ✓ Significativo |
| Chi² quintiles | p < 1e-177 | ✓ Distribución no aleatoria |
| Lift Q5 | 1.39× | Quintil superior genera 39% más eventos |
| Lift Q1 | 0.83× | Quintil inferior genera 17% menos eventos |
| K* (Bootstrap BIC) | 4 | ✓ Parsimónico y estable |
| Estabilidad multi-semilla | — | Ver diagnostics_cboe.csv |

**Feature más crítico (ablación proxy IV):** `vol_of_vol` (+0.0162 delta AUC)

---

## Backtest económico — análisis de sensibilidad a costes

Simulación de straddles ATM con hold_days=5, top quintile por LPI score.
Position size fijo: 5% del capital por posición, máximo 10 posiciones simultáneas.

### Tabla de Sharpe ratio

```
                bid_ask 5%  bid_ask 10%  bid_ask 15%  bid_ask 20%
iv_markup 1.0×   +3.53 ✓     +2.82 ✓     +2.10 ✓     +1.37 ✓
iv_markup 1.2×   +0.50 ✓     -0.37       -1.21 ✗     -2.02 ✗
iv_markup 1.5×   -3.70 ✗     -4.53 ✗     -5.29 ✗     -5.97 ✗
iv_markup 2.0×   -8.05 ✗     -8.56 ✗     -9.00 ✗     -9.37 ✗
```

### Tabla de CAGR (%)

```
                bid_ask 5%  bid_ask 10%  bid_ask 15%  bid_ask 20%
iv_markup 1.0×   +20.74%     +16.15%     +11.72%      +7.46%
iv_markup 1.2×    +2.56%      -2.13%      -6.61%     -10.90%
iv_markup 1.5×   -19.86%     -24.45%     -28.79%     -32.89%
iv_markup 2.0×   -47.12%     -51.18%     -54.94%     -58.42%
```

### Parámetros del modelo de P&L

```
cost_iv    = iv_cboe(t) × iv_markup × √(hold_days / 252)
bid_ask    = bid_ask_cost_pct × cost_iv
total_cost = cost_iv × (1 + bid_ask_cost_pct)
payoff     = |return_5d| - total_cost
pnl_usd    = payoff × (position_size_pct × capital)
```

### Interpretación

El modelo es **viable si el VIX/VXN subestima la IV individual en menos de ~20–30%**.

- `iv_markup = 1.0`: usar VIX puro como coste → Sharpe > 1 incluso con bid-ask 20%
- `iv_markup = 1.2`: margen muy estrecho → solo viable con bid-ask ≤ 5%
- `iv_markup ≥ 1.5`: pérdidas sistemáticas independientemente del spread

**Pregunta clave para el paper:** ¿Cuánto subestima el VIX/VXN la IV individual?
Literatura sugiere 5–20% para large-caps líquidos, no 50%.
Datos reales de opciones (OptionMetrics, Polygon) resolverían esta incertidumbre.

---

## Limitaciones del modelo actual

1. **IV proxy grueso:** VIX cubre todos los tickers S&P500; VXN cubre todos los Nasdaq. No hay IV individual. Esto subestima el coste real y puede inflar el AUC al mezclar señales de distintos regímenes dentro del índice.

2. **Modelo de P&L simplificado:** sin modelado de griegas (gamma, theta, vega), sin ejercicio anticipado, sin ajuste de dividendos.

3. **Ejecución a precio de cierre:** sin slippage intradiario ni impacto de mercado.

4. **Straddle ≠ opción real:** la construcción asume delta-neutral en el momento de compra; en la práctica el delta drift requiere rebalanceo.

---

## Próximos pasos

### Paper académico
- Describir el algoritmo LPI con GMM full-covariance y Bootstrap BIC
- Comparar con benchmarks: logistic regression, random forest, vol-targeting naive
- Sección de limitaciones: IV proxy, ejecutabilidad, costes reales
- Propuesta: validar con IV individual (OptionMetrics) como extensión futura

### Producto B2B
- Target: fondos de volatilidad, market makers, prop desks con book de opciones
- Propuesta de valor: señal de timing para compra/venta de volatilidad a corto plazo
- MVP: API de scores diarios (top quintile ticker list + LPI score) basada en CBOE 4-features
- Limitación a comunicar: la señal opera sobre IV agregada de índice, no individual

### Mejora técnica futura (fuera del scope de esta fase)
- Fuente de IV individual por ticker: Polygon.io (opción B2C asequible) o OptionMetrics (institucional)
- Con IV individual se espera resolver la ambigüedad del iv_markup y potencialmente mejorar el AUC hacia 0.65+
