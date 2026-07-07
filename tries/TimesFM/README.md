# OjoAlTicker · Forecast de cartera con TimesFM 2.5 (zero-shot)

Notebook de laboratorio que evalúa si un foundation model de series temporales — **TimesFM 2.5 200M** (Google, checkpoint público `google/timesfm-2.5-200m-pytorch`, sin fine-tuning) — aporta capacidad predictiva real sobre la cesta de 6 activos de `MonteCarlo/OjoAlTicker.ipynb`, a horizonte semana (5 días) y mes (21 días).

La respuesta corta, con los datos actuales: **no**. El valor del notebook no es la predicción sino el arnés de validación que lo demuestra — y que diría lo contrario si algún día dejara de ser cierto.

---

## Qué hace

1. **Datos**: 3 años de cierres ajustados (yfinance) para MSFT, JPM, JNJ, XOM, AMZN, NEE — fechas pinneadas y cache CSV en `data/` (yfinance re-ajusta el histórico retroactivamente; sin pin + cache el notebook no es reproducible).
2. **Exploración**: ACF de retornos (≈ ruido blanco) y de retornos² (volatility clustering) — motiva los baselines y el baseline de cuantiles.
3. **Modelo**: TimesFM 2.5 forecastea **precio** (nivel), no retorno; el retorno se deriva después (`P_h/P_0 − 1`). Inferencia directa multi-horizonte, 6 series por batch, cuantiles P10–P90 con quantile head continua.
4. **Backtest walk-forward** (96 orígenes, paso 5 días, corte por posición sin fuga) contra dos baselines:
   - random-walk sin drift (retorno esperado 0)
   - random-walk **con drift** histórico del contexto — el listón real: batir al RW-0 a un mes puede ser solo capturar la deriva alcista.
5. **Evaluación en cuatro planos**:
   - Error puntual: MAE/WAPE/bias + test de **Diebold-Mariano** con varianza Newey-West (los orígenes solapan con el horizonte mes) y corrección de muestra pequeña de **Harvey-Leybourne-Newbold (1997)**, p-valores sobre t de Student.
   - Cartera **equal-weight** (los pesos Max Sharpe del notebook de MonteCarlo se optimizaron sobre esta misma ventana — usarlos en el backtest sería leakage).
   - Cuantiles: cobertura empírica P10–P90 y pinball loss contra un baseline gaussiano drift + vol del contexto.
   - Direccional: hit-rate con test de **Pesaran-Timmermann** y P&L de una regla *long si pred > 0* con costes (5 bps/lado).
6. **Producción**: predicción a hoy con todo el histórico como contexto y los pesos Max Sharpe reales de la cartera.

El backtest se persiste en `data/backtest_<fecha>_step5.parquet` — re-ejecutar el notebook con el cache presente tarda ~2 min en lugar de ~10.

---

## Resultados (datos hasta 06/07/2026)

| Horizonte | Skill MAE vs RW-0 | Skill MAE vs RW-drift | DM p (drift) | Hit-rate | Cobertura P10-P90 |
|-----------|------------------:|----------------------:|-------------:|---------:|------------------:|
| Semana    | −0.5%             | −4.4%                 | 0.145        | 51.0%    | 78.5%             |
| Mes       | +4.6%             | **−6.3%**             | 0.479        | 53.1%    | 74.8%             |

- TimesFM **no bate al random-walk con drift** en ningún horizonte: lo que añade sobre capturar la deriva histórica es cero o negativo.
- Sus cuantiles tampoco mejoran al baseline gaussiano (cobertura y pinball ligeramente peores).
- El acierto direccional es indistinguible del azar (Pesaran-Timmermann p ≈ 0.53) y la regla long/flat rinde menos que buy-and-hold tras costes.
- 1/12 combinaciones ticker-horizonte significativas vs drift — compatible con los ~0.6 falsos positivos esperados bajo H0 con 12 tests al 5%, sin corrección por comparaciones múltiples.

Es el resultado que la literatura anticipa para retornos diarios de mega-caps (mercado casi eficiente a estos horizontes). Con un arnés más débil — pocos orígenes, solo el baseline sin drift, pesos de cartera seleccionados in-sample — el mismo modelo aparenta batir al random-walk a un mes; el arnés de este notebook existe precisamente para que ese tipo de espejismo no se cuele.

---

## Cómo ejecutar

```bash
pip install -r ../../requirements-notebook.txt
jupyter lab TimesFMOjoAlTicker.ipynb
```

Primera ejecución: ~10 min (descarga del checkpoint + backtest completo). Siguientes: ~2 min (precios y backtest cacheados en `data/`). Para forzar un backtest desde cero, borrar el parquet de `data/`.

Configuración relevante en la celda de constantes: `HORIZONS`, `BACKTEST_STEP`, `COST_BPS`, fechas pinneadas (`END_DATE`), y los pesos de cartera (`WEIGHTS` = Max Sharpe del notebook de MonteCarlo, solo producción; `EQUAL_WEIGHTS`, backtest).
