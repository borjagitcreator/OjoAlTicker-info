# OjoAlTicker · Notebook de análisis de carteras

Este cuaderno nació como la base metodológica de **[OjoAlTicker](https://ojoalticker.com)**, una herramienta web de optimización de carteras sobre el S&P 500. Lo que empezó como prototipo en Python se convirtió en una aplicación completa; este notebook documenta la matemática y la lógica financiera que hay detrás.

Puedes usarlo de forma completamente independiente: no necesitas la app, solo Python y las dependencias del `requirements-notebook.txt`.

---

## Qué hace

1. **Descarga** los ~503 componentes actuales del S&P 500 (Wikipedia, dinámica) y 3 años de precios ajustados por dividendos y splits (yfinance).
2. **Screening por momentum**: filtra activos cuyo retorno compuesto creció estrictamente en períodos consecutivos. Este notebook usa el método anual (años naturales). La app de producción usa semestres móviles (3 bloques de 126 días de trading contados desde el final) para evitar el sesgo de comparar el año en curso parcial contra años completos.
3. **Test de normalidad** (Jarque-Bera + Shapiro-Wilk) con skewness y curtosis en exceso por activo.
4. **Monte Carlo vectorizado**: 100.000 carteras aleatorias con pesos uniformes sobre el símplex (Dirichlet, ver Metodología). Calcula Sharpe, Sortino, CVaR ↓ y CVaR ↑ para dos horizontes (63d y 252d).
5. **Optimización exacta** (scipy SLSQP): encuentra analíticamente Min Volatilidad, Max Sharpe y Max Sortino.

---

## Cómo ejecutar

```bash
pip install -r requirements-notebook.txt
jupyter lab "Inversión MonteCarlo.ipynb"
```

Tiempo estimado: ~5–10 minutos (descarga de ~500 tickers + 2×100K simulaciones Monte Carlo).

---

## Glosario de métricas

### Volatilidad (σ)
Desviación estándar de los retornos diarios, escalada al horizonte: `σ × √h`.  
Mide dispersión total — penaliza por igual movimientos al alza y a la baja.  
**Limitación crítica**: con fat tails, el intervalo ±1σ captura **menos del 68% real** de los escenarios. El riesgo real es mayor del que σ sugiere.

### CVaR₉₅ ↓ — Conditional Value at Risk (downside)
Pérdida media esperada en el **peor 5% de días** — **valor diario, sin escalar al horizonte**.  
Ejemplo: CVaR₉₅ ↓ = 3.2% → en su peor 5% de días, la cartera pierde de media un 3.2% **ese día**.  
Más informativo que σ cuando la distribución tiene fat tails, ya que captura el riesgo de cola que σ subestima.

### CVaR₉₅ ↑ — Conditional Value at Risk (upside)
Ganancia media en el **mejor 5% de días** — **valor diario, sin escalar al horizonte**.

**Por qué diario (sin `×√h`)**: escalar el riesgo de cola por la raíz del tiempo conflactaba un extremo de un solo día con un extremo del período entero, lo que inflaba la cifra y la hacía difícil de interpretar. Reportar el CVaR diario es directo de leer y comparar entre carteras y horizontes. La volatilidad (σ) y la downside deviation sí se siguen escalando con `√h`, porque ahí la regla de la raíz del tiempo es el convenio estándar para dispersión.

**Uso correcto**: comparar el ratio CVaR↓/CVaR↑ para evaluar la asimetría de la distribución diaria:
- Ratio > 1 → días extremos bajistas más intensos que los alcistas (sesgo negativo, habitual en acciones)
- Ratio ≈ 1 → distribución aproximadamente simétrica en las colas
- Ratio < 1 → días extremos alcistas superan a los bajistas

### Sharpe Ratio
```
Sharpe = (Retorno_cartera − Rf) / σ
```
Retorno en exceso sobre la tasa libre de riesgo por unidad de riesgo total.  
`Rf` = T-Bill 3M USA (`^IRX`), descargado dinámicamente.  
Interpretación: >1 eficiencia alta · 0–1 aceptable · <0 el T-Bill supera a la cartera.  
**Limitación**: penaliza la volatilidad al alza igual que la bajista. Si la distribución es asimétrica, Sortino es más informativo.

### Sortino Ratio
```
Sortino = (Retorno_cartera − Rf) / DD
DD = √(E[min(r − Rf, 0)²])   [Downside Deviation]
```
Solo penaliza retornos **por debajo de Rf** (downside deviation). La volatilidad alcista no cuenta.  
**Cuándo es mejor que Sharpe**: cuando la distribución tiene skewness < 0 (crashes > rallies). Si Sortino/Sharpe > 1.3×: la mayor parte de la volatilidad es alcista — perfil de riesgo favorable.

---

## Metodología

| Parámetro | Valor |
|-----------|-------|
| Universo | S&P 500 (~503 tickers, dinámico desde Wikipedia) |
| Histórico | 3 años de retornos diarios |
| Tipo de retorno | Simple `(Pt−Pt-1)/Pt-1` — linealmente agregable entre activos (`R_p = Σ wᵢRᵢ`, exacto) |
| Anualización (retorno) | Geométrica: `(1+μ_d)^h − 1` — captura el efecto compounding, más precisa que `μ_d × h` para h > 20 |
| Anualización (riesgo) | `σ_d × √h` — regla de la raíz del tiempo, asume retornos i.i.d. |
| Ordenación de tickers | **Mediana trimestral** (retornos de ~12 períodos reales) — más robusta que la media con distribuciones fat-tailed |
| Horizontes | Trimestral (63d) y Anual (252d) |
| Rf | T-Bill 3M USA (`^IRX`) — dinámica, descargada al inicio de cada sesión |
| Pesos mínimos | 2% por activo |
| Pesos máximos | `max(20%, 2/n)` — dinámico según nº de activos seleccionados |
| Muestreo de pesos | **Dirichlet(1,…,1)** = uniforme sobre el símplex (normalizar variables exponenciales). Mapa afín `w = mín + (1 − n·mín)·v` que impone el peso mínimo **sin rechazo**; solo se rechaza el límite superior. Normalizar uniformes `U(0,1)` **no** es uniforme — concentra los pesos hacia el centroide `1/n` y sesga la nube hacia la equiponderada. |

**Pesos máximos dinámicos:** con n = 6 activos: max = 33.3%; n = 8: 25%; n = 10: 20%. Permite más concentración cuando hay pocos activos, converge al límite estándar con carteras amplias.

---

## La Frontera Eficiente

El espacio de todas las carteras posibles forma una nube en el plano retorno/volatilidad. La **frontera eficiente** es el borde superior de esa nube: el conjunto de carteras que maximizan el retorno para cada nivel de volatilidad (o minimizan la volatilidad para cada retorno).

Tres puntos notables:
- **Vértice izquierdo** (Min Volatilidad): menor riesgo absoluto alcanzable.
- **Punto tangente con la CML** (Capital Market Line): cartera de Max Sharpe — combinación óptima de activo libre de riesgo + cartera de mercado.
- **Max Sortino**: óptimo cuando se prioriza la protección frente a caídas extremas sobre la eficiencia total.

El Monte Carlo visualiza la nube; scipy encuentra los puntos exactos mediante SLSQP. **Min Volatilidad** es convexo (mínimo global garantizado); **Max Sharpe** y **Max Sortino** no lo son, por lo que un único arranque puede caer en un óptimo local. En este notebook se usa un único arranque por simplicidad.

---

## Ejemplo — Cartera de referencia

> Datos del **17/06/2026** · `Rf = 3.63%` (T-Bill 3M) · 751 observaciones diarias · 498 activos con datos suficientes (de 503)

**6 activos por diversificación sectorial:**

| Ticker | Empresa | Sector |
|--------|---------|--------|
| MSFT | Microsoft | Tecnología |
| JPM | JPMorgan Chase | Financiero |
| JNJ | Johnson & Johnson | Salud |
| XOM | ExxonMobil | Energía |
| AMZN | Amazon | Consumo discrecional |
| NEE | NextEra Energy | Utilities |

**Correlaciones entre pares:**

| Par | ρ | Par | ρ |
|-----|---|-----|---|
| MSFT–AMZN | 0.53 | JPM–XOM | 0.22 |
| JPM–AMZN | 0.33 | MSFT–JNJ | −0.15 |
| JNJ–NEE | 0.28 | MSFT–NEE | −0.04 |
| MSFT–JPM | 0.24 | **media \|ρ\|** | **0.17** |

---

### Resultados — optimización exacta scipy (SLSQP)

**Horizonte Trimestral (63d)** · `Rf₆₃ ≈ 0.90%`

| Cartera | Retorno | Volatilidad | Sharpe |
|---------|---------|-------------|--------|
| Max Sharpe | **5.32%** | 6.48% | **0.683** |
| Max Sortino | 5.31% | 6.47% | 0.683 |
| Min Volatilidad | 4.52% | **6.04%** | 0.599 |

**Horizonte Anual (252d)** · `Rf = 3.63%`

| Cartera | Retorno | Volatilidad | Sharpe |
|---------|---------|-------------|--------|
| Max Sharpe | **23.06%** | 12.97% | **1.498** |
| Max Sortino | 23.02% | 12.95% | 1.497 |
| Min Volatilidad | 19.36% | **12.09%** | 1.302 |

> CVaR diario cartera Max Sharpe → CVaR↓ **1.85%**, CVaR↑ **1.75%**. Max Sharpe y Max Sortino convergen (diferencia máxima 0.9pp en NEE) — distribución aproximadamente simétrica. Los Sharpe anuales (~1.5) **no son comparables** con los trimestrales (~0.68): el retorno compuesto crece super-linealmente con h mientras la volatilidad escala con √h.

**Distribución de pesos — Max Sharpe (ambos horizontes):**

```
JPM    20.0%  ████████████████████
JNJ    20.0%  ████████████████████
XOM    20.0%  ████████████████████
AMZN   20.0%  ████████████████████
NEE    12.7%  █████████████
MSFT    7.3%  ███████
```

La optimización satura en el máximo permitido (20%) los 4 activos con correlaciones más bajas. MSFT recibe el menor peso por ser la correlación más alta del grupo (0.53 con AMZN).
