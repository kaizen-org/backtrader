# Plan de Trading Híbrido con GRU: Momentum y Reversión a la Media

## Objetivo

El objetivo es construir y validar un sistema de backtesting que opere sobre los activos **A.USD** y **ABM.USD**. El sistema utilizará una estrategia híbrida que combinará dos enfoques clásicos:

1.  **Estrategia de Momentum:** Comprar activos que han tenido un buen rendimiento reciente y vender los que han tenido un mal rendimiento.
2.  **Estrategia de Reversión a la Media:** Comprar activos que han caído por debajo de su media histórica y vender los que han subido muy por encima, esperando que los precios "reviertan" a su valor promedio.

La selección entre una estrategia y otra no será fija, sino que será decidida en tiempo real por un modelo de Machine Learning, específicamente una Red Neuronal Recurrente de tipo **GRU (Gated Recurrent Unit)**.

## Fases del Proyecto

El proyecto se dividirá en tres fases principales:

1.  **Fase 1: Preparación y Enriquecimiento de Datos**
2.  **Fase 2: Construcción y Entrenamiento del Modelo GRU**
3.  **Fase 3: Implementación y Backtesting con `backtrader`**

---

### Fase 1: Preparación y Enriquecimiento de Datos

En esta fase, prepararemos los datos históricos para que puedan ser utilizados tanto por las estrategias de trading como por el modelo GRU.

**1.1. Carga de Datos:**
*   Se cargarán los ficheros `A.USD.csv` y `ABM.USD.csv` en un DataFrame de `pandas`.
*   Se combinarán ambos DataFrames en uno solo, añadiendo una columna `ticker` para identificar a qué activo pertenece cada registro.

**1.2. Cálculo de Indicadores Técnicos:**
Para que las estrategias y el modelo tengan contexto, enriqueceremos los datos con los siguientes indicadores técnicos para cada activo:
*   **Medias Móviles Simples (SMA):** Una corta (20 días), una media (50 días) y una larga (200 días) para identificar tendencias.
*   **RSI (Relative Strength Index):** Para medir la fuerza de una tendencia y detectar condiciones de sobrecompra/sobreventa (periodo de 14 días).
*   **Bandas de Bollinger:** Para medir la volatilidad y identificar precios "extremos" relativos a su media (periodo de 20 días, 2 desviaciones estándar).
*   **MACD (Moving Average Convergence Divergence):** Para identificar cambios en el momentum.
*   **ATR (Average True Range):** Para medir la volatilidad del mercado.
*   **Retornos Diarios:** `close.pct_change()` para tener una medida de la variación diaria.

**1.3. Definición de la Variable Objetivo (Target):**
Esta es la parte más importante para la GRU. Necesitamos "etiquetar" nuestros datos históricos para que el modelo pueda aprender.
*   Crearemos una variable objetivo llamada `estrategia_optima`.
*   Para cada día `D` en nuestros datos, miraremos hacia el futuro (ej. los próximos 5 días, `D+1` a `D+5`).
*   Calcularemos el retorno hipotético que habría generado una estrategia de momentum simple en ese periodo.
*   Calcularemos el retorno hipotético que habría generado una de reversión a la media.
*   La `estrategia_optima` para el día `D` será:
    *   `1` (Momentum) si la estrategia de momentum fue más rentable.
    *   `0` (Reversión a la Media) si la reversión a la media fue más rentable.
*   De esta forma, convertimos el problema en una **clasificación binaria**: la GRU aprenderá a predecir, basándose en los indicadores de los últimos días, si los próximos días serán más favorables para momentum o para reversión a la media.

---

### Fase 2: Modelo GRU para Selección de Estrategia

Con los datos ya etiquetados, construiremos el modelo.

**2.1. Preparación de Secuencias:**
*   Las GRU trabajan con secuencias de datos. Crearemos secuencias de una longitud determinada (ej. 30 días de datos de indicadores) para predecir la `estrategia_optima` del día 31.

**2.2. División y Normalización de Datos:**
*   Dividiremos el conjunto de datos cronológicamente en un 80% para entrenamiento y un 20% para validación.
*   Normalizaremos todos los indicadores (features) usando `MinMaxScaler` para que estén en un rango de [0, 1], lo cual es óptimo para las redes neuronales.

**2.3. Construcción y Entrenamiento del Modelo GRU:**
*   Usaremos `tensorflow.keras` para definir la arquitectura del modelo.
    *   Una o dos capas GRU.
    *   Capas de `Dropout` para prevenir el sobreajuste.
    *   Una capa de salida `Dense` con activación `sigmoid` para producir una probabilidad (un valor entre 0 y 1).
*   Compilaremos el modelo con el optimizador `adam` y la función de pérdida `binary_crossentropy`.
*   Entrenaremos el modelo con los datos de entrenamiento y lo guardaremos en un fichero (`gru_strategy_selector.h5`).

---

### Fase 3: Implementación y Backtesting con `backtrader`

Finalmente, usaremos el modelo entrenado dentro de una estrategia de `backtrader`.

**3.1. Cargar Modelo y Scaler:**
*   La estrategia de `backtrader` cargará el fichero `gru_strategy_selector.h5` y el `scaler` utilizado durante el entrenamiento.

**3.2. Crear la Estrategia Híbrida en `backtrader`:**
*   Definiremos una nueva clase de estrategia que herede de `bt.Strategy`.
*   En el método `__init__`, definiremos todos los indicadores de `backtrader` que se correspondan con los que usamos para entrenar el modelo (SMA, RSI, Bollinger, etc.).
*   En el método `next()` (que se ejecuta para cada barra de datos):
    1.  Recogeremos los valores de los indicadores de los últimos 30 días para formar una secuencia.
    2.  Normalizaremos esta secuencia con el `scaler` cargado.
    3.  Alimentaremos la secuencia a nuestro modelo GRU para obtener una predicción.
    4.  **Decisión:**
        *   Si la predicción > 0.5, el modelo elige **Momentum**. La estrategia ejecutará lógica de momentum: por ejemplo, comprar si la media corta cruza por encima de la larga.
        *   Si la predicción <= 0.5, el modelo elige **Reversión a la Media**. La estrategia ejecutará lógica de reversión: por ejemplo, comprar si el precio toca la Banda de Bollinger inferior.
    5.  Se implementará la lógica de compra/venta para cada una de las dos sub-estrategias, que solo se activarán si el modelo GRU las selecciona.

**3.3. Configurar y Ejecutar el Backtest:**
*   Crearemos una instancia de `backtrader.Cerebro`.
*   Añadiremos los data feeds para `A.USD` y `ABM.USD`.
*   Añadiremos nuestra estrategia híbrida.
*   Configuraremos el capital inicial, comisiones y slippage.
*   Ejecutaremos el backtest con `cerebro.run()`.
*   Analizaremos los resultados: capital final, drawdown, y visualizaremos el gráfico de la operación con `cerebro.plot()`.
