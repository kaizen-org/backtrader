# Entendiendo el Flujo de Trabajo: Entrenamiento vs. Backtesting

Al integrar un modelo de Machine Learning (como una GRU) con una herramienta de backtesting (como `backtrader`), es fundamental separar dos fases distintas: la **fase de entrenamiento** y la **fase de backtesting**.

### 1. Fase de Entrenamiento (Offline)

El objetivo de esta fase es enseñar al modelo a reconocer patrones.

- **Datos:** Se utiliza un conjunto de datos históricos estático (por ejemplo, un fichero CSV).
- **Proceso:**
    1.  Cargamos los datos en un `DataFrame` de `pandas`.
    2.  **Preparamos las características (Features):** Aquí es donde añadimos los indicadores técnicos (RSI, MACD, etc.). Como estamos trabajando con `pandas`, usamos bibliotecas compatibles como `pandas-ta` que están optimizadas para operar sobre `DataFrames` completos de forma eficiente.
    3.  Se crea un conjunto de datos de "secuencias", donde cada muestra es una ventana de tiempo (ej. 60 días) de todas las características (precio + indicadores), y la etiqueta es el precio del día siguiente.
    4.  El modelo GRU se entrena con estas secuencias.
- **Resultado:** Un fichero de modelo entrenado (ej. `gru_model.h5`) que ha aprendido los patrones de los datos históricos enriquecidos.

### 2. Fase de Backtesting (Simulación Dinámica)

El objetivo aquí es simular cómo habría operado el modelo entrenado en el pasado, día a día.

- **Datos:** `backtrader` gestiona los datos, entregándolos a la estrategia barra a barra (un día a la vez).
- **Proceso:**
    1.  La estrategia de `backtrader` se inicializa. Carga el modelo ya entrenado (`gru_model.h5`).
    2.  Dentro de la estrategia, **replicamos los mismos indicadores técnicos** que usamos en el entrenamiento, pero esta vez utilizando las funciones nativas de `backtrader` (ej. `bt.indicators.RSI`). Estos indicadores se actualizan automáticamente con cada nueva barra de datos que llega.
    3.  En cada paso (`next`):
        - La estrategia recoge los últimos 60 valores del precio y de cada uno de los indicadores de `backtrader`.
        - Construye una secuencia de entrada con la forma exacta que el modelo espera `(60, num_features)`.
        - Alimenta esta secuencia al modelo para obtener una predicción para el día siguiente.
        - Ejecuta la lógica de compra/venta basada en esa predicción.

### Conclusión

No podemos usar los indicadores de `backtrader` para el entrenamiento porque operan en un entorno dinámico, barra a barra, mientras que el entrenamiento necesita un conjunto de datos pre-procesado completo.

El enfoque correcto es:
- **Preparar datos para entrenamiento** con herramientas de `pandas` (como `pandas-ta`).
- **Simular la operativa** usando los indicadores equivalentes de `backtrader` para alimentar al modelo en cada paso del backtest.

Esto asegura que el modelo opera en la simulación con el mismo tipo de información con el que fue entrenado.
