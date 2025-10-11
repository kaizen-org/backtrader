
# Guía Operacional: Uso y Reentrenamiento del Modelo de Trading GRU

## Introducción

Este documento describe el proceso para utilizar el sistema de trading basado en GRU en un entorno real (o simulado en tiempo real) y cómo mantener su relevancia a lo largo del tiempo mediante el reentrenamiento periódico.

---

## Parte 1: Cómo Usar el Modelo para Invertir (Modo "Live")

Para obtener una recomendación de inversión para el día siguiente, hemos automatizado el proceso en el script `live_prediction.py`.

### Proceso Diario (Paso a Paso)

Al cierre de cada día de mercado, sigue estos pasos:

**1. Actualizar Datos Históricos:**
   - Primero, asegúrate de que tu fuente de datos principal (ej. los ficheros CSV en `ibkr/`) contiene los datos OHLCV del día que acaba de cerrar.
   - Luego, ejecuta el script de preparación de datos para recalcular los indicadores técnicos y actualizar el fichero `enriched_data.csv`.
     ```bash
     python data_preparation.py
     ```
   Este paso es **crucial** para que el modelo tenga la información más reciente.

**2. Ejecutar el Script de Predicción en Vivo:**
   - Una vez que `enriched_data.csv` está actualizado, ejecuta `live_prediction.py` especificando el activo que quieres analizar con el parámetro `--ticker`.

     ```bash
     # Ejemplo para el activo 'A'
     python live_prediction.py --ticker A
     ```

### Interpretar y Actuar

El script imprimirá en la consola la estrategia recomendada por el modelo para el siguiente día de mercado. Por ejemplo:

```
==================================================
RESULTADO DE LA PREDICCIÓN PARA 'A'
Valor predicho por el modelo: 0.8123
Estrategia recomendada para el próximo día de mercado: **Momentum**
==================================================
```

- **Si recomienda Momentum:** Comprueba las condiciones de tu estrategia de momentum (ej. `SMA50 > SMA200`). Si se cumplen, puedes plantear una orden de compra.
- **Si recomienda Reversión a la Media:** Comprueba las condiciones de tu estrategia de reversión (ej. `precio < banda de Bollinger inferior`). Si se cumplen, puedes plantear una orden de compra.

### ¿Qué Hace el Script `live_prediction.py` Internamente?

El script `live_prediction.py` sigue exactamente el proceso conceptual que se necesita para una predicción:
1.  **Carga Datos:** Lee el fichero `enriched_data.csv` actualizado.
2.  **Filtra y Selecciona:** Se queda con los datos del `ticker` solicitado y extrae los últimos **30 registros** (la longitud de la secuencia con la que fue entrenado el modelo).
3.  **Normaliza:** Carga el escalador (`gru_scaler.joblib`) y normaliza los 30 registros. Es importante destacar que usa la transformación ya aprendida, sin reajustar el escalador.
4.  **Predice:** Carga el modelo (`gru_strategy_selector.h5`) y le pasa la secuencia normalizada para obtener un valor entre 0 y 1.
5.  **Informa:** Traduce ese valor a una recomendación (Momentum o Reversión a la Media) y la muestra en pantalla.

---

## Parte 2: Cómo Reentrenar el Modelo Periódicamente

### ¿Por Qué Reentrenar?

Los mercados financieros cambian constantemente. Un modelo entrenado con datos de 2021-2024 puede no ser efectivo en 2026 porque las dinámicas del mercado han cambiado. Este fenómeno se llama **"Model Drift"** o deriva del modelo. El reentrenamiento periódico ajusta el modelo a los datos más recientes para mantener su poder predictivo.

### Frecuencia de Reentrenamiento

No hay una respuesta única, pero un buen punto de partida es reentrenar el modelo de forma **trimestral o semestral**.

### Proceso de Reentrenamiento (Paso a Paso)

El proceso es sorprendentemente simple, ya que consiste en volver a ejecutar los scripts que ya hemos creado sobre un conjunto de datos más grande.

**1. Acumular Nuevos Datos:**
   - Asegúrate de que tu fuente de datos históricos ha sido actualizada con todos los datos desde el último reentrenamiento.

**2. Ejecutar el Pipeline Completo:**
   - **Paso A: Ejecuta `data_preparation.py`**. Este script tomará todo tu historial (el antiguo más el nuevo), calculará los indicadores y, lo más importante, generará la variable objetivo `estrategia_optima` para todos los datos. Creará un `enriched_data.csv` actualizado.
   - **Paso B: Ejecuta `train_gru_model.py`**. Este script automáticamente:
     - Cargará el nuevo `enriched_data.csv`.
     - Lo dividirá de nuevo en un 80% para entrenamiento y un 20% para validación (el 80% ahora incluirá datos más recientes).
     - Entrenará un **nuevo modelo** desde cero.
     - Guardará los nuevos `gru_strategy_selector.h5` y `gru_scaler.joblib`, **sobrescribiendo los antiguos**.

**3. Puesta en Producción:**
   - ¡Listo! A partir de este momento, tu proceso diario de "Modo Live" usará automáticamente el modelo y el escalador recién entrenados, ya que apuntan a los mismos nombres de fichero.

### Consideraciones Importantes

- **Backup:** Antes de reentrenar, es una buena práctica hacer una copia de seguridad de los ficheros `.h5` y `.joblib` que estaban funcionando bien. Si por alguna razón el nuevo modelo funciona peor, puedes volver a la versión anterior.
- **Evaluación Continua:** Después de cada reentrenamiento, fíjate en la "Precisión (Accuracy) en el conjunto de validación" que muestra el script. Si notas que la precisión decae consistentemente con cada reentrenamiento, es una señal de que el mercado ha cambiado tanto que las características (indicadores) que usas pueden ya no ser relevantes, y sería el momento de aplicar las mejoras que discutimos (añadir nuevos indicadores, cambiar la arquitectura del modelo, etc.).

---

## Parte 3: Evaluación de la Estrategia con Backtesting

Una vez que tenemos un modelo entrenado, es fundamental evaluar su rendimiento histórico antes de arriesgar capital real. Este proceso se conoce como **backtesting**.

### Propósito

El backtesting simula la ejecución de nuestra estrategia de trading sobre datos históricos para medir su rentabilidad, riesgo y otras métricas de rendimiento. Nos permite responder a la pregunta: "¿Qué tan bien habría funcionado esta estrategia en el pasado?"

### El Flujo de Trabajo de Backtesting

El proceso se divide en dos scripts principales para maximizar la eficiencia:

**1. `predict_strategies.py`**
   - Este script es el primer paso. Carga el modelo GRU entrenado (`gru_strategy_selector.h5`) y el escalador (`gru_scaler.joblib`).
   - Procesa el conjunto de datos de validación (el 20% que no se usó para entrenar) y, para cada día, predice qué estrategia se debería usar (Momentum o Reversión a la Media).
   - Guarda estas decisiones en un archivo llamado `predictions.csv`. Este archivo actúa como una "hoja de ruta" de señales de trading para el backtest.

**2. `run_backtest.py`**
   - Este es el script que ejecuta la simulación histórica.
   - **Referencia:** `run_backtest.py`
   - Carga los datos históricos de los activos (por ejemplo, de la carpeta `ibkr/`).
   - Carga el archivo `predictions.csv` generado en el paso anterior.
   - Utiliza la librería `backtrader` para simular las operaciones:
     - Día a día, consulta `predictions.csv` para ver la señal del modelo.
     - Si la señal es "Momentum", aplica las reglas de compra/venta de la estrategia de momentum (ej. cruce de medias móviles).
     - Si la señal es "Reversión", aplica las reglas de la estrategia de reversión (ej. Bandas de Bollinger).
   - Al finalizar, informa el capital final de la cartera y genera un gráfico (`backtest_plot.png`) que muestra las operaciones realizadas sobre la evolución del precio.

### ¿Por Qué Separar la Predicción del Backtesting?

Separar estos dos pasos es una optimización clave. Cargar y ejecutar un modelo de TensorFlow (`.h5`) es un proceso computacionalmente costoso. Si lo hiciéramos dentro del bucle de `backtrader` (que itera sobre cientos o miles de días), el backtest sería extremadamente lento.

Al pre-calcular todas las decisiones del modelo y guardarlas en un archivo CSV simple, el script `run_backtest.py` solo necesita leer una línea de un archivo de texto en cada paso, lo que lo hace increíblemente rápido y eficiente.
