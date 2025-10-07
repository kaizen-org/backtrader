
# Guía Operacional: Uso y Reentrenamiento del Modelo de Trading GRU

## Introducción

Este documento describe el proceso para utilizar el sistema de trading basado en GRU en un entorno real (o simulado en tiempo real) y cómo mantener su relevancia a lo largo del tiempo mediante el reentrenamiento periódico.

---

## Parte 1: Cómo Usar el Modelo para Invertir (Modo "Live")

El objetivo es utilizar nuestro sistema para tomar una decisión de trading para el día siguiente. El proceso se debe ejecutar al final de cada día de mercado.

### Requisitos

1.  **Modelo Entrenado:** `gru_strategy_selector.h5`
2.  **Escalador de Datos:** `gru_scaler.joblib`
3.  **Fuente de Datos Diaria:** Un método para obtener los datos OHLCV del día que acaba de cerrar para los activos que sigues (ej. 'A' y 'ABM').
4.  **Historial de Datos:** El fichero `enriched_data.csv` o una base de datos con la misma estructura.

### Proceso Diario (Paso a Paso)

Al cierre de cada día de mercado, sigue estos pasos:

**1. Obtener Datos Recientes:**
   - Descarga los datos del día para tus activos.

**2. Actualizar Indicadores:**
   - Añade los nuevos datos a tu historial.
   - Vuelve a ejecutar una versión del script `data_preparation.py` para calcular los indicadores técnicos (SMA, RSI, etc.) para todo el conjunto de datos actualizado. Es crucial que los cálculos se hagan sobre el historial completo para que los indicadores sean precisos.

**3. Crear la Secuencia de Inferencia:**
   - Una vez actualizados los indicadores, toma los últimos **30 registros** (el `SEQUENCE_LENGTH` que usamos) de los datos de indicadores para el activo que quieres analizar.

**4. Normalizar la Secuencia:**
   - Carga el escalador con `joblib.load('gru_scaler.joblib')`.
   - Aplica el escalador a tu secuencia de 30 días. **Importante:** Usa `scaler.transform()`, **NUNCA** `scaler.fit_transform()`, ya que debes usar la misma escala con la que el modelo fue entrenado.

**5. Hacer la Predicción:**
   - Carga el modelo con `load_model('gru_strategy_selector.h5')`.
   - Llama a `model.predict()` sobre tu secuencia normalizada.

**6. Interpretar y Actuar:**
   - El modelo devolverá un valor entre 0 y 1.
     - **Si es > 0.5:** El modelo recomienda **Momentum**.
     - **Si es <= 0.5:** El modelo recomienda **Reversión a la Media**.
   - Con esta decisión, comprueba las condiciones de tu estrategia para el día siguiente. Por ejemplo, si el modelo dice "Momentum", comprueba si la SMA de 50 días está por encima de la de 200. Si se cumple, puedes colocar una orden de compra.

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
