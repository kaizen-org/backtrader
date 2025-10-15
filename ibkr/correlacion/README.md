# Modelo de Correlación para Predicción de Rentabilidad

Este documento explica el funcionamiento del sistema implementado en esta carpeta, cuyo objetivo es predecir los 5 tickers con mayor potencial de rentabilidad para la semana siguiente.

## Visión General

El sistema se compone de tres scripts principales que deben ejecutarse en orden:

1.  **`data_prep_correlation.py`**: Prepara los datos.
2.  **`train_correlation_model.py`**: Entrena el modelo de red neuronal.
3.  **`predict_top_5.py`**: Utiliza el modelo para generar las predicciones.

## Flujo de Trabajo

### 1. Preparación de Datos (`data_prep_correlation.py`)

Este script es el primer paso y se encarga de consolidar y enriquecer los datos históricos.

-   **Fuente de datos**: Lee todos los archivos de datos históricos (`.csv`) ubicados en la carpeta `../historic`.
-   **Enriquecimiento**: Para cada ticker, calcula una serie de indicadores técnicos que servirán como `features` (características) para el modelo. Estos incluyen:
    -   RSI (Relative Strength Index)
    -   MACD (Moving Average Convergence Divergence)
    -   Bandas de Bollinger
    -   Precio relativo a las medias móviles (50 y 200 días)
    -   Volatilidad histórica
-   **Cálculo del Target**: La variable a predecir (`target`) es la rentabilidad futura a 5 días (`future_return_5d`). Se calcula como el cambio porcentual entre el precio de cierre actual y el precio de cierre 5 días en el futuro.
-   **Salida**: Todos los datos enriquecidos de todos los tickers se guardan en un único archivo: `all_enriched_data.csv`.

### 2. Entrenamiento del Modelo (`train_correlation_model.py`)

Una vez que los datos están preparados, este script entrena una red neuronal de tipo GRU (Gated Recurrent Unit).

-   **Carga de datos**: Lee `all_enriched_data.csv`.
-   **Creación de secuencias**: Las redes neuronales recurrentes trabajan con secuencias. El script transforma los datos de cada ticker en secuencias de una longitud definida (`SEQUENCE_LENGTH`), donde cada secuencia de `features` se asocia a una rentabilidad futura.
-   **División y Escalado**: Los datos se dividen en conjuntos de entrenamiento y validación. Luego, las `features` se normalizan (escalan) para que el modelo pueda procesarlas eficientemente. El `scaler` utilizado se guarda para usarlo después en la predicción.
-   **Arquitectura del Modelo**: Se define un modelo GRU secuencial con varias capas, diseñado para problemas de regresión (predecir un valor numérico).
-   **Entrenamiento**: El modelo se entrena con los datos de entrenamiento, buscando minimizar el error cuadrático medio (`mean_squared_error`) entre sus predicciones y los valores reales.
-   **Salida**: 
    -   El mejor modelo entrenado se guarda como `correlation_model.h5`.
    -   El normalizador de datos se guarda como `correlation_scaler.joblib`.

### 3. Predicción del Top 5 (`predict_top_5.py`)

Este es el script final que utiliza el trabajo de los dos anteriores para generar el resultado deseado.

-   **Carga**: Carga el modelo entrenado (`correlation_model.h5`) y el `scaler` (`correlation_scaler.joblib`).
-   **Procesamiento**: Para cada ticker disponible en `all_enriched_data.csv`, toma la última secuencia de datos disponible.
-   **Predicción**: Prepara y escala esta última secuencia de la misma forma que en el entrenamiento y la introduce en el modelo para obtener una predicción de la rentabilidad a 5 días.
-   **Ranking y Salida**: Recopila las predicciones de todos los tickers, las ordena de mayor a menor y muestra en pantalla los 5 tickers con la rentabilidad predicha más alta. Según las premisas, este script se ejecutaría los miércoles para planificar la semana siguiente.

## Cómo Usarlo

Para obtener una nueva predicción desde cero, sigue estos pasos en orden desde la carpeta `ibkr/correlacion`:

1.  **Ejecuta el script de preparación de datos** para asegurar que tienes la información más reciente:
    ```bash
    python data_prep_correlation.py
    ```

2.  **Entrena el modelo** con los datos actualizados:
    ```bash
    python train_correlation_model.py
    ```

3.  **Obtén las predicciones** del top 5:
    ```bash
    python predict_top_5.py
    ```
