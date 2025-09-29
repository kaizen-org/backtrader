# Trading Algorítmico con Transformers y Backtrader

En este tutorial, construiremos y entrenaremos un modelo basado en la arquitectura Transformer para predecir los precios de las acciones y luego usaremos este modelo para crear una estrategia de trading algorítmico con `backtrader`.

## 1. Flujo de Trabajo

El proceso se divide en los siguientes pasos:

1.  **Cargar y Preparar los Datos**: Cargaremos los datos históricos que descargamos previamente (ej. `AAPL.csv`) y los prepararemos para el modelo Transformer. Esto incluye la normalización de los datos y la creación de secuencias de entrada.

2.  **Construir y Entrenar el Modelo Transformer**: Construiremos un modelo Transformer utilizando `tensorflow` y `keras`. Entrenaremos el modelo con los datos preparados para predecir el precio de cierre del siguiente día.

3.  **Guardar el Modelo Entrenado**: Una vez entrenado, guardaremos el modelo para poder cargarlo más tarde en nuestra estrategia de `backtrader`.

4.  **Crear la Estrategia en `backtrader`**: Crearemos una estrategia de `backtrader` que:
    *   Cargará el modelo Transformer entrenado.
    *   En cada paso del backtest, utilizará los datos más recientes para predecir el precio del día siguiente.
    *   Generará señales de compra/venta basadas en la predicción del modelo.

5.  **Ejecutar el Backtest**: Finalmente, ejecutaremos la estrategia en `backtrader` para evaluar su rendimiento.

## 2. Requisitos Previos

- **TensorFlow**: Necesitaremos la biblioteca `tensorflow` para construir y entrenar el modelo Transformer.
- **Datos Históricos**: Asegúrate de tener un archivo CSV con datos históricos (ej. `AAPL.csv` en el directorio `yahoo`).

## 3. Notebook del Tutorial

Todo el proceso se desarrollará en un notebook de Jupyter llamado `transformer_tutorial.ipynb`.
