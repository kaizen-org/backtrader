# Trading Algorítmico con GRU y Backtrader

En este tutorial, construiremos y entrenaremos un modelo de Red Neuronal Recurrente con Gated Recurrent Unit (GRU) para predecir los precios de las acciones y luego usaremos este modelo para crear una estrategia de trading algorítmico con `backtrader`.

## 1. Flujo de Trabajo

El proceso se divide en los siguientes pasos:

1.  **Cargar y Preparar los Datos**: Cargaremos los datos históricos que descargamos previamente (ej. `AAPL.csv`) y los prepararemos para el modelo GRU. Esto incluye la normalización de los datos y la creación de secuencias de entrada.

2.  **Construir y Entrenar el Modelo GRU**: Construiremos un modelo GRU utilizando `tensorflow` y `keras`. Entrenaremos el modelo con los datos preparados para predecir el precio de cierre del siguiente día.

3.  **Guardar el Modelo Entrenado**: Una vez entrenado, guardaremos el modelo para poder cargarlo más tarde en nuestra estrategia de `backtrader`.

4.  **Crear la Estrategia en `backtrader`**: Crearemos una estrategia de `backtrader` que:
    *   Cargará el modelo GRU entrenado.
    *   En cada paso del backtest, utilizará los datos más recientes para predecir el precio del día siguiente.
    *   Generará señales de compra/venta basadas en la predicción del modelo.

5.  **Ejecutar el Backtest**: Finalmente, ejecutaremos la estrategia en `backtrader` para evaluar su rendimiento.

## 2. Requisitos Previos

- **TensorFlow**: Necesitaremos la biblioteca `tensorflow` para construir y entrenar el modelo GRU. El siguiente paso la instalará.
- **Datos Históricos**: Asegúrate de tener un archivo CSV con datos históricos (ej. `AAPL.csv` en el directorio `yahoo`).

## 3. Instalar TensorFlow

El siguiente comando instalará `tensorflow`:

```bash
!pip install tensorflow
```

## 4. Notebook del Tutorial

Todo el proceso se desarrollará en un notebook de Jupyter llamado `gru_tutorial.ipynb`.
