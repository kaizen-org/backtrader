# Descargar Datos Históricos de Interactive Brokers (IBKR)

En este tutorial, aprenderemos a descargar datos históricos de Interactive Brokers (IBKR) utilizando la API de IBKR para Python.

## 1. Requisitos Previos

- **TWS o IB Gateway en ejecución**: Debes tener la aplicación Trader Workstation (TWS) o IB Gateway de Interactive Brokers en ejecución en tu máquina. La API se conecta a una de estas aplicaciones para acceder a los datos del mercado.
- **API Habilitada en TWS/Gateway**: Asegúrate de que la API esté habilitada en la configuración de TWS o IB Gateway. Ve a `File -> Global Configuration -> API -> Settings` y marca la opción `Enable ActiveX and Socket Clients`.
- **Librería `ibapi` de IBKR**: Necesitamos la librería oficial de Python de IBKR. El siguiente paso la instalará.

## 2. Instalar la librería de IBKR

El siguiente comando instalará la librería `ibapi`:

```bash
!pip install ibapi
```

## 3. Script de Descarga de Datos

Crearemos un script de Python llamado `download_ibkr.py` que se conectará a TWS/Gateway y solicitará los datos históricos.

El script hará lo siguiente:
- Se conectará a TWS/Gateway.
- Definirá el contrato del instrumento que queremos descargar (ej. EUR.USD).
- Solicitará los datos históricos para un período de tiempo específico.
- Guardará los datos en un archivo CSV.

A continuación se muestra el código del script `download_ibkr.py`.
