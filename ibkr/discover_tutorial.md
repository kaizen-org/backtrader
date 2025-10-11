# Descubrir y Descargar Contratos de IBKR

Como no es posible consultar todos los instrumentos disponibles en Interactive Brokers de una sola vez, hemos creado un script que te permite buscar contratos basados en criterios específicos y luego descargar los datos para cada uno de ellos.

## 1. El Script `discover_and_download.py`

Este script realiza dos tareas principales:

1.  **Descubre Contratos**: Se conecta a la API de IBKR y busca todos los contratos que coincidan con los criterios que le proporciones (como tipo de seguridad, exchange y moneda).
2.  **Descarga Datos**: Para cada contrato encontrado, ejecuta automáticamente el script `download_ibkr.py` para descargar el historial de datos de ese contrato.

## 2. ¿Cómo Usarlo?

Debes ejecutar el script desde la línea de comandos, proporcionando los criterios de búsqueda que desees. El argumento más importante es `--sectype`.

### Ejemplo: Descargar todas las Acciones (STK) en USD del exchange SMART

El siguiente comando buscará todos los contratos de acciones (`STK`) en el exchange `SMART` con `USD` como moneda y descargará los datos para cada uno:

```bash
python ibkr/discover_and_download.py --sectype STK --exchange SMART --currency USD
```

### Argumentos del Script

- `--sectype` (Obligatorio): El tipo de seguridad que quieres buscar. Ejemplos: `STK` (acciones), `CASH` (divisas), `FUT` (futuros), `IND` (índices).
- `--exchange` (Opcional, por defecto `SMART`): El exchange en el que quieres buscar.
- `--currency` (Opcional, por defecto `USD`): La moneda de los contratos que quieres buscar.
- `--host` (Opcional, por defecto `127.0.0.1`): El host donde se ejecuta TWS o IB Gateway.
- `--port` (Opcional, por defecto `7497`): El puerto de TWS o IB Gateway.
