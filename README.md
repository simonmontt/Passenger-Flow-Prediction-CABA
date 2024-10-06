

Características

backfill.ipynb
Descarga de Datos: Descarga manual debido a inconsistencias en algunos meses, la próxima mejora sera en la descarga automática de datos 
Unificación de Datos: Combina los datos de conteo de pasajeros de varios años (2023 y 2024).
Limpieza de Datos: Corrige las inconsistencias en los nombres de las estaciones en el conjunto de datos.
Manejo de Datos Faltantes: Asegura que haya registros horarios continuos para todas las estaciones y líneas del subterráneo, completando las marcas de tiempo faltantes.
Integración con Hopsworks: Sube los datos de series temporales procesados a una feature store en Hopsworks, donde pueden utilizarse para entrenar modelos de machine learning.