# README

## Características

**backfill.ipynb**  

**Descarga de Datos:** Descarga manual debido a inconsistencias en algunos meses. La próxima mejora será en la descarga automática de datos.

**Unificación de Datos:** Combina los datos de conteo de pasajeros de varios años (2023 y 2024).

**Limpieza de Datos:** Corrige las inconsistencias en los nombres de las estaciones en el conjunto de datos.

**Manejo de Datos Faltantes:** Asegura que haya registros horarios continuos para todas las estaciones y líneas del subterráneo, completando las marcas de tiempo faltantes.

**Integración con Hopsworks:** Sube los datos de series temporales procesados a una feature store en Hopsworks, donde pueden utilizarse para entrenar modelos de machine learning.

---

**Carga del Archivo CSV:** Carga un archivo CSV (222324_together_total_pax.csv) que contiene los datos de pasajeros del subte (hour_of_entry, line, station, y total_pax).

**Preparación de Datos:** El script maneja datos faltantes de 2024 desplazando los datos de 2023 hacia adelante por un año, y del mismo modo desplaza los datos de 2022 para cubrir vacíos en los datos de 2023. Se retienen los datos originales de 2023 y 2024 para asegurar que se cubran todos los períodos relevantes para una predicción y análisis precisos.

**Combinación de Datos:** Combina los datos originales y desplazados de 2022 a 2024, asegurando que no haya duplicados y ordena los datos por hour_of_entry.

**Filtrado Final:** Se crean dos conjuntos de datos: uno que cubre los últimos 18 días de 2024, y otro del mismo período en 2023, desplazado por 365 días. Estos conjuntos de datos se concatenan para crear un conjunto de datos final de series temporales para el entrenamiento de modelos.

**Integración con el Almacén de Características de Hopsworks:** El script se conecta a un proyecto de Hopsworks y carga los datos procesados en un grupo de características dentro del almacén de características para su uso en futuros pipelines de aprendizaje automático.

---

**training_pipeline.ipynb:**  

**Conexión a Hopsworks:** Se conecta al proyecto de Hopsworks y al almacén de características para acceder y manipular los datos.

**Creación y Acceso a la Vista de Características:** Se crea una vista de características si no existe y se obtienen los datos necesarios para el entrenamiento. Estos datos son horarios de flujo total de pasajeros.

**Preprocesamiento de Datos:** Se ordenan los datos por hour_of_entry y se transforman en un formato adecuado para el modelado. Se aplica codificación de etiquetas a las columnas 'line' y 'station'.

**Transformación de Datos en Características y Objetivos:** Se transforma el conjunto de datos en características y objetivos utilizando una función personalizada (transform_ts_data_into_features_and_target), configurando una ventana de entrada de 14 días y un tamaño de paso de una hora.

**División de Datos:** Se divide el conjunto de datos en conjuntos de entrenamiento y prueba utilizando una fecha de corte definida.

**Optimización de Hiperparámetros:** Utilizando Optuna, se define una función objetivo para entrenar el modelo con diferentes configuraciones de hiperparámetros y se evalúa el rendimiento a través de validación cruzada con TimeSeriesSplit.

**Entrenamiento y Evaluación del Modelo:** Se entrena el modelo utilizando los mejores hiperparámetros y se evalúa mediante el cálculo del error absoluto medio (MAE).

**Guardado y Carga del Modelo:** El modelo entrenado se guarda en un archivo .pkl para su posterior uso, y también se proporciona un mecanismo para cargar el modelo guardado.

**Registro del Modelo:** Se registra el modelo en un sistema de gestión de modelos (como Comet ML) para un seguimiento y administración más efectivos.

---

**Inference.ipynb**  

**Configuración del Entorno:** Se habilita la recarga automática para facilitar el desarrollo iterativo. Se importan las bibliotecas necesarias, como pytz, pandas y módulos personalizados.

**Manejo de la Hora Actual:** La canalización establece la zona horaria de Argentina (GMT-3) y obtiene la fecha y hora actual, redondeándola a la hora más cercana.

**Carga de Características:** Las características se cargan desde Hopsworks utilizando la función load_batch_of_features_from_store. Después de la carga, se codifican las características categóricas (line y station) usando LabelEncoder.

**Carga del Modelo:** Se recupera el modelo más reciente del registro de modelos, específicamente la versión de producción del modelo de predicción de flujo de pasajeros.

**Realización de Predicciones:** Se generan predicciones usando el modelo cargado y las características procesadas. Se agrega la fecha actual al DataFrame de predicciones.

**Almacenamiento de Predicciones:** Se crea un grupo de características para las predicciones del modelo en Hopsworks (o se recupera si ya existe). Las predicciones se insertan en el grupo de características con las claves primarias y el tiempo del evento apropiados.
