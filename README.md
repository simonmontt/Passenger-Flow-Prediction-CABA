# Predictor de Flujo de Pasajeros del Subte de Buenos Aires 🚇

Este proyecto tiene como objetivo predecir el flujo de pasajeros del subte de Buenos Aires, proporcionando una solución para optimizar la frecuencia de los vagones y reducir la congestión en las horas pico. Al conocer de antemano la cantidad de pasajeros que utilizarán el servicio, se pueden tomar decisiones informadas para gestionar mejor los recursos y garantizar una experiencia de viaje más cómoda. Entre otras posibles ventajas puede ser:

- **Mejora de la Planificación del Servicio**: Con las predicciones de pasajeros, se puede ajustar la programación de trenes para maximizar la eficiencia del servicio.
- **Optimización de Recursos**: Ayuda a reducir costos operativos al evitar el envío de trenes vacíos o con poca demanda.
- **Incremento en la Satisfacción del Cliente**: Al reducir la congestión, se mejora la experiencia del usuario, lo que puede llevar a un aumento en la utilización del servicio.

## 🗂️ Estructura del Proyecto

El proyecto se organiza en cuatro notebooks como recomienda la empresa ***Hopsworks***, cada una desempeñando un papel crucial en el flujo de trabajo:

1. **Backfill.ipynb**: 
   - Realiza el procesamiento inicial de los datos, incluyendo la descarga desde la web oficial del Gobierno de la Ciudad de Buenos Aires.
   - Los datos se suben a la feature store de Hopsworks, donde quedan disponibles para su posterior análisis.

2. **Feature Pipeline.ipynb**: 
   - Obtiene los datos de la última hora y los sube a la feature store, asegurando que siempre se tenga acceso a información actualizada para el entrenamiento del modelo.

3. **Training Pipeline.ipynb**: 
   - Estructura los datos en un formato adecuado para el entrenamiento del modelo XGBoost, que predice el flujo de pasajeros en las próximas tres horas.
   - El modelo se guarda en el registro de modelos de COMET_ML para su gestión y seguimiento.

4. **Inference.ipynb**: 
   - Realiza predicciones sobre el flujo de pasajeros tomando en cuenta los datos desde el tiempo actual hasta 14 días atrás.
   - Las predicciones se suben nuevamente a la feature store para ser consumidas por un frontend desarrollado en Streamlit.

## ⚙️ Implementación de MLOps

Para garantizar las mejores prácticas en MLOps, se utilizan **GitHub Actions** para automatizar el proceso de ejecución de las notebooks de features y inference cada hora, manteniendo así los datos actualizados.

Además, el proyecto utiliza un entorno **Poetry** para gestionar las dependencias y ejecutar el código de manera eficiente.

## 📊 Visualización

La visualización de los datos y predicciones se realiza utilizando **Streamlit Community Cloud**, donde se ha implementado un código en Python que obtiene las características actuales, predichas e históricas de los datos. Este frontend permite interactuar con los datos de manera sencilla y visual.

- Los datos actuales, desde mayo hasta la fecha, son simulados utilizando los datos del mismo período del año anterior, lo que permite evaluar el comportamiento del modelo con datos cercanos a la realidad actual.
