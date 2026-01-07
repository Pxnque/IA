# Implementación de RAG con qwen2.5:3b

Para el Proyecto 3 se implementó un sistema de Recuperación Aumentada por Generación (RAG) utilizando el modelo **qwen2.5:3b**. A continuación se describen las etapas realizadas:

## Recopilación de Documentos

Se descargaron documentos en formato PDF relacionados con los siguientes temas:

- La posible crisis de sentido en la Generación Z debido a la hiperconectividad.
- La pérdida gradual de autonomía humana frente al avance de los algoritmos y la inteligencia artificial.

## Preprocesamiento del Texto

Los textos extraídos de los PDFs fueron sometidos a un proceso de limpieza y preparación:

- Extracción del contenido textual.
- Eliminación de palabras vacías (*stop words*).
- Segmentación del texto en fragmentos (*chunks*) de tamaño adecuado para su procesamiento por el modelo.

## Creación de la Base de Datos Vectorial

Se construyó una base de datos vectorial para almacenar y recuperar eficientemente los fragmentos de texto:

- Se utilizó **ChromaDB** como sistema de almacenamiento vectorial.
- La base de datos se integró con **AnythingLLM** a través del puerto **11347**.

## Resultados Obtenidos

Mediante la interfaz de AnythingLLM fue posible realizar consultas relacionadas con los temas de los documentos. El modelo:

- Generó respuestas coherentes y contextualizadas.
- Citó de forma precisa fragmentos relevantes de los textos proporcionados.

## Conclusión

La implementación del sistema RAG con el modelo qwen2.5:3b demostró ser efectiva para responder preguntas complejas basadas en fuentes documentales específicas, manteniendo fidelidad al contenido original y ofreciendo respuestas bien fundamentadas.