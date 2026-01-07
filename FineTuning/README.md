 # Proceso de Fine Tuning para que un modelo LLM sea tutor de programación

A diferencia del RAG que no modifica ni cambia el modelo, el proceso de fine tuning lo que hace es cambiar los pesos del modelo para que se ajuste a un tema más en específico, mejorando su desempeño en esa área designada. Para la parte de la explicación me hubiera gustado poner links a las imágenes de los procesos que hice, pero profe, como lo dejé para al último y no ando en Morelia, pues será por escrito.

# Como se hizo

- Primero se preguntó a diferentes IAs que generaran un set de preguntas y respuestas sobre programación básica en formato Alpaca, ya que ese formato era el que Axolotl aceptaba.
- Luego, para el ajuste de pesos, se utilizó Axolotl.ai, el cual no pude instalar de manera local, por lo que descargué la versión de Docker para poder usar el poder de mi GPU de escritorio, la cual es de 6 GB de VRAM.
- Ya con los pesos generados por Axolotl.ai, se combinó el `checkpoint12` con el modelo `qwen2.5:latest` para poder generar el archivo `.gguf`, el cual es el que Ollama puede leer.

# Resultados

Con la configuración inicial, el modelo se quedaba en un bucle respondiendo la misma pregunta y alucinando las respuestas. Para prevenir esto, se puso un límite en la cantidad de tokens usados por respuesta. Con esto, el modelo fue capaz de responder con la definición, un ejemplo y un ejercicio de la pregunta de programación deseada.