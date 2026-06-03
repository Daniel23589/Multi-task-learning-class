# Multi-task Learning Class

Proyecto de **clasificación de imágenes con aprendizaje multitarea**. El repositorio reúne varias partes de un flujo experimental para predecir información relacionada con imágenes de alimentos y explorar el uso de grafos.

## Objetivo

Comparar enfoques de aprendizaje multitarea para aprovechar relaciones entre etiquetas, clases y atributos dentro del conjunto de datos MAFood121.

## Archivos principales

- `mtlclass1.py`: primera parte del código de aprendizaje multitarea.
- `mtlclass2.py`: segunda parte del código de aprendizaje multitarea.
- `GNN.ipynb`: notebook con el flujo completo y una tercera parte centrada en redes neuronales de grafos (GNN).

## Conjunto de datos

Dentro de `MAFood121/` se organiza el conjunto de datos:

- `annotations/`: anotaciones de platos, grupos alimenticios y divisiones de entrenamiento, validación y prueba.
- `images/`: imágenes agrupadas por clases.

## Puesta en marcha

1. Crea un entorno de Python.
2. Instala las dependencias utilizadas por los scripts y el notebook.
3. Comprueba las rutas de acceso a `MAFood121/`.
4. Ejecuta los scripts en el orden correspondiente o abre el notebook:

```bash
python mtlclass1.py
python mtlclass2.py
jupyter notebook GNN.ipynb
```

## Recomendaciones

Para mejorar la reproducibilidad, conviene documentar los hiperparámetros, añadir un archivo `requirements.txt` y guardar los resultados de cada experimento en carpetas separadas.
