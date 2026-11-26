# Stanford Dogs: procesamiento de bordes

Proyecto independiente de LabTrack para la Fase 1 del trabajo. Descarga una
muestra reproducible de 5,000 imágenes del dataset Stanford Dogs, procesa cada
imagen y combina los resultados con un mosaico.

El dataset completo se obtiene del [sitio de Stanford Dogs](https://vision.stanford.edu/aditya86/ImageNetDogs/).
La descarga se transmite directamente al extractor; solo se conservan las
5,000 imágenes de la muestra. Los datos no se incluyen en Git.

## Requisitos

- Python 3.10 o superior
- Docker Engine 24 o superior para construir la imagen
- Conexión a Internet y espacio libre para las imágenes seleccionadas y sus
  resultados

## Preparar el dataset

Desde la raíz de este proyecto:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python download_dataset.py
```

El script hace una selección aleatoria reproducible con semilla `1208` y guarda
las imágenes en `datos/entrada/stanforddogs/`. Genera `datos/dataset.env` y
`datos/dataset_report.json`, con el número de fotos, sus tamaños y dimensiones.
No sobrescribe una descarga que ya exista.

## Procesamiento local con cuatro índices

Cada worker lista el mismo conjunto de imágenes y procesa un bloque
determinista. El pipeline es escala de grises, desenfoque gaussiano, Sobel y
umbral.

```bash
for index in 0 1 2 3; do
  python worker.py \
    --index "$index" \
    --total-tasks 4 \
    --input-dir datos/entrada/stanforddogs \
    --output-dir datos/salida/demo
done

python merge.py --results-dir datos/salida/demo --expected-tasks 4
```

El worker escribe `part-XXXX.csv` y `part-XXXX.json`; el merge produce
`resultados.csv`, `summary.json` y un `mosaico.png` de hasta 25 resultados.
Los errores de lectura de una imagen quedan anotados en el CSV y no detienen el
procesamiento de las demás.

## Construir y probar Docker

```bash
docker build -t bordes:1.0 .
docker run --rm \
  -e JOB_COMPLETION_INDEX=0 \
  -e TOTAL_TASKS=4 \
  -v "$PWD/datos:/data" \
  bordes:1.0
```

Para el clúster, monta el volumen compartido en `/data`; los valores de índice
y número de tareas pueden llegar mediante `JOB_COMPLETION_INDEX` y
`TOTAL_TASKS`. La imagen procesa el bloque asignado y guarda la parte
correspondiente en `/data/salida/demo`.

## GitHub

El repositorio contiene el código y el reporte de tamaños, no las fotografías.
`datos/entrada/` y `datos/salida/` están excluidos de Git; al clonar el
proyecto, ejecuta `python download_dataset.py` para descargar la muestra.
