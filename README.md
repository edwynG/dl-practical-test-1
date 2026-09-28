# Examen parcial 1: Parte práctica
**Redes neuronales profundas [UCV] · Semestre I-2026**

Este repositorio contiene la resolución del primer parcial práctico enfocado en el diseño, entrenamiento, comparación analítica y análisis del sesgo inductivo de redes neuronales profundas sobre el conjunto de datos **Fashion-MNIST**.



## Resumen del parcial

El parcial se estructura en 6 secciones principales desarrolladas en el notebook [`DL26_Parcial1_practico_GAP_v2.ipynb`](./DL26_Parcial1_practico_GAP_v2.ipynb):

1. **Los datos (Fashion-MNIST):**
   - Manejo de tensores de imágenes en escala de grises ($1 \times 28 \times 28$) divididos en 12.000 ejemplos de entrenamiento y 10.000 de prueba.
   - Comprensión del orden de dimensiones `(N, C, H, W)` en PyTorch y preparación de data loaders.

2. **Tres redes: FNN, CNN y CNN con GAP:**
   - **FNN:** red completamente conectada (MLP) con capas `Linear` y activaciones `ReLU`.
   - **CNN:** red convolucional clásica con bloques `Conv2d + ReLU + MaxPool2d` y cabeza densa `Linear(3136 → 128 → 10)`.
   - **CNN con GAP:** red convolucional que sustituye las capas densas intermedias por *global average pooling* (`AdaptiveAvgPool2d(1)`), reduciendo drásticamente el número de parámetros.
   - Cálculo manual y computacional del tamaño de salida por capa y conteo de parámetros entrenables.

3. **Entrenamiento y comparación:**
   - Entrenamiento controlado (Adam, 3 épocas, batch de 128) bajo la misma semilla aleatoria.
   - Comparación de curvas de pérdida, exactitud (*train* vs. *test*), tiempos de cómputo y sobreajuste.

4. **Sensibilidad al tamaño de entrada:**
   - Evaluación de las tres redes frente a imágenes reescaladas a $56 \times 56$.
   - Análisis de por qué la arquitectura tradicional falla por incompatibilidad dimensional en capas `Linear`, mientras que `CNN_GAP` se adapta gracias al pooling adaptativo.

5. **Permutación espacial de píxeles:**
   - Aplicación de una permutación fija e idéntica a todos los píxeles de las imágenes (destruyendo la coherencia y localidad espacial).
   - Análisis empírico de por qué la FNN mantiene su desempeño casi intacto mientras las CNNs sufren una degradación significativa al romperse sus supuestos de localidad y pesos compartidos.

6. **Cierre conceptual y sesgo inductivo:**
   - Discusión formal sobre el sesgo inductivo (*inductive bias*), invariancia y equivariancia a traslaciones, y criterios de diseño al trabajar con datos espaciales vs. datos tabulares/clínicos.


## Configuración del entorno de ejecución

El proyecto utiliza el entorno virtual local `.venv` ubicado en la raíz del proyecto (Python 3.12).

### 1. Activar el entorno virtual
```bash
source .venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Seleccionar el kernel / entorno en Jupyter o el editor
Al abrir el notebook, en el selector de kernel o entorno de Python:
- Seleccionar el entorno de la raíz: **`.venv`** (`.venv/bin/python`).



> [!NOTE]
> ### Aclaratoria sobre dependencias NVIDIA en requirements.txt
>
> Este proyecto fue configurado y ejecutado localmente en una máquina con GPU dedicada **NVIDIA GeForce RTX 3050 Laptop GPU**.
>
> Al generar el archivo [`requirements.txt`](./requirements.txt) mediante `pip freeze`, se listan paquetes con prefijo `nvidia-*` (tales como `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`, `nvidia-cublas`, `nvidia-curand`, etc.).
> 
> **¿Por qué están presentes?**
> Son las librerías binarias oficiales de NVIDIA distribuidas a través de PyPI. Estas proveen el entorno de ejecución de CUDA 13 y los aceleradores de cómputo cuDNN y cuBLAS requeridos para que **PyTorch** ejecute los tensores y gradientes directamente en la GPU NVIDIA de forma acelerada sin necesidad de instalar manualmente el CUDA Toolkit global en el sistema operativo.
