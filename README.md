# PigPen - Reconocimiento de Cifrado Masónico con IA

Este proyecto utiliza inteligencia artificial para identificar y descifrar caracteres del cifrado Pigpen (también conocido como cifrado Tic-tac-toe o cifrado Francmasón).

## Descripción

El cifrado Pigpen es un sistema de sustitución geométrica simple que reemplaza letras por símbolos basados en fragmentos de una cuadrícula. Este proyecto implementa un sistema de reconocimiento óptico de caracteres (OCR) especializado para este cifrado histórico utilizando técnicas de aprendizaje profundo.

## Estructura del Proyecto

```
pigpen/
├── data/
│   ├── classified/          # Imágenes clasificadas por letra (A-Z)
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   ├── generated/           # Imágenes generadas sintéticamente
│   ├── prueba/              # Imágenes de prueba con texto conocido
│   └── unclassified/        # Imágenes pendientes de clasificación
├── models/                  # Modelos entrenados
├── scripts/
│   ├── segment_image.py     # Divide imágenes en caracteres individuales
│   ├── train_model.py       # Entrena el modelo con imágenes clasificadas
│   ├── decrypt_image.py     # Identifica y descifra texto en imágenes
│   ├── generate_data.py     # Genera imágenes para entrenamiento/evaluación
│   └── create_test_images.py # Crea imágenes de prueba con texto conocido
├── tests/                   # Pruebas unitarias del proyecto
├── utils/                   # Funciones auxiliares
└── requirements.txt         # Dependencias del proyecto
```

## Funcionalidades

### 1. Segmentación de Imágenes (`segment_image.py`)

Este script permite procesar una imagen que contiene múltiples caracteres del cifrado Pigpen y dividirla en imágenes individuales para cada carácter. Las imágenes resultantes se guardan en la carpeta `data/unclassified/` para su posterior clasificación manual.

```bash
python scripts/segment_image.py --input path/to/image.jpg
```

### 2. Entrenamiento del Modelo (`train_model.py`)

Entrena un modelo de aprendizaje profundo utilizando las imágenes clasificadas. El script aplica técnicas de aumento de datos (data augmentation) para expandir el conjunto de entrenamiento mediante deformaciones ligeras de las imágenes existentes.

```bash
python scripts/train_model.py --epochs 50 --batch-size 32
```

### 3. Descifrado de Imágenes (`decrypt_image.py`)

Utiliza el modelo entrenado para identificar caracteres Pigpen en una imagen y mostrar el texto descifrado correspondiente.

```bash
python scripts/decrypt_image.py --model path/to/model --input path/to/encrypted_image.jpg
```

### 4. Generación de Datos (`generate_data.py`)

Crea imágenes sintéticas de caracteres Pigpen para ampliar el conjunto de entrenamiento o para evaluación manual.

```bash
python scripts/generate_data.py --count 100
```

### 5. Creación de Imágenes de Prueba (`create_test_images.py`)

Genera imágenes de prueba con texto conocido concatenando caracteres de `data/classified`. Útil para validar el sistema de descifrado.

```bash
python scripts/create_test_images.py
```

Esto crea imágenes como:
- Pangramas en español e inglés
- Alfabeto completo (A-Z)
- Mensajes en una y múltiples líneas
- Frases cortas para pruebas rápidas

Las imágenes se guardan en `data/prueba/` junto con documentación del texto esperado.

## Requisitos

- Python 3.8+
- TensorFlow o PyTorch
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Flujo de Trabajo Recomendado

1. Recolecta imágenes que contengan caracteres del cifrado Pigpen
2. Usa `segment_image.py` para dividir las imágenes en caracteres individuales
3. Clasifica manualmente las imágenes en las carpetas correspondientes (A-Z)
4. Entrena el modelo con `train_model.py`
5. Evalúa y mejora el modelo según sea necesario
6. Utiliza `decrypt_image.py` para descifrar nuevas imágenes

## Pruebas y Validación

### Pruebas Unitarias

El proyecto incluye pruebas unitarias para validar las funcionalidades principales:

```bash
# Ejecutar todas las pruebas
python -m unittest discover tests -v

# O usar el script de pruebas
python tests/run_tests.py
```

### Imágenes de Prueba

El directorio `data/prueba/` contiene imágenes de prueba con texto conocido para validar el sistema:

- **Pangramas:** Textos que contienen todas las letras del alfabeto
- **Alfabeto completo:** A-Z en orden
- **Mensajes multilínea:** Texto en varias líneas
- **Frases cortas:** Para pruebas rápidas

Para generar las imágenes de prueba:

```bash
python scripts/create_test_images.py
```

Para validar el sistema con estas imágenes:

```bash
# Probar segmentación
python scripts/segment_image.py --input data/prueba/hello_world.png --debug

# Probar descifrado (requiere modelo entrenado)
python scripts/decrypt_image.py --model models/tu_modelo.pth --input data/prueba/pangrama_ingles.png
```

Consulta `data/prueba/README.md` para ver los textos esperados de cada imagen.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, siente libre de abrir un issue o enviar un pull request con mejoras.

## Licencia

[Especificar la licencia del proyecto] 