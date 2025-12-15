# Imágenes de Prueba - Cifrado Pigpen

Este directorio contiene imágenes de prueba con texto conocido en cifrado Pigpen para validar el funcionamiento del sistema de reconocimiento.

## Propósito

Estas imágenes permiten:
- Probar el sistema de segmentación de caracteres
- Validar el modelo de reconocimiento entrenado
- Verificar la precisión del descifrado
- Comparar el texto descifrado con el texto original conocido

## Imágenes Generadas

### 1. pangrama_espanol.png
**Texto esperado:** `EL VELOZ MURCIELAGO HINDU COMIA FELIZ CARDILLO Y KIWI`

Pangrama en español (contiene todas las letras del alfabeto español) sin acentos ni diéresis para compatibilidad con A-Z.

### 2. pangrama_ingles.png
**Texto esperado:** `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG`

Pangrama clásico en inglés que contiene todas las letras del alfabeto inglés.

### 3. hello_world.png
**Texto esperado:** `HELLO WORLD`

Frase simple y corta para pruebas básicas.

### 4. alfabeto.png
**Texto esperado:** `ABCDEFGHIJKLMNOPQRSTUVWXYZ`

Alfabeto completo de A a Z en orden. Útil para verificar que todas las letras se reconocen correctamente.

### 5. mensaje_multilinea.png
**Texto esperado:**
```
PIGPEN CIPHER
ALSO KNOWN AS
MASONIC CIPHER
```

Mensaje en múltiples líneas para probar la capacidad del sistema de procesar texto en varias líneas.

### 6. frase_corta.png
**Texto esperado:** `DECODE THIS MESSAGE`

Frase corta para pruebas rápidas de descifrado.

## Cómo Usar Estas Imágenes

### Generar las Imágenes de Prueba

Para generar las imágenes concatenando caracteres de `data/classified`:

```bash
python scripts/create_test_images.py
```

Opciones disponibles:
```bash
python scripts/create_test_images.py \
  --classified-dir data/classified \
  --output-dir data/prueba
```

### Probar la Segmentación

```bash
python scripts/segment_image.py --input data/prueba/hello_world.png --debug
```

### Probar el Descifrado

```bash
python scripts/decrypt_image.py \
  --model models/pigpen_model_resnet18_TIMESTAMP.pth \
  --input data/prueba/pangrama_ingles.png
```

## Validación de Resultados

Para cada imagen de prueba, compara el texto descifrado con el texto esperado documentado arriba. Calcula métricas como:

- **Precisión de caracteres:** Porcentaje de caracteres correctamente identificados
- **Precisión de palabras:** Porcentaje de palabras completamente correctas
- **Distancia de Levenshtein:** Número de ediciones necesarias para corregir el texto

## Agregar Nuevas Imágenes de Prueba

Para agregar nuevas imágenes de prueba:

1. Edita el script `scripts/create_test_images.py`
2. Agrega la nueva imagen en la función `main()`
3. Documenta el texto esperado en este README
4. Ejecuta el script para regenerar las imágenes

Ejemplo de código para agregar una nueva imagen:

```python
create_text_image(
    "TU TEXTO AQUI",
    args.classified_dir,
    os.path.join(args.output_dir, 'nombre_archivo.png'),
    target_height=80,
    spacing=8
)
```

## Notas Importantes

- Las imágenes generadas concatenan caracteres reales de `data/classified`, por lo que su apariencia puede variar según las imágenes disponibles
- Las imágenes descargadas son de fuentes externas y pueden contener errores o variaciones en el cifrado
- Todas las imágenes generadas usan solo letras A-Z (sin números ni símbolos especiales)
- Los espacios entre palabras se representan con espacios en blanco en las imágenes

## Estructura del Directorio

```
data/prueba/
├── README.md                    # Este archivo
├── pangrama_espanol.png         # Pangrama español
├── pangrama_ingles.png          # Pangrama inglés
├── hello_world.png              # "HELLO WORLD"
├── alfabeto.png                 # A-Z completo
├── mensaje_multilinea.png       # Mensaje en 3 líneas
└── frase_corta.png              # "DECODE THIS MESSAGE"
```

## Requisitos

Para generar las imágenes de prueba, necesitas:
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

Instalar dependencias:
```bash
pip install opencv-python numpy
```
