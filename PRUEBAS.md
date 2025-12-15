# Carpeta de Archivos Fuente de Prueba - Proyecto PigPen

## Resumen

Se ha creado una carpeta completa con archivos fuente de prueba para validar el sistema de reconocimiento de cifrado Pigpen. Este directorio (`data/prueba/`) contiene imágenes con texto conocido que permite verificar la precisión del sistema.

## Contenido Creado

### 1. Imágenes de Prueba Generadas (6 imágenes)

Todas generadas concatenando caracteres reales de `data/classified`:

- **pangrama_espanol.png** - "EL VELOZ MURCIELAGO HINDU COMIA FELIZ CARDILLO Y KIWI"
- **pangrama_ingles.png** - "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
- **hello_world.png** - "HELLO WORLD"
- **alfabeto.png** - "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
- **mensaje_multilinea.png** - Mensaje en 3 líneas
- **frase_corta.png** - "DECODE THIS MESSAGE"

### 2. Documentación

- **README.md** - Documentación completa de todas las imágenes de prueba
- **textos_esperados.txt** - Lista de textos esperados en formato fácil de procesar

### 3. Scripts

- **scripts/create_test_images.py** - Genera las imágenes de prueba concatenando caracteres
- **scripts/validate_results.py** - Script de ejemplo para validar resultados del descifrado

### 4. Pruebas Unitarias

Se creó un directorio `tests/` con pruebas unitarias para:

- **test_segment_image.py** - Pruebas para segmentación de imágenes
- **test_generate_data.py** - Pruebas para generación de datos
- **test_train_model.py** - Pruebas para funciones de entrenamiento del modelo

## Cómo Usar

### Generar las Imágenes de Prueba

```bash
python scripts/create_test_images.py
```

### Probar Segmentación de Imágenes

```bash
python scripts/segment_image.py --input data/prueba/hello_world.png --debug
```

### Validar Resultados del Descifrado

```bash
# 1. Descifrar una imagen (requiere modelo entrenado)
python scripts/decrypt_image.py \
  --model models/tu_modelo.pth \
  --input data/prueba/pangrama_ingles.png

# 2. Validar el resultado usando el script de validación
python scripts/validate_results.py
```

### Ejecutar Pruebas Unitarias

```bash
# Todas las pruebas
python -m unittest discover tests -v

# O usar el script de pruebas
python tests/run_tests.py
```

## Características

### Textos Conocidos

Todas las imágenes generadas tienen texto conocido documentado, lo que permite:

- **Validación automática** - Comparar texto descifrado vs texto esperado
- **Métricas de precisión** - Calcular precisión de caracteres, palabras, etc.
- **Detección de errores** - Identificar qué caracteres se descifran incorrectamente

### Variedad de Pruebas

- **Pangramas** - Contienen todas las letras del alfabeto
- **Alfabeto completo** - Prueba cada letra A-Z
- **Mensajes cortos** - Para pruebas rápidas
- **Multilínea** - Prueba capacidad de procesar varias líneas

### Métricas de Validación

El script de validación incluye:

- **Precisión de caracteres** - Porcentaje de caracteres correctos
- **Distancia de Levenshtein** - Número de ediciones necesarias para corregir
- **Comparación de longitudes** - Detecta caracteres faltantes o extra
- **Reporte detallado** - Muestra exactamente dónde están los errores

## Estructura de Archivos

```
pigpen/
├── data/
│   └── prueba/                          # Carpeta de pruebas
│       ├── README.md                    # Documentación de las imágenes
│       ├── textos_esperados.txt         # Textos esperados
│       ├── pangrama_espanol.png         # Pangrama español
│       ├── pangrama_ingles.png          # Pangrama inglés
│       ├── hello_world.png              # Texto simple
│       ├── alfabeto.png                 # A-Z completo
│       ├── mensaje_multilinea.png       # Mensaje en 3 líneas
│       └── frase_corta.png              # Frase corta
├── scripts/
│   ├── create_test_images.py            # Genera imágenes de prueba
│   └── validate_results.py              # Valida resultados
└── tests/                               # Pruebas unitarias
    ├── __init__.py
    ├── README.md
    ├── run_tests.py
    ├── test_segment_image.py
    ├── test_generate_data.py
    └── test_train_model.py
```

## Próximos Pasos

Para completar el sistema de pruebas:

1. **Entrenar un modelo** usando `scripts/train_model.py`
2. **Descifrar las imágenes de prueba** usando el modelo entrenado
3. **Validar los resultados** comparando con los textos esperados
4. **Iterar y mejorar** el modelo basándose en los errores detectados
5. **Agregar más imágenes de prueba** según sea necesario

## Notas Técnicas

- Las imágenes se generan concatenando caracteres seleccionados aleatoriamente de `data/classified`
- Cada ejecución del script puede producir imágenes ligeramente diferentes (por la selección aleatoria)
- Los textos solo incluyen letras A-Z (sin números ni caracteres especiales)
- Las imágenes externas de URLs no se pudieron descargar automáticamente por restricciones de red, pero pueden agregarse manualmente

## Beneficios

✓ **Validación objetiva** - Texto conocido permite medir precisión exacta
✓ **Desarrollo iterativo** - Fácil identificar y corregir errores
✓ **Documentado** - Toda la información está claramente documentada
✓ **Reproducible** - Scripts permiten regenerar las pruebas fácilmente
✓ **Extensible** - Fácil agregar nuevas imágenes de prueba
✓ **Completo** - Cubre casos simples y complejos (multilínea, pangramas, etc.)
