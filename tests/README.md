# Tests - PigPen

Este directorio contiene las pruebas unitarias para el proyecto PigPen.

## Estructura

```
tests/
├── __init__.py                # Inicialización del paquete de pruebas
├── test_segment_image.py      # Pruebas para la segmentación de imágenes
├── test_generate_data.py      # Pruebas para la generación de datos
├── test_train_model.py        # Pruebas para el entrenamiento del modelo
└── README.md                  # Este archivo
```

## Ejecutar las Pruebas

### Ejecutar todas las pruebas

```bash
# Desde el directorio raíz del proyecto
python -m unittest discover tests

# O usando pytest (si está instalado)
pytest tests/
```

### Ejecutar un archivo de pruebas específico

```bash
# Pruebas de segmentación
python -m unittest tests.test_segment_image

# Pruebas de generación de datos
python -m unittest tests.test_generate_data

# Pruebas de entrenamiento del modelo
python -m unittest tests.test_train_model
```

### Ejecutar una clase de prueba específica

```bash
python -m unittest tests.test_segment_image.TestPreprocessImage
```

### Ejecutar una prueba específica

```bash
python -m unittest tests.test_segment_image.TestPreprocessImage.test_preprocess_color_image
```

### Ejecutar con verbosidad

```bash
python -m unittest discover tests -v
```

## Cobertura de Pruebas

Las pruebas cubren las siguientes funcionalidades:

### test_segment_image.py
- ✓ Preprocesamiento de imágenes (color y escala de grises)
- ✓ Segmentación de caracteres múltiples
- ✓ Filtrado por tamaño mínimo
- ✓ Aplicación de padding
- ✓ Guardado de caracteres segmentados
- ✓ Creación de directorios de salida
- ✓ Formato de nombres de archivo

### test_generate_data.py
- ✓ Dibujo de letras base (A, B, E, S/V) con y sin punto
- ✓ Transformaciones de imágenes (flip horizontal, flip vertical, rotación 90°)
- ✓ Aplicación de mutaciones aleatorias
- ✓ Generación de IDs aleatorios
- ✓ Preservación de propiedades de imagen

### test_train_model.py
- ✓ Configuración de semillas para reproducibilidad
- ✓ Creación de modelos (ResNet18, ResNet34, ResNet50)
- ✓ Forward pass de modelos
- ✓ CombinedDataset para combinar múltiples datasets
- ✓ Funciones de pérdida
- ✓ Movimiento de modelos a diferentes dispositivos

## Requisitos

Las pruebas requieren las mismas dependencias que el proyecto principal:

```bash
pip install -r requirements.txt
```

## Agregar Nuevas Pruebas

Para agregar nuevas pruebas:

1. Crea un nuevo archivo `test_<nombre>.py` en este directorio
2. Importa `unittest` y los módulos a probar
3. Crea clases que hereden de `unittest.TestCase`
4. Escribe métodos de prueba que comiencen con `test_`
5. Usa métodos `setUp()` y `tearDown()` si necesitas configuración/limpieza

Ejemplo:

```python
import unittest

class TestMiModulo(unittest.TestCase):
    def setUp(self):
        """Configuración antes de cada prueba."""
        pass
    
    def test_mi_funcion(self):
        """Descripción de lo que prueba este test."""
        resultado = mi_funcion()
        self.assertEqual(resultado, valor_esperado)
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        pass

if __name__ == '__main__':
    unittest.main()
```

## Mejores Prácticas

- Cada prueba debe ser independiente y no depender de otras pruebas
- Usa nombres descriptivos para las pruebas (`test_funcion_cuando_condicion_entonces_resultado`)
- Limpia los recursos temporales en `tearDown()`
- Usa `setUp()` para crear fixtures comunes
- Escribe docstrings para explicar qué prueba cada test
- Mantén las pruebas simples y enfocadas en una sola funcionalidad

## Continuous Integration

Estas pruebas se ejecutan automáticamente en CI/CD cuando se hace push al repositorio.
