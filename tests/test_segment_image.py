#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruebas unitarias para el script de segmentación de imágenes (segment_image.py).
"""

import unittest
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar los módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.segment_image import (
    preprocess_image,
    segment_characters,
    save_characters
)


class TestPreprocessImage(unittest.TestCase):
    """Pruebas para la función de preprocesamiento de imágenes."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        # Crear una imagen de prueba simple
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Dibujar un rectángulo negro en el centro
        cv2.rectangle(self.test_image, (30, 30), (70, 70), (0, 0, 0), -1)
    
    def test_preprocess_color_image(self):
        """Prueba el preprocesamiento de una imagen en color."""
        result = preprocess_image(self.test_image)
        
        # Verificar que el resultado es una imagen binaria
        self.assertEqual(len(result.shape), 2, "La imagen preprocesada debe ser 2D")
        self.assertEqual(result.dtype, np.uint8, "El tipo debe ser uint8")
        
        # Verificar que solo contiene valores 0 o 255
        unique_values = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_values, [0, 255])), 
                       "La imagen debe ser binaria (0 o 255)")
    
    def test_preprocess_grayscale_image(self):
        """Prueba el preprocesamiento de una imagen en escala de grises."""
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        result = preprocess_image(gray_image)
        
        # Verificar que el resultado es una imagen binaria
        self.assertEqual(len(result.shape), 2, "La imagen preprocesada debe ser 2D")
        self.assertEqual(result.dtype, np.uint8, "El tipo debe ser uint8")
    
    def test_preprocess_empty_image(self):
        """Prueba el preprocesamiento de una imagen completamente blanca."""
        white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = preprocess_image(white_image)
        
        # Verificar que la imagen fue procesada
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (100, 100))


class TestSegmentCharacters(unittest.TestCase):
    """Pruebas para la función de segmentación de caracteres."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        # Crear una imagen binaria con varios objetos
        self.binary_image = np.zeros((200, 300), dtype=np.uint8)
        # Dibujar tres rectángulos blancos (simulando caracteres)
        cv2.rectangle(self.binary_image, (20, 20), (60, 80), 255, -1)
        cv2.rectangle(self.binary_image, (120, 20), (160, 80), 255, -1)
        cv2.rectangle(self.binary_image, (220, 20), (260, 80), 255, -1)
    
    def test_segment_multiple_characters(self):
        """Prueba la segmentación de múltiples caracteres."""
        characters = segment_characters(self.binary_image, min_size=100, padding=5)
        
        # Verificar que se detectaron caracteres
        self.assertGreater(len(characters), 0, "Debe detectar al menos un carácter")
        
        # Verificar la estructura de los caracteres
        for char in characters:
            self.assertIn('image', char, "Cada carácter debe tener una imagen")
            self.assertIn('bbox', char, "Cada carácter debe tener un bbox")
            self.assertEqual(len(char['bbox']), 4, "bbox debe tener 4 coordenadas")
    
    def test_segment_with_min_size_filter(self):
        """Prueba que el filtro de tamaño mínimo funciona correctamente."""
        # Añadir un objeto muy pequeño
        cv2.rectangle(self.binary_image, (100, 150), (105, 155), 255, -1)
        
        # Segmentar con tamaño mínimo grande
        characters = segment_characters(self.binary_image, min_size=500, padding=5)
        
        # El objeto pequeño no debe ser detectado
        # Los tres rectángulos grandes (~40x60 = 2400 píxeles) deben ser detectados
        self.assertGreater(len(characters), 0, "Debe detectar caracteres grandes")
    
    def test_segment_empty_image(self):
        """Prueba la segmentación de una imagen vacía."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        characters = segment_characters(empty_image, min_size=20, padding=5)
        
        # No debe detectar ningún carácter
        self.assertEqual(len(characters), 0, "No debe detectar caracteres en imagen vacía")
    
    def test_segment_with_padding(self):
        """Prueba que el padding se aplica correctamente."""
        characters = segment_characters(self.binary_image, min_size=100, padding=10)
        
        # Verificar que los caracteres tienen padding
        for char in characters:
            minr, minc, maxr, maxc = char['bbox']
            # El padding debe hacer que las dimensiones sean mayores
            self.assertGreater(maxr - minr, 0, "Alto debe ser positivo")
            self.assertGreater(maxc - minc, 0, "Ancho debe ser positivo")


class TestSaveCharacters(unittest.TestCase):
    """Pruebas para la función de guardado de caracteres."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        # Crear un directorio temporal para pruebas
        self.test_dir = '/tmp/test_pigpen_output'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Crear caracteres de prueba
        char_image = np.ones((50, 50), dtype=np.uint8) * 255
        self.test_characters = [
            {'image': char_image, 'bbox': (10, 10, 60, 60)},
            {'image': char_image, 'bbox': (70, 10, 120, 60)}
        ]
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        # Eliminar archivos de prueba
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_characters_creates_directory(self):
        """Prueba que se crea el directorio de salida."""
        output_dir = os.path.join(self.test_dir, 'new_dir')
        save_characters(self.test_characters, output_dir)
        
        # Verificar que el directorio fue creado
        self.assertTrue(os.path.exists(output_dir), 
                       "El directorio de salida debe ser creado")
    
    def test_save_characters_creates_files(self):
        """Prueba que se guardan archivos para cada carácter."""
        save_characters(self.test_characters, self.test_dir)
        
        # Verificar que se crearon archivos
        files = [f for f in os.listdir(self.test_dir) if f.endswith('.png')]
        self.assertEqual(len(files), len(self.test_characters), 
                        "Debe crear un archivo por carácter")
    
    def test_save_characters_file_format(self):
        """Prueba que los archivos guardados tienen el formato correcto."""
        save_characters(self.test_characters, self.test_dir)
        
        # Verificar el formato de nombre de archivo
        files = [f for f in os.listdir(self.test_dir) if f.endswith('.png')]
        for filename in files:
            # Debe tener el formato char_timestamp_###.png
            self.assertTrue(filename.startswith('char_'), 
                           "El nombre debe comenzar con 'char_'")
            self.assertTrue(filename.endswith('.png'), 
                           "El archivo debe ser .png")
    
    def test_save_empty_characters_list(self):
        """Prueba el guardado de una lista vacía de caracteres."""
        save_characters([], self.test_dir)
        
        # El directorio debe existir pero sin archivos
        self.assertTrue(os.path.exists(self.test_dir))
        files = [f for f in os.listdir(self.test_dir) if f.endswith('.png')]
        self.assertEqual(len(files), 0, "No debe crear archivos")


if __name__ == '__main__':
    unittest.main()
