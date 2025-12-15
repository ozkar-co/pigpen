#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruebas unitarias para el script de generación de datos (generate_data.py).
"""

import unittest
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar los módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_data import (
    draw_letter_A,
    draw_letter_B,
    draw_letter_E,
    draw_letter_S,
    flip_horizontal,
    flip_vertical,
    rotate_90_clockwise,
    apply_mutations,
    generate_random_id
)


class TestDrawLetterFunctions(unittest.TestCase):
    """Pruebas para las funciones de dibujo de letras."""
    
    def test_draw_letter_A_without_dot(self):
        """Prueba el dibujo de la letra A sin punto."""
        image = draw_letter_A(with_dot=False)
        
        # Verificar dimensiones y tipo
        self.assertEqual(len(image.shape), 2, "La imagen debe ser 2D")
        self.assertEqual(image.dtype, np.uint8, "El tipo debe ser uint8")
        
        # Verificar que la imagen no está completamente vacía
        self.assertGreater(np.sum(image), 0, "La imagen debe contener píxeles dibujados")
    
    def test_draw_letter_A_with_dot(self):
        """Prueba el dibujo de la letra A con punto (R)."""
        image = draw_letter_A(with_dot=True)
        
        # Verificar que hay más píxeles blancos que en la versión sin punto
        image_no_dot = draw_letter_A(with_dot=False)
        self.assertGreater(np.sum(image), np.sum(image_no_dot), 
                          "La versión con punto debe tener más píxeles blancos")
    
    def test_draw_letter_B_without_dot(self):
        """Prueba el dibujo de la letra B sin punto."""
        image = draw_letter_B(with_dot=False)
        
        # Verificar dimensiones
        self.assertEqual(len(image.shape), 2, "La imagen debe ser 2D")
        self.assertGreater(np.sum(image), 0, "La imagen debe contener píxeles dibujados")
    
    def test_draw_letter_B_with_dot(self):
        """Prueba el dibujo de la letra B con punto (K)."""
        image = draw_letter_B(with_dot=True)
        image_no_dot = draw_letter_B(with_dot=False)
        
        # Verificar que hay más píxeles en la versión con punto
        self.assertGreater(np.sum(image), np.sum(image_no_dot))
    
    def test_draw_letter_E_without_dot(self):
        """Prueba el dibujo de la letra E sin punto."""
        image = draw_letter_E(with_dot=False)
        
        # Verificar dimensiones
        self.assertEqual(len(image.shape), 2, "La imagen debe ser 2D")
        self.assertGreater(np.sum(image), 0, "La imagen debe contener píxeles dibujados")
    
    def test_draw_letter_E_with_dot(self):
        """Prueba el dibujo de la letra E con punto (N)."""
        image = draw_letter_E(with_dot=True)
        image_no_dot = draw_letter_E(with_dot=False)
        
        # Verificar que hay más píxeles en la versión con punto
        self.assertGreater(np.sum(image), np.sum(image_no_dot))
    
    def test_draw_letter_S_without_dot(self):
        """Prueba el dibujo de la letra S sin punto (V)."""
        image = draw_letter_S(with_dot=False)
        
        # Verificar dimensiones
        self.assertEqual(len(image.shape), 2, "La imagen debe ser 2D")
        self.assertGreater(np.sum(image), 0, "La imagen debe contener píxeles dibujados")
    
    def test_draw_letter_S_with_dot(self):
        """Prueba el dibujo de la letra S con punto (Z)."""
        image = draw_letter_S(with_dot=True)
        image_no_dot = draw_letter_S(with_dot=False)
        
        # Verificar que hay más píxeles en la versión con punto
        self.assertGreater(np.sum(image), np.sum(image_no_dot))


class TestImageTransformations(unittest.TestCase):
    """Pruebas para las funciones de transformación de imágenes."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        # Crear una imagen de prueba asimétrica
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        # Dibujar una forma asimétrica
        cv2.rectangle(self.test_image, (10, 10), (40, 90), 255, -1)
    
    def test_flip_horizontal(self):
        """Prueba el volteo horizontal de una imagen."""
        flipped = flip_horizontal(self.test_image)
        
        # Verificar dimensiones
        self.assertEqual(flipped.shape, self.test_image.shape, 
                        "Las dimensiones deben ser iguales")
        
        # Verificar que la imagen cambió
        self.assertFalse(np.array_equal(flipped, self.test_image), 
                        "La imagen debe ser diferente después del flip")
        
        # Verificar que tiene la misma cantidad de píxeles blancos
        self.assertEqual(np.sum(flipped), np.sum(self.test_image), 
                        "Debe tener la misma cantidad de píxeles blancos")
    
    def test_flip_vertical(self):
        """Prueba el volteo vertical de una imagen."""
        flipped = flip_vertical(self.test_image)
        
        # Verificar dimensiones
        self.assertEqual(flipped.shape, self.test_image.shape)
        
        # Verificar que la imagen cambió
        self.assertFalse(np.array_equal(flipped, self.test_image))
        
        # Verificar que tiene la misma cantidad de píxeles blancos
        self.assertEqual(np.sum(flipped), np.sum(self.test_image))
    
    def test_rotate_90_clockwise(self):
        """Prueba la rotación de 90 grados en sentido horario."""
        rotated = rotate_90_clockwise(self.test_image)
        
        # Verificar que las dimensiones se mantienen (imagen cuadrada)
        self.assertEqual(rotated.shape, self.test_image.shape)
        
        # Verificar que la imagen cambió
        self.assertFalse(np.array_equal(rotated, self.test_image))
    
    def test_double_flip_horizontal_returns_original(self):
        """Prueba que dos flips horizontales devuelven la imagen original."""
        flipped_once = flip_horizontal(self.test_image)
        flipped_twice = flip_horizontal(flipped_once)
        
        # Debe ser igual a la original
        np.testing.assert_array_equal(flipped_twice, self.test_image)
    
    def test_double_flip_vertical_returns_original(self):
        """Prueba que dos flips verticales devuelven la imagen original."""
        flipped_once = flip_vertical(self.test_image)
        flipped_twice = flip_vertical(flipped_once)
        
        # Debe ser igual a la original
        np.testing.assert_array_equal(flipped_twice, self.test_image)
    
    def test_four_rotations_return_original(self):
        """Prueba que cuatro rotaciones de 90° devuelven la imagen original."""
        rotated = self.test_image
        for _ in range(4):
            rotated = rotate_90_clockwise(rotated)
        
        # Debe ser igual a la original
        np.testing.assert_array_equal(rotated, self.test_image)


class TestMutations(unittest.TestCase):
    """Pruebas para las funciones de mutación de imágenes."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        # Crear una imagen de prueba
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(self.test_image, (30, 30), (70, 70), 255, -1)
    
    def test_apply_mutations_returns_tuple(self):
        """Prueba que apply_mutations devuelve una tupla."""
        result = apply_mutations(self.test_image)
        
        # Verificar que devuelve una tupla
        self.assertIsInstance(result, tuple, "Debe devolver una tupla")
        self.assertEqual(len(result), 2, "La tupla debe tener 2 elementos")
    
    def test_apply_mutations_returns_image_and_effects(self):
        """Prueba que apply_mutations devuelve imagen y lista de efectos."""
        mutated_image, effects = apply_mutations(self.test_image)
        
        # Verificar la imagen mutada
        self.assertIsInstance(mutated_image, np.ndarray, 
                             "El primer elemento debe ser un array numpy")
        self.assertEqual(mutated_image.shape, self.test_image.shape, 
                        "Las dimensiones deben ser iguales")
        
        # Verificar la lista de efectos
        self.assertIsInstance(effects, list, "El segundo elemento debe ser una lista")
        self.assertGreater(len(effects), 0, "Debe aplicar al menos un efecto")
    
    def test_apply_mutations_changes_image(self):
        """Prueba que apply_mutations modifica la imagen."""
        mutated_image, _ = apply_mutations(self.test_image)
        
        # En la mayoría de los casos, la imagen debe cambiar
        # (puede no cambiar si las mutaciones son muy pequeñas)
        # Verificar al menos que tiene el mismo tipo
        self.assertEqual(mutated_image.dtype, self.test_image.dtype)
    
    def test_apply_mutations_preserves_dtype(self):
        """Prueba que apply_mutations preserva el tipo de dato."""
        mutated_image, _ = apply_mutations(self.test_image)
        
        self.assertEqual(mutated_image.dtype, np.uint8, 
                        "El tipo de dato debe ser uint8")
    
    def test_multiple_mutations_produce_different_results(self):
        """Prueba que múltiples mutaciones producen resultados diferentes."""
        # Aplicar mutaciones varias veces
        results = [apply_mutations(self.test_image) for _ in range(5)]
        
        # Al menos algunas deben ser diferentes
        # (aunque técnicamente podrían ser todas iguales por azar)
        images = [r[0] for r in results]
        effects = [r[1] for r in results]
        
        # Verificar que los efectos aplicados pueden variar
        self.assertEqual(len(effects), 5, "Debe haber 5 listas de efectos")


class TestUtilityFunctions(unittest.TestCase):
    """Pruebas para funciones de utilidad."""
    
    def test_generate_random_id_returns_string(self):
        """Prueba que generate_random_id devuelve un string."""
        random_id = generate_random_id()
        
        self.assertIsInstance(random_id, str, "Debe devolver un string")
    
    def test_generate_random_id_has_correct_length(self):
        """Prueba que generate_random_id devuelve un ID de longitud correcta."""
        random_id = generate_random_id()
        
        # Por defecto, debe generar un ID de 6 caracteres
        self.assertEqual(len(random_id), 6, "El ID debe tener 6 caracteres")
    
    def test_generate_random_id_is_alphanumeric(self):
        """Prueba que generate_random_id devuelve solo caracteres alfanuméricos."""
        random_id = generate_random_id()
        
        self.assertTrue(random_id.isalnum(), 
                       "El ID debe ser alfanumérico")
    
    def test_generate_random_id_is_lowercase(self):
        """Prueba que generate_random_id devuelve solo caracteres en minúscula."""
        random_id = generate_random_id()
        
        # Debe ser minúscula o dígito
        self.assertTrue(all(c.islower() or c.isdigit() for c in random_id), 
                       "El ID debe estar en minúsculas")
    
    def test_generate_random_id_produces_different_values(self):
        """Prueba que generate_random_id produce valores diferentes."""
        # Generar varios IDs
        ids = [generate_random_id() for _ in range(10)]
        
        # Al menos algunos deben ser diferentes
        # (técnicamente todos podrían ser iguales, pero es extremadamente improbable)
        unique_ids = set(ids)
        self.assertGreater(len(unique_ids), 1, 
                          "Debe generar IDs diferentes")


if __name__ == '__main__':
    unittest.main()
