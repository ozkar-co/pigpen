#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruebas unitarias para el script de entrenamiento del modelo (train_model.py).
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Añadir el directorio raíz al path para importar los módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model import (
    set_seed,
    get_model,
    CombinedDataset
)


class TestSetSeed(unittest.TestCase):
    """Pruebas para la función de configuración de semilla."""
    
    def test_set_seed_makes_random_reproducible(self):
        """Prueba que set_seed hace que random sea reproducible."""
        import random
        
        # Configurar semilla y generar números aleatorios
        set_seed(42)
        random_numbers_1 = [random.random() for _ in range(10)]
        
        # Configurar la misma semilla y generar de nuevo
        set_seed(42)
        random_numbers_2 = [random.random() for _ in range(10)]
        
        # Deben ser iguales
        self.assertEqual(random_numbers_1, random_numbers_2, 
                        "Los números aleatorios deben ser reproducibles")
    
    def test_set_seed_makes_numpy_reproducible(self):
        """Prueba que set_seed hace que numpy sea reproducible."""
        # Configurar semilla y generar arrays aleatorios
        set_seed(42)
        array_1 = np.random.rand(10)
        
        # Configurar la misma semilla y generar de nuevo
        set_seed(42)
        array_2 = np.random.rand(10)
        
        # Deben ser iguales
        np.testing.assert_array_equal(array_1, array_2, 
                                      "Los arrays numpy deben ser reproducibles")
    
    def test_set_seed_makes_torch_reproducible(self):
        """Prueba que set_seed hace que torch sea reproducible."""
        # Configurar semilla y generar tensores aleatorios
        set_seed(42)
        tensor_1 = torch.rand(10)
        
        # Configurar la misma semilla y generar de nuevo
        set_seed(42)
        tensor_2 = torch.rand(10)
        
        # Deben ser iguales
        self.assertTrue(torch.equal(tensor_1, tensor_2), 
                       "Los tensores de torch deben ser reproducibles")
    
    def test_different_seeds_produce_different_results(self):
        """Prueba que diferentes semillas producen resultados diferentes."""
        # Configurar semilla 42
        set_seed(42)
        tensor_1 = torch.rand(10)
        
        # Configurar semilla 123
        set_seed(123)
        tensor_2 = torch.rand(10)
        
        # Deben ser diferentes
        self.assertFalse(torch.equal(tensor_1, tensor_2), 
                        "Diferentes semillas deben producir resultados diferentes")


class TestGetModel(unittest.TestCase):
    """Pruebas para la función de creación de modelos."""
    
    def test_get_model_resnet18(self):
        """Prueba la creación de un modelo ResNet18."""
        num_classes = 26  # A-Z
        model = get_model('resnet18', num_classes)
        
        # Verificar que es una instancia de nn.Module
        self.assertIsInstance(model, nn.Module, "Debe devolver un nn.Module")
        
        # Verificar que la última capa tiene el número correcto de clases
        self.assertEqual(model.fc.out_features, num_classes, 
                        "La última capa debe tener el número correcto de clases")
    
    def test_get_model_resnet34(self):
        """Prueba la creación de un modelo ResNet34."""
        num_classes = 10
        model = get_model('resnet34', num_classes)
        
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.fc.out_features, num_classes)
    
    def test_get_model_resnet50(self):
        """Prueba la creación de un modelo ResNet50."""
        num_classes = 15
        model = get_model('resnet50', num_classes)
        
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.fc.out_features, num_classes)
    
    def test_get_model_invalid_name_raises_error(self):
        """Prueba que un nombre de modelo inválido lanza un error."""
        with self.assertRaises(ValueError):
            get_model('invalid_model', 26)
    
    def test_get_model_forward_pass(self):
        """Prueba que el modelo puede hacer un forward pass."""
        num_classes = 26
        model = get_model('resnet18', num_classes)
        
        # Crear un tensor de entrada de prueba
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # Hacer forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Verificar la forma de la salida
        self.assertEqual(output.shape, (batch_size, num_classes), 
                        "La salida debe tener la forma correcta")
    
    def test_get_model_different_num_classes(self):
        """Prueba que se pueden crear modelos con diferentes números de clases."""
        class_counts = [5, 10, 26, 50, 100]
        
        for num_classes in class_counts:
            model = get_model('resnet18', num_classes)
            self.assertEqual(model.fc.out_features, num_classes, 
                           f"Debe soportar {num_classes} clases")


class TestCombinedDataset(unittest.TestCase):
    """Pruebas para la clase CombinedDataset."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        # Crear datasets de prueba simulados
        class MockDataset:
            def __init__(self, size, classes):
                self.size = size
                self.classes = classes
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Devolver un tensor de prueba y una etiqueta
                return torch.randn(3, 224, 224), idx % len(self.classes)
        
        # Crear clases de prueba
        self.classes = ['A', 'B', 'C', 'D', 'E']
        
        # Crear dos datasets de prueba
        self.dataset1 = MockDataset(100, self.classes)
        self.dataset2 = MockDataset(50, self.classes)
    
    def test_combined_dataset_creation(self):
        """Prueba la creación de un CombinedDataset."""
        combined = CombinedDataset([self.dataset1, self.dataset2])
        
        # Verificar que se creó correctamente
        self.assertIsNotNone(combined)
        self.assertEqual(len(combined.datasets), 2, 
                        "Debe tener 2 datasets")
    
    def test_combined_dataset_length(self):
        """Prueba que la longitud del CombinedDataset es correcta."""
        combined = CombinedDataset([self.dataset1, self.dataset2])
        
        # La longitud debe ser la suma de las longitudes de los datasets
        expected_length = len(self.dataset1) + len(self.dataset2)
        self.assertEqual(len(combined), expected_length, 
                        "La longitud debe ser la suma de los datasets")
    
    def test_combined_dataset_getitem(self):
        """Prueba que se puede obtener un item del CombinedDataset."""
        combined = CombinedDataset([self.dataset1, self.dataset2])
        
        # Obtener el primer item
        item = combined[0]
        
        # Verificar que es una tupla (imagen, etiqueta)
        self.assertIsInstance(item, tuple, "Debe devolver una tupla")
        self.assertEqual(len(item), 2, "La tupla debe tener 2 elementos")
    
    def test_combined_dataset_accesses_all_datasets(self):
        """Prueba que CombinedDataset accede a todos los datasets."""
        combined = CombinedDataset([self.dataset1, self.dataset2])
        
        # Obtener items del primer dataset
        item_from_first = combined[0]
        self.assertIsNotNone(item_from_first)
        
        # Obtener items del segundo dataset
        item_from_second = combined[len(self.dataset1)]
        self.assertIsNotNone(item_from_second)
    
    def test_combined_dataset_classes(self):
        """Prueba que CombinedDataset hereda las clases correctamente."""
        combined = CombinedDataset([self.dataset1, self.dataset2])
        
        # Verificar que tiene las clases del primer dataset
        self.assertEqual(combined.classes, self.classes, 
                        "Debe tener las mismas clases que los datasets")
        self.assertEqual(combined.class_to_idx, self.dataset1.class_to_idx, 
                        "Debe tener el mismo mapeo de clases")
    
    def test_combined_dataset_single_dataset(self):
        """Prueba CombinedDataset con un solo dataset."""
        combined = CombinedDataset([self.dataset1])
        
        # Debe funcionar con un solo dataset
        self.assertEqual(len(combined), len(self.dataset1))
    
    def test_combined_dataset_multiple_datasets(self):
        """Prueba CombinedDataset con múltiples datasets."""
        # Crear un tercer dataset
        class MockDataset:
            def __init__(self, size, classes):
                self.size = size
                self.classes = classes
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return torch.randn(3, 224, 224), idx % len(self.classes)
        
        dataset3 = MockDataset(75, self.classes)
        combined = CombinedDataset([self.dataset1, self.dataset2, dataset3])
        
        # Verificar la longitud total
        expected_length = len(self.dataset1) + len(self.dataset2) + len(dataset3)
        self.assertEqual(len(combined), expected_length)


class TestModelTrainingHelpers(unittest.TestCase):
    """Pruebas para funciones auxiliares de entrenamiento."""
    
    def test_model_can_be_moved_to_device(self):
        """Prueba que un modelo puede moverse a un dispositivo."""
        model = get_model('resnet18', 26)
        device = torch.device("cpu")
        
        # Mover el modelo al dispositivo
        model = model.to(device)
        
        # Verificar que está en el dispositivo correcto
        self.assertEqual(next(model.parameters()).device.type, 'cpu')
    
    def test_model_can_handle_batch(self):
        """Prueba que el modelo puede procesar un batch."""
        model = get_model('resnet18', 26)
        
        # Crear un batch de prueba
        batch_size = 4
        inputs = torch.randn(batch_size, 3, 224, 224)
        
        # Procesar el batch
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        
        # Verificar las dimensiones de salida
        self.assertEqual(outputs.shape[0], batch_size)
        self.assertEqual(outputs.shape[1], 26)
    
    def test_loss_function_works(self):
        """Prueba que la función de pérdida funciona correctamente."""
        criterion = nn.CrossEntropyLoss()
        
        # Crear predicciones y etiquetas de prueba
        batch_size = 4
        num_classes = 26
        predictions = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Calcular la pérdida
        loss = criterion(predictions, labels)
        
        # Verificar que la pérdida es un escalar
        self.assertEqual(loss.ndim, 0, "La pérdida debe ser un escalar")
        
        # Verificar que la pérdida es positiva
        self.assertGreater(loss.item(), 0, "La pérdida debe ser positiva")


if __name__ == '__main__':
    unittest.main()
