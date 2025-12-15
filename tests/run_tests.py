#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar todas las pruebas del proyecto PigPen.
"""

import sys
import unittest
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests(verbosity=2):
    """
    Ejecuta todas las pruebas del proyecto.
    
    Args:
        verbosity: Nivel de detalle en la salida (0=quiet, 1=normal, 2=verbose)
    
    Returns:
        True si todas las pruebas pasaron, False si alguna falló
    """
    # Descubrir y ejecutar todas las pruebas
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Ejecutar las pruebas
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Devolver True si todas las pruebas pasaron
    return result.wasSuccessful()


def main():
    """Función principal."""
    print("=" * 70)
    print("Ejecutando pruebas para el proyecto PigPen")
    print("=" * 70)
    print()
    
    # Ejecutar las pruebas
    success = run_tests(verbosity=2)
    
    # Imprimir resumen
    print()
    print("=" * 70)
    if success:
        print("✓ Todas las pruebas pasaron exitosamente")
        print("=" * 70)
        sys.exit(0)
    else:
        print("✗ Algunas pruebas fallaron")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
