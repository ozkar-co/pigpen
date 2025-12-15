#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de ejemplo para validar el sistema de descifrado usando las imágenes de prueba.
"""

import os
import cv2
from pathlib import Path


def load_expected_texts(filepath='data/prueba/textos_esperados.txt'):
    """
    Carga los textos esperados desde el archivo.
    
    Returns:
        Dict con nombre de archivo como clave y lista de textos esperados como valor
    """
    expected = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line and not line.startswith('#') and line:
                # Dividir solo en el primer |
                parts = line.split('|', 1)
                if len(parts) == 2:
                    filename, text = parts
                    if filename not in expected:
                        expected[filename] = []
                    expected[filename].append(text)
    return expected


def character_accuracy(expected, actual):
    """
    Calcula la precisión a nivel de caracteres.
    
    Args:
        expected: Texto esperado
        actual: Texto obtenido
        
    Returns:
        Porcentaje de precisión (0-100)
    """
    if len(expected) == 0:
        return 0.0
    
    # Normalizar longitudes
    max_len = max(len(expected), len(actual))
    expected_padded = expected.ljust(max_len)
    actual_padded = actual.ljust(max_len)
    
    # Contar caracteres correctos
    correct = sum(1 for e, a in zip(expected_padded, actual_padded) if e == a)
    
    return (correct / max_len) * 100


def levenshtein_distance(s1, s2):
    """
    Calcula la distancia de Levenshtein entre dos strings.
    
    Args:
        s1: Primer string
        s2: Segundo string
        
    Returns:
        Distancia de Levenshtein (número de ediciones necesarias)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Costo de inserción, eliminación o sustitución
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def validate_result(filename, decrypted_text, expected_texts):
    """
    Valida el resultado del descifrado.
    
    Args:
        filename: Nombre del archivo
        decrypted_text: Texto descifrado
        expected_texts: Diccionario con textos esperados
        
    Returns:
        Tupla (success, message, metrics)
    """
    if filename not in expected_texts:
        return None, "No hay texto esperado para esta imagen", {}
    
    expected = expected_texts[filename]
    
    # Para imágenes de una línea
    if len(expected) == 1:
        expected_text = expected[0]
        
        # Calcular métricas
        char_acc = character_accuracy(expected_text, decrypted_text)
        lev_dist = levenshtein_distance(expected_text, decrypted_text)
        
        metrics = {
            'character_accuracy': char_acc,
            'levenshtein_distance': lev_dist,
            'expected_length': len(expected_text),
            'actual_length': len(decrypted_text)
        }
        
        if decrypted_text == expected_text:
            return True, "✓ Texto descifrado correctamente", metrics
        else:
            return False, f"✗ Esperado: '{expected_text}'\n   Obtenido: '{decrypted_text}'", metrics
    
    # Para imágenes multilínea
    else:
        decrypted_lines = decrypted_text.split('\n')
        
        # Calcular precisión total
        total_expected = '\n'.join(expected)
        char_acc = character_accuracy(total_expected, decrypted_text)
        lev_dist = levenshtein_distance(total_expected, decrypted_text)
        
        metrics = {
            'character_accuracy': char_acc,
            'levenshtein_distance': lev_dist,
            'expected_lines': len(expected),
            'actual_lines': len(decrypted_lines),
            'expected_length': len(total_expected),
            'actual_length': len(decrypted_text)
        }
        
        if decrypted_lines == expected:
            return True, "✓ Todas las líneas descifradas correctamente", metrics
        else:
            message = f"✗ Esperado {len(expected)} líneas, obtenido {len(decrypted_lines)} líneas\n"
            for i, (exp_line, act_line) in enumerate(zip(expected, decrypted_lines)):
                if exp_line != act_line:
                    message += f"   Línea {i+1} - Esperado: '{exp_line}', Obtenido: '{act_line}'\n"
            return False, message, metrics


def print_validation_report(results):
    """
    Imprime un reporte de validación.
    
    Args:
        results: Lista de tuplas (filename, success, message, metrics)
    """
    print("=" * 80)
    print("REPORTE DE VALIDACIÓN - SISTEMA DE DESCIFRADO PIGPEN")
    print("=" * 80)
    print()
    
    total = len(results)
    passed = sum(1 for r in results if r[1])
    
    print(f"Total de imágenes: {total}")
    print(f"Descifradas correctamente: {passed}")
    print(f"Con errores: {total - passed}")
    print(f"Tasa de éxito: {(passed/total*100):.2f}%")
    print()
    print("=" * 80)
    print("DETALLES POR IMAGEN")
    print("=" * 80)
    print()
    
    for filename, success, message, metrics in results:
        print(f"Archivo: {filename}")
        print(f"Estado: {'✓ ÉXITO' if success else '✗ FALLO'}")
        print(message)
        
        if metrics:
            print(f"Métricas:")
            if 'character_accuracy' in metrics:
                print(f"  - Precisión de caracteres: {metrics['character_accuracy']:.2f}%")
            if 'levenshtein_distance' in metrics:
                print(f"  - Distancia de Levenshtein: {metrics['levenshtein_distance']}")
            if 'expected_length' in metrics:
                print(f"  - Longitud esperada: {metrics['expected_length']}")
            if 'actual_length' in metrics:
                print(f"  - Longitud obtenida: {metrics['actual_length']}")
        
        print()
    
    print("=" * 80)


def main():
    """Función principal - Ejemplo de uso."""
    print("Script de validación de imágenes de prueba")
    print()
    
    # Cargar textos esperados
    expected_texts = load_expected_texts('data/prueba/textos_esperados.txt')
    
    print(f"Textos esperados cargados para {len(expected_texts)} imágenes:")
    for filename, texts in expected_texts.items():
        print(f"  - {filename}: {len(texts)} línea(s)")
    print()
    
    # Ejemplo de validación (simulado)
    print("=" * 80)
    print("EJEMPLO DE VALIDACIÓN")
    print("=" * 80)
    print()
    print("Para usar este script con resultados reales:")
    print()
    print("1. Descifra las imágenes usando decrypt_image.py")
    print("2. Guarda los resultados en una variable")
    print("3. Valida usando validate_result()")
    print()
    print("Ejemplo de código:")
    print()
    print("    # Simular resultado de descifrado")
    print("    decrypted = 'HELLO WORLD'")
    print("    ")
    print("    # Validar")
    print("    success, message, metrics = validate_result(")
    print("        'hello_world.png', ")
    print("        decrypted,")
    print("        expected_texts")
    print("    )")
    print("    ")
    print("    print(message)")
    print("    print(f'Precisión: {metrics[\"character_accuracy\"]:.2f}%')")
    print()
    
    # Ejemplo con datos simulados
    print("=" * 80)
    print("EJEMPLO CON DATOS SIMULADOS")
    print("=" * 80)
    print()
    
    # Simular algunos resultados
    simulated_results = [
        ('hello_world.png', 'HELLO WORLD'),  # Correcto
        ('alfabeto.png', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),  # Correcto
        ('frase_corta.png', 'DECODE THIS MESAGE'),  # Error: falta S
        ('pangrama_ingles.png', 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG'),  # Correcto
    ]
    
    validation_results = []
    for filename, decrypted in simulated_results:
        success, message, metrics = validate_result(filename, decrypted, expected_texts)
        validation_results.append((filename, success, message, metrics))
    
    print_validation_report(validation_results)
    
    print()
    print("NOTA: Estos son resultados simulados para demostración.")
    print("Para validar resultados reales, integra este script con decrypt_image.py")


if __name__ == '__main__':
    main()
