#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para crear imágenes de prueba con texto conocido en cifrado Pigpen.
Concatena imágenes de caracteres individuales de la carpeta data/classified
para formar palabras y frases conocidas.
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import random


def get_character_image(letter, classified_dir):
    """
    Obtiene una imagen de un carácter específico de la carpeta classified.
    
    Args:
        letter: Letra a obtener (A-Z)
        classified_dir: Directorio donde están las imágenes clasificadas
    
    Returns:
        Imagen del carácter o None si no se encuentra
    """
    letter = letter.upper()
    letter_dir = os.path.join(classified_dir, letter)
    
    if not os.path.exists(letter_dir):
        print(f"Advertencia: No se encontró el directorio para la letra {letter}")
        return None
    
    # Obtener todas las imágenes de esa letra
    images = [f for f in os.listdir(letter_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"Advertencia: No hay imágenes para la letra {letter}")
        return None
    
    # Seleccionar una imagen aleatoria
    selected_image = random.choice(images)
    image_path = os.path.join(letter_dir, selected_image)
    
    # Cargar la imagen
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


def resize_to_common_height(images, target_height=100):
    """
    Redimensiona todas las imágenes a una altura común manteniendo la relación de aspecto.
    
    Args:
        images: Lista de imágenes
        target_height: Altura objetivo
    
    Returns:
        Lista de imágenes redimensionadas
    """
    resized = []
    for img in images:
        if img is None:
            continue
        h, w = img.shape
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, target_height))
        resized.append(resized_img)
    return resized


def concatenate_horizontally(images, spacing=10):
    """
    Concatena imágenes horizontalmente con espaciado.
    
    Args:
        images: Lista de imágenes a concatenar
        spacing: Espacio entre imágenes en píxeles
    
    Returns:
        Imagen concatenada
    """
    if not images:
        return None
    
    # Obtener la altura máxima
    max_height = max(img.shape[0] for img in images)
    
    # Calcular el ancho total
    total_width = sum(img.shape[1] for img in images) + spacing * (len(images) - 1)
    
    # Crear una imagen en blanco
    result = np.zeros((max_height, total_width), dtype=np.uint8)
    
    # Pegar cada imagen
    x_offset = 0
    for img in images:
        h, w = img.shape
        y_offset = (max_height - h) // 2  # Centrar verticalmente
        result[y_offset:y_offset+h, x_offset:x_offset+w] = img
        x_offset += w + spacing
    
    return result


def create_text_image(text, classified_dir, output_path, target_height=100, spacing=10):
    """
    Crea una imagen concatenando caracteres para formar texto.
    
    Args:
        text: Texto a crear
        classified_dir: Directorio con imágenes clasificadas
        output_path: Ruta donde guardar la imagen resultante
        target_height: Altura de los caracteres
        spacing: Espacio entre caracteres
    
    Returns:
        True si se creó exitosamente, False en caso contrario
    """
    # Obtener imágenes para cada carácter
    char_images = []
    for char in text.upper():
        if char == ' ':
            # Agregar un espacio en blanco
            space_img = np.zeros((target_height, target_height // 2), dtype=np.uint8)
            char_images.append(space_img)
        elif char.isalpha():
            img = get_character_image(char, classified_dir)
            if img is not None:
                char_images.append(img)
    
    if not char_images:
        print(f"Error: No se pudieron obtener imágenes para el texto '{text}'")
        return False
    
    # Redimensionar a altura común
    resized_images = resize_to_common_height(char_images, target_height)
    
    # Concatenar horizontalmente
    result_image = concatenate_horizontally(resized_images, spacing)
    
    if result_image is None:
        print(f"Error: No se pudo crear la imagen para el texto '{text}'")
        return False
    
    # Guardar la imagen
    cv2.imwrite(output_path, result_image)
    print(f"Imagen creada: {output_path}")
    print(f"  Texto: {text}")
    print(f"  Dimensiones: {result_image.shape[1]}x{result_image.shape[0]}")
    
    return True


def create_multiline_image(lines, classified_dir, output_path, target_height=100, 
                          spacing=10, line_spacing=20):
    """
    Crea una imagen con múltiples líneas de texto.
    
    Args:
        lines: Lista de líneas de texto
        classified_dir: Directorio con imágenes clasificadas
        output_path: Ruta donde guardar la imagen
        target_height: Altura de los caracteres
        spacing: Espacio entre caracteres
        line_spacing: Espacio entre líneas
    
    Returns:
        True si se creó exitosamente, False en caso contrario
    """
    line_images = []
    
    # Crear imagen para cada línea
    for line in lines:
        char_images = []
        for char in line.upper():
            if char == ' ':
                space_img = np.zeros((target_height, target_height // 2), dtype=np.uint8)
                char_images.append(space_img)
            elif char.isalpha():
                img = get_character_image(char, classified_dir)
                if img is not None:
                    char_images.append(img)
        
        if char_images:
            resized_images = resize_to_common_height(char_images, target_height)
            line_img = concatenate_horizontally(resized_images, spacing)
            if line_img is not None:
                line_images.append(line_img)
    
    if not line_images:
        print("Error: No se pudieron crear las líneas de texto")
        return False
    
    # Calcular dimensiones de la imagen final
    max_width = max(img.shape[1] for img in line_images)
    total_height = sum(img.shape[0] for img in line_images) + line_spacing * (len(line_images) - 1)
    
    # Crear imagen en blanco
    result = np.zeros((total_height, max_width), dtype=np.uint8)
    
    # Pegar cada línea
    y_offset = 0
    for img in line_images:
        h, w = img.shape
        result[y_offset:y_offset+h, 0:w] = img
        y_offset += h + line_spacing
    
    # Guardar la imagen
    cv2.imwrite(output_path, result)
    print(f"Imagen multilínea creada: {output_path}")
    print(f"  Líneas: {len(lines)}")
    print(f"  Dimensiones: {result.shape[1]}x{result.shape[0]}")
    
    return True




def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Crea imágenes de prueba con texto conocido en cifrado Pigpen'
    )
    parser.add_argument('--classified-dir', default='data/classified',
                       help='Directorio con imágenes clasificadas')
    parser.add_argument('--output-dir', default='data/prueba',
                       help='Directorio de salida para las imágenes de prueba')
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Creando imágenes de prueba para el proyecto PigPen")
    print("=" * 70)
    print()
    
    # 1. Pangrama en español: "El veloz murciélago hindú comía feliz cardillo y kiwi"
    # Versión simplificada sin acentos ni diéresis
    print("1. Creando pangrama en español...")
    pangram_es = "EL VELOZ MURCIELAGO HINDU COMIA FELIZ CARDILLO Y KIWI"
    create_text_image(
        pangram_es,
        args.classified_dir,
        os.path.join(args.output_dir, 'pangrama_espanol.png'),
        target_height=80,
        spacing=8
    )
    print()
    
    # 2. Pangrama en inglés: "The quick brown fox jumps over the lazy dog"
    print("2. Creando pangrama en inglés...")
    pangram_en = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    create_text_image(
        pangram_en,
        args.classified_dir,
        os.path.join(args.output_dir, 'pangrama_ingles.png'),
        target_height=80,
        spacing=8
    )
    print()
    
    # 3. Texto simple: "HELLO WORLD"
    print("3. Creando texto simple 'HELLO WORLD'...")
    create_text_image(
        "HELLO WORLD",
        args.classified_dir,
        os.path.join(args.output_dir, 'hello_world.png'),
        target_height=100,
        spacing=10
    )
    print()
    
    # 4. Alfabeto completo
    print("4. Creando alfabeto completo...")
    create_text_image(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        args.classified_dir,
        os.path.join(args.output_dir, 'alfabeto.png'),
        target_height=80,
        spacing=8
    )
    print()
    
    # 5. Mensaje multilínea
    print("5. Creando mensaje multilínea...")
    multiline_text = [
        "PIGPEN CIPHER",
        "ALSO KNOWN AS",
        "MASONIC CIPHER"
    ]
    create_multiline_image(
        multiline_text,
        args.classified_dir,
        os.path.join(args.output_dir, 'mensaje_multilinea.png'),
        target_height=80,
        spacing=8,
        line_spacing=30
    )
    print()
    
    # 6. Frase corta
    print("6. Creando frase corta...")
    create_text_image(
        "DECODE THIS MESSAGE",
        args.classified_dir,
        os.path.join(args.output_dir, 'frase_corta.png'),
        target_height=100,
        spacing=10
    )
    print()
    
    print("=" * 70)
    print("Proceso completado")
    print("=" * 70)
    print(f"Las imágenes de prueba se guardaron en: {args.output_dir}")
    print()
    print("Imágenes creadas:")
    print("  - pangrama_espanol.png: Pangrama en español sin acentos")
    print("  - pangrama_ingles.png: 'The quick brown fox jumps over the lazy dog'")
    print("  - hello_world.png: 'HELLO WORLD'")
    print("  - alfabeto.png: A-Z completo")
    print("  - mensaje_multilinea.png: Mensaje en 3 líneas")
    print("  - frase_corta.png: 'DECODE THIS MESSAGE'")



if __name__ == '__main__':
    main()
