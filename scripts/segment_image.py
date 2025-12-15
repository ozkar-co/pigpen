#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para segmentar imágenes que contienen múltiples caracteres del cifrado Pigpen
y dividirlas en imágenes individuales para cada carácter.
"""

import os
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

# Try to import from the same directory
try:
    from segmentation_utils import detect_text_lines, extract_characters_from_line, CONTENT_THRESHOLD
except ImportError:
    from scripts.segmentation_utils import detect_text_lines, extract_characters_from_line, CONTENT_THRESHOLD

def parse_arguments():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Segmenta una imagen con caracteres Pigpen en caracteres individuales.')
    parser.add_argument('--input', '-i', required=True, help='Ruta a la imagen de entrada')
    parser.add_argument('--output', '-o', default='data/unclassified', help='Directorio de salida para los caracteres segmentados')
    parser.add_argument('--min-size', type=int, default=20, help='Tamaño mínimo de los componentes conectados')
    parser.add_argument('--padding', type=int, default=10, help='Padding alrededor de cada carácter')
    parser.add_argument('--debug', action='store_true', help='Mostrar imágenes de depuración')
    return parser.parse_args()

def preprocess_image(image):
    """
    Preprocesa la imagen para facilitar la segmentación.
    
    Args:
        image: Imagen de entrada en formato BGR
        
    Returns:
        Imagen binaria preprocesada
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Aplicar umbral adaptativo para binarizar la imagen
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary

def segment_characters(binary_image, min_size=20, padding=10, debug=False):
    """
    Segmenta los caracteres individuales de la imagen binaria usando un enfoque de dos fases:
    Fase 1: Detecta líneas de texto
    Fase 2: Extrae caracteres de cada línea
    
    Args:
        binary_image: Imagen binaria preprocesada
        min_size: Tamaño mínimo de los componentes (usado para filtrado)
        padding: Padding alrededor de cada carácter
        debug: Si es True, muestra imágenes de depuración
        
    Returns:
        Lista de imágenes de caracteres segmentados con sus coordenadas
    """
    # Fase 1: Detectar líneas de texto
    lines = detect_text_lines(binary_image, min_line_height=min_size, threshold=CONTENT_THRESHOLD)
    
    if not lines:
        # Fallback: usar contornos cuando no se detectan líneas
        # (útil para imágenes con una sola línea o pocos caracteres)
        print("No se detectaron líneas de texto. Usando detección por contornos.")
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_characters = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_size:
                x, y, w, h = cv2.boundingRect(contour)
                # Añadir padding
                x_padded = max(0, x - padding)
                y_padded = max(0, y - padding)
                w_padded = min(binary_image.shape[1] - x_padded, w + 2 * padding)
                h_padded = min(binary_image.shape[0] - y_padded, h + 2 * padding)
                
                # Extraer el carácter
                char_img = binary_image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
                
                if char_img.size > 0 and char_img.any():
                    all_characters.append({
                        'image': char_img,
                        'bbox': (y_padded, x_padded, y_padded+h_padded, x_padded+w_padded)
                    })
        
        # Ordenar por posición horizontal
        all_characters.sort(key=lambda c: c['bbox'][1])
        return all_characters
    
    print(f"Detectadas {len(lines)} líneas de texto.")
    
    # Fase 2: Extraer caracteres de cada línea
    all_characters = []
    
    for line_idx, (y_start, y_end) in enumerate(lines):
        # Extraer la región de la línea
        line_image = binary_image[y_start:y_end, :]
        
        # Extraer caracteres de esta línea usando proyección vertical
        char_regions = extract_characters_from_line(
            line_image, 
            y_offset=y_start,
            min_char_width=min_size // 2,
            padding=padding,
            threshold=CONTENT_THRESHOLD
        )
        
        # Guardar los caracteres con sus coordenadas
        for x, y, w, h in char_regions:
            # Extraer la imagen del carácter
            char_img = binary_image[y:y+h, x:x+w]
            
            if char_img.size > 0 and char_img.any():
                all_characters.append({
                    'image': char_img,
                    'line': line_idx,
                    'bbox': (y, x, y+h, x+w)  # (minr, minc, maxr, maxc) format for compatibility
                })
    
    if debug and all_characters:
        # Visualizar todos los caracteres detectados
        debug_img = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Dibujar líneas detectadas
        for y_start, y_end in lines:
            cv2.line(debug_img, (0, y_start), (binary_image.shape[1], y_start), (255, 0, 0), 2)
            cv2.line(debug_img, (0, y_end), (binary_image.shape[1], y_end), (255, 0, 0), 2)
        
        # Dibujar rectángulos de caracteres
        for char in all_characters:
            minr, minc, maxr, maxc = char['bbox']
            cv2.rectangle(debug_img, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(debug_img)
        plt.title(f'Líneas Detectadas: {len(lines)}, Caracteres Detectados: {len(all_characters)}')
        plt.axis('off')
        plt.show()
        
        # Mostrar los caracteres segmentados
        num_chars = len(all_characters)
        cols = min(10, num_chars)
        rows = (num_chars + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))
        if num_chars == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, char in enumerate(all_characters):
            axes[i].imshow(char['image'], cmap='gray')
            line_label = f'L{char.get("line", 0)+1}-C{i+1}' if 'line' in char else f'C{i+1}'
            axes[i].set_title(line_label)
            axes[i].axis('off')
        
        # Ocultar ejes vacíos
        for i in range(num_chars, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return all_characters

def save_characters(characters, output_dir):
    """
    Guarda los caracteres segmentados en el directorio de salida.
    
    Args:
        characters: Lista de imágenes de caracteres segmentados
        output_dir: Directorio donde se guardarán las imágenes
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar un timestamp único para este lote de imágenes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar cada carácter como una imagen individual
    for i, char in enumerate(characters):
        filename = os.path.join(output_dir, f'char_{timestamp}_{i+1:03d}.png')
        cv2.imwrite(filename, char['image'])
        print(f'Carácter guardado: {filename}')

def main():
    """Función principal."""
    args = parse_arguments()
    
    # Verificar que la imagen de entrada existe
    if not os.path.isfile(args.input):
        print(f"Error: No se pudo encontrar la imagen de entrada: {args.input}")
        return
    
    # Cargar la imagen
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: No se pudo cargar la imagen: {args.input}")
        return
    
    # Preprocesar la imagen
    binary = preprocess_image(image)
    
    # Segmentar caracteres
    characters = segment_characters(
        binary, 
        min_size=args.min_size, 
        padding=args.padding,
        debug=args.debug
    )
    
    if not characters:
        print("No se detectaron caracteres en la imagen.")
        return
    
    print(f"Se detectaron {len(characters)} caracteres.")
    
    # Guardar los caracteres segmentados
    save_characters(characters, args.output)
    
    print(f"Proceso completado. Los caracteres segmentados se guardaron en: {args.output}")

if __name__ == "__main__":
    main() 