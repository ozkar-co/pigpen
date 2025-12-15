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

def detect_text_lines(binary_image, min_line_height=10, debug=False):
    """
    Fase 1: Detecta las líneas o renglones que contienen texto usando proyección horizontal.
    
    Args:
        binary_image: Imagen binaria preprocesada
        min_line_height: Altura mínima de una línea de texto
        debug: Si es True, muestra imágenes de depuración
        
    Returns:
        Lista de tuplas (y_inicio, y_fin) que representan las líneas de texto
    """
    # Calcular la proyección horizontal (suma de píxeles por fila)
    horizontal_projection = np.sum(binary_image, axis=1)
    
    # Normalizar la proyección
    if horizontal_projection.max() > 0:
        horizontal_projection = horizontal_projection / horizontal_projection.max()
    
    # Encontrar regiones con contenido (umbral adaptativo)
    threshold = 0.1  # Umbral para detectar presencia de contenido
    content_rows = horizontal_projection > threshold
    
    # Encontrar límites de las líneas
    lines = []
    in_line = False
    line_start = 0
    
    for i, has_content in enumerate(content_rows):
        if has_content and not in_line:
            # Inicio de una nueva línea
            line_start = i
            in_line = True
        elif not has_content and in_line:
            # Fin de la línea actual
            line_end = i
            if line_end - line_start >= min_line_height:
                lines.append((line_start, line_end))
            in_line = False
    
    # Manejar el caso donde la línea llega hasta el final
    if in_line:
        line_end = len(content_rows)
        if line_end - line_start >= min_line_height:
            lines.append((line_start, line_end))
    
    if debug:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Imagen Binaria')
        for y_start, y_end in lines:
            plt.axhline(y=y_start, color='r', linestyle='--', linewidth=1)
            plt.axhline(y=y_end, color='r', linestyle='--', linewidth=1)
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        plt.plot(horizontal_projection, range(len(horizontal_projection)))
        plt.gca().invert_yaxis()
        plt.title('Proyección Horizontal')
        plt.xlabel('Densidad de píxeles (normalizada)')
        plt.ylabel('Fila')
        for y_start, y_end in lines:
            plt.axhline(y=y_start, color='r', linestyle='--', linewidth=1)
            plt.axhline(y=y_end, color='r', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return lines

def extract_characters_from_line(line_image, min_char_width=10, padding=5, debug=False):
    """
    Fase 2: Extrae caracteres individuales de una línea de texto usando proyección vertical.
    Asume que los caracteres son aproximadamente cuadrados.
    
    Args:
        line_image: Imagen binaria de una línea de texto
        min_char_width: Ancho mínimo de un carácter
        padding: Padding alrededor de cada carácter
        debug: Si es True, muestra imágenes de depuración
        
    Returns:
        Lista de imágenes de caracteres segmentados
    """
    # Calcular la proyección vertical (suma de píxeles por columna)
    vertical_projection = np.sum(line_image, axis=0)
    
    # Normalizar la proyección
    if vertical_projection.max() > 0:
        vertical_projection = vertical_projection / vertical_projection.max()
    
    # Encontrar regiones con contenido
    threshold = 0.1  # Umbral para detectar presencia de contenido
    content_cols = vertical_projection > threshold
    
    # Encontrar límites de los caracteres
    char_regions = []
    in_char = False
    char_start = 0
    
    for i, has_content in enumerate(content_cols):
        if has_content and not in_char:
            # Inicio de un nuevo carácter
            char_start = i
            in_char = True
        elif not has_content and in_char:
            # Fin del carácter actual
            char_end = i
            if char_end - char_start >= min_char_width:
                char_regions.append((char_start, char_end))
            in_char = False
    
    # Manejar el caso donde el carácter llega hasta el final
    if in_char:
        char_end = len(content_cols)
        if char_end - char_start >= min_char_width:
            char_regions.append((char_start, char_end))
    
    # Extraer caracteres con padding
    characters = []
    line_height = line_image.shape[0]
    
    for x_start, x_end in char_regions:
        # Calcular el ancho del carácter
        char_width = x_end - x_start
        
        # Asumir que los caracteres son aproximadamente cuadrados
        # Usar el tamaño promedio entre ancho y alto para hacer el recorte
        expected_size = max(char_width, line_height)
        
        # Ajustar para mantener proporciones cuadradas si es necesario
        # (esto ayuda a mantener la forma original del carácter)
        
        # Añadir padding
        x_with_padding_start = max(0, x_start - padding)
        x_with_padding_end = min(line_image.shape[1], x_end + padding)
        
        # Extraer el carácter
        char_img = line_image[:, x_with_padding_start:x_with_padding_end]
        
        # Asegurarse de que la imagen no está vacía
        if char_img.size > 0 and char_img.any():
            characters.append(char_img)
    
    if debug and characters:
        fig, axes = plt.subplots(1, len(characters) + 1, figsize=(15, 3))
        
        # Mostrar la línea completa con regiones marcadas
        debug_line = cv2.cvtColor(line_image.copy(), cv2.COLOR_GRAY2BGR)
        for x_start, x_end in char_regions:
            cv2.rectangle(debug_line, (x_start, 0), (x_end, line_height), (0, 255, 0), 2)
        axes[0].imshow(debug_line)
        axes[0].set_title('Línea con caracteres detectados')
        axes[0].axis('off')
        
        # Mostrar cada carácter extraído
        for i, char in enumerate(characters):
            axes[i+1].imshow(char, cmap='gray')
            axes[i+1].set_title(f'Char {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return characters

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
    lines = detect_text_lines(binary_image, min_line_height=min_size, debug=debug)
    
    if not lines:
        print("No se detectaron líneas de texto.")
        return []
    
    print(f"Detectadas {len(lines)} líneas de texto.")
    
    # Fase 2: Extraer caracteres de cada línea
    all_characters = []
    
    for line_idx, (y_start, y_end) in enumerate(lines):
        # Extraer la región de la línea
        line_image = binary_image[y_start:y_end, :]
        
        # Extraer caracteres de esta línea
        line_characters = extract_characters_from_line(
            line_image, 
            min_char_width=min_size // 2,
            padding=padding,
            debug=debug
        )
        
        # Guardar los caracteres con sus coordenadas originales
        for char_img in line_characters:
            # Calcular las coordenadas en la imagen original
            # (necesitamos estimar la posición x basándonos en el orden)
            all_characters.append({
                'image': char_img,
                'line': line_idx,
                'bbox': (y_start, 0, y_end, char_img.shape[1])  # Aproximado
            })
    
    if debug and all_characters:
        # Visualizar todos los caracteres detectados
        debug_img = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Dibujar líneas detectadas
        for y_start, y_end in lines:
            cv2.line(debug_img, (0, y_start), (binary_image.shape[1], y_start), (255, 0, 0), 2)
            cv2.line(debug_img, (0, y_end), (binary_image.shape[1], y_end), (255, 0, 0), 2)
        
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
            axes[i].set_title(f'L{char["line"]+1}-C{i+1}')
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