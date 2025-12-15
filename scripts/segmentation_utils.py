#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilidades comunes para la segmentación de caracteres del cifrado Pigpen.
Contiene las funciones de detección de líneas y extracción de caracteres.
"""

import cv2
import numpy as np

# Umbral para detectar presencia de contenido (configurable)
CONTENT_THRESHOLD = 0.1

def detect_text_lines(binary_image, min_line_height=10, threshold=CONTENT_THRESHOLD):
    """
    Fase 1: Detecta las líneas o renglones que contienen texto usando proyección horizontal.
    
    Args:
        binary_image: Imagen binaria preprocesada
        min_line_height: Altura mínima de una línea de texto
        threshold: Umbral para detectar presencia de contenido (0-1)
        
    Returns:
        Lista de tuplas (y_inicio, y_fin) que representan las líneas de texto
    """
    # Calcular la proyección horizontal (suma de píxeles por fila)
    horizontal_projection = np.sum(binary_image, axis=1)
    
    # Normalizar la proyección
    if horizontal_projection.max() > 0:
        horizontal_projection = horizontal_projection / horizontal_projection.max()
    
    # Encontrar regiones con contenido
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
    
    return lines

def extract_characters_from_line(line_image, y_offset=0, min_char_width=10, 
                                  padding=5, threshold=CONTENT_THRESHOLD):
    """
    Fase 2: Extrae caracteres individuales de una línea de texto usando proyección vertical.
    Asume que los caracteres son aproximadamente cuadrados.
    
    Args:
        line_image: Imagen binaria de una línea de texto
        y_offset: Desplazamiento vertical de la línea en la imagen original
        min_char_width: Ancho mínimo de un carácter
        padding: Padding alrededor de cada carácter
        threshold: Umbral para detectar presencia de contenido (0-1)
        
    Returns:
        Lista de tuplas (x, y, w, h) que representan las regiones de caracteres
    """
    # Calcular la proyección vertical (suma de píxeles por columna)
    vertical_projection = np.sum(line_image, axis=0)
    
    # Normalizar la proyección
    if vertical_projection.max() > 0:
        vertical_projection = vertical_projection / vertical_projection.max()
    
    # Encontrar regiones con contenido
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
                # Añadir padding
                x = max(0, char_start - padding)
                w = min(line_image.shape[1] - x, char_end - char_start + 2 * padding)
                h = line_image.shape[0]
                y = y_offset
                char_regions.append((x, y, w, h))
            in_char = False
    
    # Manejar el caso donde el carácter llega hasta el final
    if in_char:
        char_end = len(content_cols)
        if char_end - char_start >= min_char_width:
            x = max(0, char_start - padding)
            w = min(line_image.shape[1] - x, char_end - char_start + 2 * padding)
            h = line_image.shape[0]
            y = y_offset
            char_regions.append((x, y, w, h))
    
    return char_regions
