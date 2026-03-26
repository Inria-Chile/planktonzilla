"""
Configuración de pytest para el framework CLIP OOD.

Añade el directorio raíz del proyecto al path de Python para que todos los
imports de utils.* y supreme.* funcionen correctamente desde cualquier test.
"""

import sys
import os

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.dirname(__file__))
