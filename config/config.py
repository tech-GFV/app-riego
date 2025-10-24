"""
Configuración y constantes del proyecto.
"""
import os
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorio de datos
DATA_DIR = BASE_DIR / 'data' / 'raw'

# Credenciales Kobo
KOBO_TOKEN = 'c7e3cb8f6ae27f4e35148c5e529e473491bfa373'
KOBO_FORM_ID = 'aM693SUegTpTjVKobB7d2h'
KOBO_API_URL = 'https://eu.kobotoolbox.org/api/v2'

# Rutas de archivos
PATH_SHAPEFILE = DATA_DIR / '251022_Lotes_SN.shp'
URL_COIRCO = "https://www.coirco.gov.ar/download/hidrologia/hidrologia_cp01.xls"

# Configuración de riego
MAX_HORAS_RIEGO = 24
FECHA_INICIO_DATOS = '2023-01-01'

# Configuración de campaña
MES_INICIO_CAMPAÑA = 7  # Julio

# Google Sheets - Planificación de Riego
# Reemplazar con la URL de tu Google Sheets (debe ser público o compartido)
PLANIFICACION_SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1ztSwLN2nttkV16ALsTp53SQ-OrYKkvSBmi4McA46ULM/edit?usp=sharing'