"""
Funciones de carga de datos desde fuentes externas.
"""
import pandas as pd
import geopandas as gpd
import datetime
from koboextractor import KoboExtractor


def cargar_datos_kobo(kobo_token, form_id):
    """
    Carga los datos de Kobo en un DataFrame.
    """
    kobo = KoboExtractor(kobo_token, 'https://eu.kobotoolbox.org/api/v2')
    json_data = kobo.get_data(form_id, query=None, start=None, limit=None, submitted_after=None)
    df = pd.json_normalize(json_data['results'])
    # Renombrar columnas y eliminar columnas innecesarias
    df = df[['end', 'Acci_n', 'chacra', '_submitted_by']]
    df = df.rename(columns={'end': 'timestamp', 'Acci_n': 'accion', 'chacra': 'id_riego', '_submitted_by': 'usuario'})
    df['id_riego'] = df['id_riego'].str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
    # Filtrar los datos de 2023 en adelante
    df = df[df['timestamp'].dt.year >= 2023]
    return df


def cargar_chacras(path):
    """
    Lee el shapefile de parcelas, normaliza el ID y re-proyecta a WGS84.
    """
    gdf = gpd.read_file(path)
    gdf = gdf[['id_riego', 'Has', 'ID_SIMPLE', 'geometry']]
    gdf = gdf.set_geometry('geometry')
    #gdf = gdf.rename_geometry('geometry')
    gdf['id_riego'] = gdf['id_riego'].str.strip()
    gdf = gdf.to_crs("WGS84")
    print(gdf)
    
    return gdf


def cargar_caudales_cdp(url):
    """
    Lee los datos de caudal de COIRCO desde el URL y devuelve un DataFrame.
    """
    df = pd.read_excel(url, skiprows=18, engine='xlrd')
    # Quedarme solo con la columna  Unnamed: 1; Unnamed: 2 y Total
    df = df[['Unnamed: 1', 'Unnamed: 2', 'Total']]
    # Eliminar las dos primeras filas
    df = df.iloc[2:].reset_index(drop=True)
    # Renombrar Unnamed: 1 = Mes; Unnamed: 2 = Dia
    df = df.rename(columns={'Unnamed: 1': 'Año-Mes', 'Unnamed: 2': 'Dia', 'Total': 'caudal'})
    # Completar la columna Año-Mes con el valor de arriba
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    
    # Función para procesar cada valor de año-mes
    def procesar_año_mes(valor):
        if pd.isna(valor):
            return None

        # Si es datetime, extraer año-mes
        if isinstance(valor, datetime.datetime):
            return valor.strftime('%Y-%m')

        # Si es string, limpiar y formatear
        valor_str = str(valor).strip().replace('.', '').replace(' ', '')

        # Si ya está en formato YYYY-MM, devolverlo
        if len(valor_str.split('-')) == 2 and len(valor_str.split('-')[0]) == 4:
            return valor_str

        # Si está en formato mes-año abreviado (ej: oct-22)
        try:
            fecha = pd.to_datetime(valor_str, format='%b-%y')
            return fecha.strftime('%Y-%m')
        except:
            return None

    # Procesar la columna año-mes
    año_mes = df.iloc[:, 0].apply(procesar_año_mes)

    # Crear la fecha combinando año-mes y día
    df['fecha'] = pd.to_datetime(año_mes + '-' + df.iloc[:, 1].astype(str), errors='coerce')

    # Quedarme solo con la columna  caudal y fecha
    df = df[['fecha', 'caudal']]

    return df