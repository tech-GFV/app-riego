"""
Funciones de carga de datos desde fuentes externas.
"""
import pandas as pd
import geopandas as gpd
import datetime
from koboextractor import KoboExtractor
import streamlit as st
from streamlit_gsheets import GSheetsConnection


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
    gdf = gdf[['id_riego', 'Has', 'ID_SIMPLE', 'geometry', 'Lote_2', '25-26 V']]
    gdf = gdf.set_geometry('geometry')
    #gdf = gdf.rename_geometry('geometry')
    gdf['id_riego'] = gdf['id_riego'].str.strip()
    #renombrar Lote_2 a lote_albor
    gdf = gdf.rename(columns={'Lote_2': 'lote_albor'})
    gdf = gdf.to_crs("WGS84")

    
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


def cargar_planificacion_riego(spreadsheet_url):
    """
    Lee los datos de planificación de riego desde Google Sheets usando streamlit-gsheets.
    """
    try:

        # Crear conexión con Google Sheets
        conn = st.connection("gsheets", type=GSheetsConnection)

        # Leer las dos primeras columnas (Semana (lunes) y Lote Albor)
        # Intenta primero sin especificar sheet
        try:
            df = conn.read(spreadsheet=spreadsheet_url, usecols=[0, 1])
        except:
            # Si falla, intenta con sheet="Planificacion"
            df = conn.read(spreadsheet=spreadsheet_url, sheet="Planificacion", usecols=[0, 1])

        # Renombrar columnas esperadas
        # Asumiendo que las columnas son "Semana (lunes)" y "Lote Albor"
        if 'Semana (lunes)' in df.columns:
            df = df.rename(columns={'Semana (lunes)': 'semana_lunes'})
        elif len(df.columns) >= 1:
            # Si no tiene el nombre esperado, usar la primera columna
            df = df.rename(columns={df.columns[0]: 'semana_lunes'})

        if 'Lote Albor' in df.columns:
            df = df.rename(columns={'Lote Albor': 'lote_albor'})
        elif len(df.columns) >= 2:
            # Si no tiene el nombre esperado, usar la segunda columna
            df = df.rename(columns={df.columns[1]: 'lote_albor'})

        # Convertir la columna de semana a fecha
        if 'semana_lunes' in df.columns:
            # Intentar diferentes formatos de fecha
            df['semana_lunes'] = pd.to_datetime(df['semana_lunes'], errors='coerce', dayfirst=True)
            df['semana_lunes'] = df['semana_lunes'].dt.date

        # Eliminar filas con valores nulos en semana_lunes o lote_albor
        df = df.dropna(subset=['semana_lunes', 'lote_albor'])

        # Limpiar espacios en blanco en lote_albor
        if 'lote_albor' in df.columns:
            df['lote_albor'] = df['lote_albor'].astype(str).str.strip()

        return df

    except Exception as e:
        st.error(f"❌ Error al cargar planificación de riego: {e}")
        import traceback
        st.code(traceback.format_exc())
        # Retornar DataFrame vacío en caso de error
        return pd.DataFrame(columns=['semana_lunes', 'lote_albor'])