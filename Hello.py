import pandas as pd
import datetime
import numpy as np
import geopandas as gpd
import plotly.express as px
import streamlit as st
from streamlit.logger import get_logger
import requests
from lxml import html
from streamlit_gsheets import GSheetsConnection
from koboextractor import KoboExtractor
from datetime import date

LOGGER = get_logger(__name__)

# =============================================================================
# Preprocesamiento / Carga de datos
# =============================================================================

@st.cache_data(ttl=60, max_entries=1)
def cargar_appsheet():
    """
    Lee la tabla base (AppSheet -> Google Sheets) y devuelve un DF con el
    MISMO esquema que 'cargar_kobo' generaba:
      - ID_chacra  (desde 'Compuerta', sin espacios)
      - ID         (ID del evento)
      - Acci_n     ('apertura' / 'cierre')
      - end        (timestamp del evento)
      - _submitted_by (usuario que registró)
    """
    # 1) Conexión a la hoja que usa AppSheet
    conn = st.connection("gsheets", type=GSheetsConnection)
    url = st.secrets.connections.appsheet.spreadsheet

    # Si necesitás una pestaña específica:
    worksheet_registros = st.secrets.connections.appsheet.get("Registros", None)
    df = conn.read(spreadsheet=url, worksheet=worksheet_registros)

    # 2) Normalización de columnas desde AppSheet
    # Esperado en la hoja (según captura): ID | Fecha y hora | Compuerta | Acción | Usuario
    df = df.rename(columns={
        "Fecha y hora": "end",
        "Compuerta": "ID_chacra",
        "Acción": "Acci_n",
        "Usuario": "_submitted_by",
    })

    # 3) Tipos y limpieza
    df['end'] = pd.to_datetime(df['end'], dayfirst=True, errors='coerce')
    df['ID_chacra'] = df['ID_chacra'].astype(str).str.replace(' ', '', regex=False)

    # Mapear valores de AppSheet a los que usabas con Kobo
    # (Kobo: 'apertura'/'cierre'; AppSheet: 'ABRIR'/'CERRAR')
    df['Acci_n'] = (
        df['Acci_n']
        .astype(str).str.strip().str.upper()
        .map({"ABRIR": "apertura", "CERRAR": "cierre"})
        .fillna("cierre")  # fallback conservador
    )

    # 4) Filtro de inicio de campaña (idéntico a tu función Kobo)
    df = df.loc[df['end'] >= '2025-08-18']

    # 5) Selección de columnas (las que tu flujo usa más adelante)
    df = df[['ID_chacra', 'ID', 'Acci_n', 'end', '_submitted_by']].copy()

    return df


def cargar_kobo(token):
    """
    Descarga y normaliza los datos de KoboToolbox.
    - Filtra registros anteriores a '2025-08-18' (inicio campaña dado).
    """
    kobo = KoboExtractor(token, 'https://eu.kobotoolbox.org/api/v2')
    form_id = 'aM693SUegTpTjVKobB7d2h'

    data = kobo.get_data(form_id, query=None, start=None, limit=None, submitted_after=None)
    df_kobo = pd.json_normalize(data['results'])

    # Normalización de ID / columnas innecesarias / fechas
    df_kobo = df_kobo.rename(columns={"chacra": "ID_chacra"})
    df_kobo['ID'] = df_kobo['ID_chacra'].str.strip()
    df_kobo['ID_chacra'] = df_kobo['ID_chacra'].str.replace(' ', '')

    df_kobo = df_kobo.drop(columns=[
        '_id', 'formhub/uuid', 'start', '__version__', 'meta/instanceID',
        '_xform_id_string', '_uuid', '_attachments', '_status',
        '_geolocation', '_submission_time', '_tags', '_notes'
    ])

    df_kobo['end'] = pd.to_datetime(df_kobo['end'])

    # Inicio de campaña (no modificar para mantener el mismo resultado)
    df_kobo = df_kobo.loc[(df_kobo['end'] >= '2025-08-18')]
    return df_kobo


def cargar_chacras():
    """
    Lee el padrón de chacras desde CSV, normaliza IDs y devuelve DataFrame.
    """
    df_chacras = pd.read_csv('base_chacras.csv', sep=';', encoding='utf-8')
    df_chacras = df_chacras.rename(columns={"ID_QR": "ID_chacra"})
    df_chacras['ID_chacra'] = df_chacras['ID_chacra'].str.replace(' ', '')
    df_chacras['ID_xls'] = df_chacras['ID_xls'].str.replace(' ', '')
    return df_chacras


def cargar_geometria():
    """
    Lee el shapefile de parcelas, normaliza el ID y re-proyecta a WGS84.
    """
    sn_shp = gpd.read_file('PARCELAS_.shp')
    sn_shp = sn_shp.rename(columns={"ID_xls": "ID_chacra"})
    sn_shp['ID_chacra'] = sn_shp['ID_chacra'].str.replace(' ', '')
    sn_shp = sn_shp.to_crs("WGS84")
    return sn_shp


# =============================================================================
# Construcción de riegos (apertura/cierre) y unión con chacras
# =============================================================================

def crear_riegos(df_kobo):
    """
    A partir de los registros de Kobo, construye pares apertura/cierre por chacra
    en un rango < 24h y devuelve eventos con tiempos y responsables.
    """
    df_apertura = df_kobo[df_kobo['Acci_n'] == "apertura"]
    df_cierre = df_kobo[df_kobo['Acci_n'] == "cierre"]

    merged = pd.merge(df_apertura, df_cierre, on='ID_chacra', suffixes=('_ap', '_ci'))
    # Emparejar sólo cuando cierre > apertura y dentro de 24h
    filt = merged[
        (merged['end_ci'] > merged['end_ap']) &
        (merged['end_ci'] < merged['end_ap'] + pd.Timedelta(days=1))
    ]

    df_riego = filt[['ID_ap', 'ID_chacra', 'end_ap', 'end_ci', '_submitted_by_ap', '_submitted_by_ci']].copy()
    df_riego.columns = ['ID', 'ID_chacra', 'time_ap', 'time_ci', 'reg_ap', 'reg_ci']
    df_riego['time_regado'] = df_riego['time_ci'] - df_riego['time_ap']

    # Si se desea incorporar GSheet, descomentar:
    # if not df_gsheet.empty:
    #     df_riego = pd.concat([df_riego, df_gsheet])

    return df_riego


def unir_chacra_riego(df_riego_aux, df_chacra):
    """
    Une la info de riegos con el padrón de chacras, calculando:
      - ciclos, tiempo promedio de riego, última fecha de cierre,
      - semana ISO (año/semana), lunes ISO y label 'YYYY-Www'.
    """
    # Resumen de riegos por chacra
    agg = (
        df_riego_aux.groupby('ID_chacra', as_index=False)
        .agg(
            ciclos=('ID_chacra', 'count'),
            t_riego_prom=('time_regado', 'mean'),
            time_ci=('time_ci', 'last'),
            ID=('ID', 'first')
        )
    )

    # Base chacras + resumen (LEFT para mantener las sin riego)
    df_riego = df_chacra[['ID_chacra', 'ID_xls', 'SUPERFICIE', 'ACTIVIDAD', 'ID_CAMPAÑA']] \
        .merge(agg, on='ID_chacra', how='left') \
        .rename(columns={'SUPERFICIE': 'superficie', 'ACTIVIDAD': 'actividad', 'ID_CAMPAÑA': 'ID_campaña'})

    # Normalizar superficie (coma decimal / #N/D)
    def _parse_superficie_local(x):
        s = str(x)
        if s in (None, 'None', '', '#N/D'):
            return 0.0
        return float(s.replace(',', '.'))
    df_riego['superficie'] = df_riego['superficie'].apply(_parse_superficie_local)

    # Campos de fecha/semana ISO (si hay time_ci)
    df_riego['fecha_ult_ejec'] = pd.to_datetime(df_riego['time_ci']).dt.date
    iso = pd.to_datetime(df_riego['time_ci']).dt.isocalendar()
    df_riego['iso_year'] = iso.year.astype('Int64')
    df_riego['iso_week'] = iso.week.astype('Int64')

    # Lunes de la semana ISO (para ordenar y mostrar prolijo)
    def _monday(y, w):
        if pd.isna(y) or pd.isna(w):
            return pd.NaT
        base = pd.to_datetime(date(int(y), 1, 4))  # jueves de la semana 1 ISO
        return base + pd.to_timedelta((int(w) - 1) * 7, unit='D') - pd.to_timedelta(base.weekday(), unit='D')

    df_riego['semana_monday'] = df_riego.apply(lambda r: _monday(r['iso_year'], r['iso_week']), axis=1)

    # Label ISO 'YYYY-Www'
    df_riego['semana_label'] = (
        df_riego['iso_year'].astype('Int64').astype('string') + '-W' +
        df_riego['iso_week'].astype('Int64').astype('string').str.zfill(2)
    )

    # Completar nulos
    df_riego['ciclos'] = df_riego['ciclos'].fillna(0).astype(int)

    return df_riego


# =============================================================================
# Helpers de formato / estructuras temporales
# =============================================================================

def _parse_superficie(x):
    """Normaliza superficie a float, contemplando coma decimal y '#N/D'."""
    s = str(x)
    if s in (None, 'None', '', '#N/D'):
        return 0.0
    return float(s.replace(',', '.'))


def construir_sup_semanal(df_riego_pre, df_chacras, unique_per_week=True):
    """
    Construye un DF semanal con:
      - semana_monday (fecha del lunes ISO),
      - semana_label ('YYYY-Www'),
      - sup_regada (Ha).
    Suma la superficie de chacras que tuvieron al menos un riego esa semana.
    Si unique_per_week=True, una chacra cuenta 1 vez por semana, aunque tenga múltiples riegos.
    """
    # Eventos con cierre + semana ISO
    df = df_riego_pre[['ID_chacra', 'time_ci']].dropna().copy()
    df['time_ci'] = pd.to_datetime(df['time_ci'])
    iso = df['time_ci'].dt.isocalendar()
    df['iso_year'] = iso.year.astype(int)
    df['iso_week'] = iso.week.astype(int)

    # Superficie por chacra
    chac = df_chacras[['ID_chacra', 'SUPERFICIE']].copy()
    chac['SUPERFICIE'] = chac['SUPERFICIE'].apply(_parse_superficie)
    df = df.merge(chac, on='ID_chacra', how='left')

    # Evitar doble conteo de una misma chacra en la MISMA semana
    if unique_per_week:
        df = df.drop_duplicates(subset=['ID_chacra', 'iso_year', 'iso_week'])

    # Fecha lunes ISO + label
    df['semana_monday'] = pd.to_datetime([date.fromisocalendar(y, w, 1) for y, w in zip(df['iso_year'], df['iso_week'])])
    df['semana_label'] = df['iso_year'].astype(str) + '-W' + df['iso_week'].astype(str).str.zfill(2)

    # Completar semanas faltantes con 0
    if not df.empty:
        full_idx = pd.date_range(df['semana_monday'].min(), df['semana_monday'].max(), freq='W-MON')
        sup_semanal = (
            df.groupby('semana_monday', as_index=True)['SUPERFICIE']
              .sum(min_count=1)
              .reindex(full_idx, fill_value=0.0)
              .rename('sup_regada')
              .to_frame()
              .reset_index()
              .rename(columns={'index': 'semana_monday'})
        )
        iso_full = sup_semanal['semana_monday'].dt.isocalendar()
        sup_semanal['semana_label'] = iso_full.year.astype(str) + '-W' + iso_full.week.astype(str).str.zfill(2)
    else:
        sup_semanal = pd.DataFrame(columns=['semana_monday', 'semana_label', 'sup_regada'])

    sup_semanal['sup_regada'] = sup_semanal['sup_regada'].round(0).astype(int)
    return sup_semanal


def construir_ultima_semana_por_chacra(df_riego_pre, df_chacras):
    """
    Para cada chacra, identifica:
      - last_time_ci (último cierre),
      - last_semana_monday (lunes ISO),
      - last_semana_label ('YYYY-Www' o 'Nunca'),
      - has_riego (1 si regó alguna vez).
    """
    ev = df_riego_pre[['ID_chacra', 'time_ci']].dropna().copy()
    ev['time_ci'] = pd.to_datetime(ev['time_ci'])
    ult = (
        ev.sort_values('time_ci')
          .groupby('ID_chacra', as_index=False)
          .tail(1)
          .rename(columns={'time_ci': 'last_time_ci'})
    )

    base = df_chacras[['ID_chacra', 'ID_xls', 'SUPERFICIE', 'ACTIVIDAD']].copy()
    base['SUPERFICIE'] = base['SUPERFICIE'].apply(_parse_superficie)

    df = base.merge(ult, on='ID_chacra', how='left')
    iso = df['last_time_ci'].dt.isocalendar()
    df['iso_year'] = iso.year
    df['iso_week'] = iso.week
    df['last_semana_monday'] = pd.to_datetime([
        date.fromisocalendar(int(y), int(w), 1)
        if pd.notna(y) and pd.notna(w) else pd.NaT
        for y, w in zip(df['iso_year'], df['iso_week'])
    ])
    df['last_semana_label'] = df.apply(
        lambda r: (f"{int(r['iso_year'])}-W{int(r['iso_week']):02d}")
        if pd.notna(r['iso_year']) and pd.notna(r['iso_week']) else "Nunca",
        axis=1
    )
    df['has_riego'] = (df['last_semana_monday'].notna()).astype(int)

    return df.rename(columns={'SUPERFICIE': 'superficie', 'ACTIVIDAD': 'actividad'})


# =============================================================================
# Mapas (Plotly)
# =============================================================================

def mapa_status_compuertas(df_estados, sn_shp):
    """
    Mapa de estado de compuertas (abierta/cerrada/sin cierre).
    """
    fig_status = px.choropleth(
        df_estados,
        geojson=sn_shp.set_index("ID_chacra").geometry,
        locations="ID_xls",
        color="estado",
        color_discrete_map={"abierta": "green", "cerrada": "red", "s/cierre": "black"},
        projection="mercator",
        basemap_visible=True,
        hover_name='ID_chacra',
        hover_data=['Tiempo_estado', 'ACTIVIDAD']
    )
    fig_status.update_geos(fitbounds="geojson")
    fig_status.update_layout(autosize=False, width=800, height=800)
    return fig_status


# Escala continua compartida: 0=rojo (viejo), 1=azul (reciente)
_COLOR_SCALE_R2B = [(0.0, 'red'), (1.0, 'blue')]


def mapa_ultima_semana(df_last, sn_shp, monday_seleccion=None):
    """
    Choropleth de 'última semana regada':
      - Si monday_seleccion es None -> "TODAS": gradiente por recencia (rojo→azul).
      - Si monday_seleccion es un lunes ISO -> binario: sólo esa semana en azul, resto en rojo.
    """
    df = df_last.copy()

    if monday_seleccion is None:
        # ----- MODO TODAS (gradiente por recencia) -----
        valid = df[df['has_riego'] == 1]
        if valid.empty:
            df['score'] = 0.0  # sin riegos: todo rojo
        else:
            mn = valid['last_semana_monday'].min()
            mx = valid['last_semana_monday'].max()
            span = max((mx - mn).days, 1)

            score = (df['last_semana_monday'] - mn).dt.days / span
            score = score.fillna(0.0)  # sin riego -> rojo

            # Si todos regaron la misma semana, los que regaron van a 1 (azul)
            if (mx - mn).days == 0:
                score = np.where(df['has_riego'] == 1, 1.0, 0.0)

            df['score'] = score.astype(float)

        fig = px.choropleth(
            df,
            geojson=sn_shp.set_index("ID_chacra").geometry,
            locations="ID_xls",
            color="score",
            color_continuous_scale=_COLOR_SCALE_R2B,
            range_color=(0.0, 1.0),
            projection="mercator",
            basemap_visible=True,
            hover_data=['actividad', 'superficie', 'last_semana_label']
        )
        fig.update_coloraxes(colorbar_title="Recencia (rojo→azul)")

    else:
        # ----- MODO FILTRO (binario) -----
        df['score'] = np.where(df['last_semana_monday'] == monday_seleccion, 1.0, 0.0)
        fig = px.choropleth(
            df,
            geojson=sn_shp.set_index("ID_chacra").geometry,
            locations="ID_xls",
            color="score",
            color_continuous_scale=_COLOR_SCALE_R2B,
            range_color=(0.0, 1.0),
            projection="mercator",
            basemap_visible=True,
            hover_data=['actividad', 'superficie', 'last_semana_label']
        )
        fig.update_coloraxes(showscale=False)  # sin colorbar en modo filtro

    fig.update_geos(fitbounds="geojson")
    fig.update_layout(autosize=False, width=800, height=800)
    return fig


def mapa_ciclos(df_riego, sn_shp, ciclos):
    """
    Choropleth por cantidad de ciclos (con selector para resaltar un valor).
    """
    df_aux = df_riego.copy()
    if ciclos != "TODAS":
        df_aux['ciclos'] = df_aux['ciclos'].apply(lambda x: x if x == ciclos else 0)

    fig = px.choropleth(
        df_aux,
        geojson=sn_shp.set_index("ID_chacra").geometry,
        locations="ID_xls",
        color="ciclos",
        color_continuous_scale="Bluered_r",
        projection="mercator",
        basemap_visible=True,
        hover_data=['actividad'],
    )
    fig.update_geos(fitbounds="geojson")
    fig.update_layout(autosize=False, width=800, height=800)
    return fig


def mapa_actividad(df_riego, sn_shp):
    """
    Choropleth por actividad declarada en el padrón.
    """
    fig = px.choropleth(
        df_riego,
        geojson=sn_shp.set_index("ID_chacra").geometry,
        locations="ID_xls",
        color="actividad",
        color_continuous_scale="RdBu",
        projection="mercator",
        basemap_visible=True,
        hover_data=['actividad'],
    )
    fig.update_geos(fitbounds="geojson")
    fig.update_layout(autosize=False, width=800, height=800)
    return fig


def mapa_tiempo_promedio(df_riego, sn_shp):
    """
    Choropleth del tiempo promedio de riego por hectárea (hs/ha).
    """
    df_aux = df_riego.copy()
    df_aux['t_riego_prom'] = df_aux['t_riego_prom'].dt.total_seconds() / 3600
    df_aux['t/sup'] = df_aux['t_riego_prom'] / df_aux['superficie']

    fig = px.choropleth(
        df_aux,
        geojson=sn_shp.set_index("ID_chacra").geometry,
        locations="ID_xls",
        color="t/sup",
        color_continuous_scale="Bluered",
        projection="mercator",
        basemap_visible=True,
        hover_data=['t_riego_prom', 'superficie', 'actividad'],
    )
    fig.update_geos(fitbounds="geojson")
    fig.update_layout(autosize=False, width=800, height=800)
    return fig


# =============================================================================
# Gráficos (Plotly)
# =============================================================================

def graficar_sup_semanal(sup_semanal_df):
    """
    Barras de superficie regada por semana (X = lunes ISO).
    """
    fig = px.bar(
        sup_semanal_df,
        x="semana_monday",
        y="sup_regada",
        hover_data=['semana_label'],
        text="sup_regada",
        width=900,
        height=500
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title='Semana (lunes)',
        yaxis_title='Superficie regada [Ha]',
        showlegend=False,
        margin=dict(t=40, l=40, r=20, b=40)
    )
    # Para eje Y fijo, descomentar:
    # fig.update_yaxes(range=[0, 1000])
    return fig


def graficar_riegos_por_regador(df_riego_pre):
    """
    Barras: cantidad de riegos por regador (apertura/cierre y total).
    """
    df_ap = (
        df_riego_pre.groupby('reg_ap')
        .agg(cant_ap=('reg_ap', 'count'))
        .reset_index()
    )
    df_ci = (
        df_riego_pre.groupby('reg_ci')
        .agg(cant_ci=('reg_ci', 'count'))
        .reset_index()
    )
    df_reg = df_ap.merge(df_ci, left_on='reg_ap', right_on='reg_ci')
    df_reg['total'] = df_reg['cant_ap'] + df_reg['cant_ci']
    df_reg = df_reg.sort_values(by=['total'], ascending=False)

    fig = px.bar(df_reg, x='reg_ap', y=['cant_ap', 'cant_ci'], width=900, height=500)
    return fig


# =============================================================================
# Auxiliares (KPIs, estado compuertas, scraping, etc.)
# =============================================================================

def cantidad_compuertas_abiertas(estados_df):
    """Cuenta compuertas con estado 'abierta' para KPI."""
    return estados_df[estados_df['estado'] == 'abierta'].shape[0]


def status_compuertas(df_kobo_raw, df_chacra):
    """
    Reconstruye el estado actual de cada compuerta combinando eventos de apertura y cierre.
    - Si estuvo abierta por más de 24h, se fuerza como 'cerrada'.
    """
    df_ap = df_kobo_raw[df_kobo_raw['Acci_n'] == "apertura"].copy()
    df_ci = df_kobo_raw[df_kobo_raw['Acci_n'] == "cierre"].copy()

    df_ap.rename(columns={'end': 'fecha_hora_apertura'}, inplace=True)
    df_ci.rename(columns={'end': 'fecha_hora_cierre'}, inplace=True)

    df = pd.concat([df_ap, df_ci])
    df['fecha_hora'] = df['fecha_hora_apertura'].fillna(df['fecha_hora_cierre'])
    df.sort_values(by='fecha_hora', inplace=True)

    # Marcador de secuencia por ID (no se utiliza luego, pero se conserva para mantener el resultado)
    df['tipo'] = df.groupby('ID').cumcount().astype(str)

    # Simulación de estado final por compuerta
    estado_actual = {}
    for _, row in df.iterrows():
        id_chacra = row['ID_chacra']
        accion = row['Acci_n']
        fecha = row['fecha_hora']

        if id_chacra not in estado_actual:
            estado_actual[id_chacra] = {'estado': None, 'fecha_hora': None}

        if accion == 'apertura':
            estado_actual[id_chacra]['estado'] = 'abierta'
            estado_actual[id_chacra]['fecha_hora'] = fecha
        elif accion == 'cierre':
            estado_actual[id_chacra]['estado'] = 'cerrada'
            estado_actual[id_chacra]['fecha_hora'] = fecha

    estados_df = pd.DataFrame.from_dict(estado_actual, orient='index').reset_index()
    estados_df.columns = ['ID_chacra', 'estado', 'fecha_hora']

    # Enriquecer con info de chacras
    estados_df = estados_df.merge(
        df_chacra[['ID_chacra', 'ID_xls', 'SUPERFICIE', 'ACTIVIDAD']],
        on='ID_chacra', how='left'
    )

    # Tiempo transcurrido y regla de autocierre > 24h
    estados_df['fecha_hora'] = pd.to_datetime(estados_df['fecha_hora'])
    estados_df['Tiempo_estado'] = (
        pd.to_datetime('now', utc=estados_df['fecha_hora'].dt.tz) - estados_df['fecha_hora']
    ).dt.total_seconds() / 3600
    estados_df['Tiempo_estado'] = estados_df['Tiempo_estado'].round(decimals=0)

    mask_autocierre = (estados_df['Tiempo_estado'] > 24) & (estados_df['estado'] == 'abierta')
    estados_df.loc[mask_autocierre, 'estado'] = 'cerrada'

    return estados_df


def obtener_caudal_casa_piedra():
    """
    Scrapea caudal y diferencia vs. ayer desde coirco.gov.ar (XPath estático).
    IMPORTANTE: No se alteró la lógica para no cambiar resultados en la app.
    """
    url = 'https://www.coirco.gov.ar/'
    response = requests.get(url)

    if response.status_code == 200:
        tree = html.fromstring(response.content)

        xpath_caudal = '/html/body/div[2]/div[1]/div[3]/div/div/p[2]/span[2]/span[2]/text()'
        xpath_caudal_ayer = '/html/body/div[2]/div[1]/div[3]/div/div/p[2]/span[2]/text()'

        caudal_element = tree.xpath(xpath_caudal)
        caudal_ayer_element = tree.xpath(xpath_caudal_ayer)

        raw_caudal = str(caudal_element[1])
        raw_caudal_ayer = str(caudal_ayer_element[1])

        caudal = raw_caudal.strip()

        # Extraer números antes de ' m³/s'
        idx_hoy = raw_caudal.find(' m³/s')
        idx_ayer = raw_caudal_ayer.find(' m³/s')

        caudal_num = raw_caudal[:idx_hoy].strip()
        caudal_ayer_num = raw_caudal_ayer[:idx_ayer].strip()

        try:
            dif_caudal = int(caudal_num) - int(caudal_ayer_num)
        except:
            dif_caudal = 0

    # Mantener el mismo comportamiento (variables tomadas del bloque anterior)
    return caudal, dif_caudal


def obtener_ultimo_registro(df_riego):
    """
    Retorna string con la fecha/hora del último cierre ('time_ci') en formato dd/mm/yy HH:MM.
    """
    df_sorted = df_riego.sort_values('time_ci', ascending=False)
    ultimo = df_sorted.head(1)
    time_str = list(ultimo['time_ci'])[0].strftime("%d/%m/%y %H:%M")
    return time_str


def calcular_kpis(df_riego):
    """
    Devuelve lista [caudal actual, último registro de riego, cantidad total de riegos].
    """
    caudal_casa_piedra = obtener_caudal_casa_piedra()[0]
    ultimo_registro = obtener_ultimo_registro(df_riego)
    cantidad_riegos = df_riego['ciclos'].sum()
    return [caudal_casa_piedra, ultimo_registro, cantidad_riegos]


def mostrar_kpis(kpis, kpi_names, kpis_dif):
    """
    Muestra 3 KPIs en columnas con su delta (cuando aplica).
    """
    st.header("Parametros")
    for col, (kpi_name, kpi_value, kpi_dif) in zip(st.columns(3), zip(kpi_names, kpis, kpis_dif)):
        col.metric(label=kpi_name, value=kpi_value, delta=kpi_dif)


# =============================================================================
# App (Streamlit)
# =============================================================================

def run():
    st.set_page_config(
        page_title="APP RIEGO",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title('🌱 Santa Nicolasa - Faro verde')
    st.header('💧 APP Riego', divider="green")

    # --- Carga de datos principales ---
    KOBO_TOKEN = 'c7e3cb8f6ae27f4e35148c5e529e473491bfa373'
    df_kobo = cargar_kobo(KOBO_TOKEN)
    #df_kobo = cargar_appsheet() #APPSHEET
    df_chacras = cargar_chacras()
    sn_shp = cargar_geometria()
    # df_gsheet = cargar_gsheet()  # opcional

    # Construcciones derivadas
    df_riego_pre = crear_riegos(df_kobo)
    df_regadores = df_riego_pre.merge(
        df_chacras[['ID_chacra', 'ID_xls', 'SUPERFICIE', 'ACTIVIDAD']],
        on='ID_chacra', how='left'
    )
    df_riego = unir_chacra_riego(df_riego_pre, df_chacras)
    df_status_compuertas = status_compuertas(df_kobo, df_chacras)

    # Descargas (CSV)
    csv_kobo = df_kobo.to_csv().encode('utf-8')
    csv_riego = df_riego.to_csv().encode('utf-8')

    hoy = datetime.datetime.today()

    c1, c2, _ = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Descargar riegos",
            csv_riego,
            f"status_riego_{hoy}.csv",
            "text/csv",
            key='download-riegos'
        )
    with c2:
        st.download_button(
            "⬇️ Descargar raw data",
            csv_kobo,
            f"raw_data_{hoy}.csv",
            "text/csv",
            key='download-raw-data'
        )

    # KPIs
    kpis = calcular_kpis(df_riego)
    kpis_dif = [f'{obtener_caudal_casa_piedra()[1]} m³/s respecto a ayer', '', '']
    kpi_names = ['Caudal Casa de Piedra', 'Ultimo registro', 'Riegos ejecutados']
    mostrar_kpis(kpis, kpi_names, kpis_dif)

    st.divider()

    # ----- Status compuertas -----
    st.header('Status compuertas')
    cant_abiertas = cantidad_compuertas_abiertas(df_status_compuertas)
    col_a, col_b = st.columns(2)
    col_a.metric("Compuertas abiertas", cant_abiertas)
    fig_status = mapa_status_compuertas(df_status_compuertas, sn_shp)
    st.plotly_chart(fig_status, use_container_width=True)

    st.divider()
    st.header('Mapas')

    # ----- Mapa: ciclos -----
    st.subheader('Cantidad de ciclos de riego ejecutados')
    ciclos = sorted(df_riego['ciclos'].dropna().unique().astype(int).tolist())
    tipos_ciclo = ["TODAS"] + ciclos
    seleccion_ciclo = st.selectbox('Seleccione la cantidad de ciclos', tipos_ciclo)
    fig_ciclos = mapa_ciclos(df_riego, sn_shp, seleccion_ciclo)
    st.plotly_chart(fig_ciclos, use_container_width=True)

    # ----- Mapa: última semana regada -----
    st.subheader('Semana del último riego ejecutado')

    df_last = construir_ultima_semana_por_chacra(df_riego_pre, df_chacras)

    # Semana "actual" (última entre las que regaron)
    df_valid = df_last[df_last['has_riego'] == 1]
    if not df_valid.empty:
        sem_act_monday = df_valid.loc[df_valid['last_semana_monday'].idxmax(), 'last_semana_monday']
        st.text(f"Semana actual (lunes): {sem_act_monday.strftime('%d/%m/%Y')}")
    else:
        sem_act_monday = None
        st.text("Semana actual (lunes): N/D")

    # Dropdown: "TODAS" + lunes únicos ordenados
    semanas_df = (
        df_valid[['last_semana_monday']]
        .dropna()
        .drop_duplicates()
        .sort_values('last_semana_monday')
    )
    opciones = ["TODAS"] + semanas_df['last_semana_monday'].dt.strftime('%d/%m/%Y').tolist()

    eleccion = st.selectbox('Seleccione lunes de semana', opciones)
    if eleccion == "TODAS":
        monday_sel = None
    else:
        monday_sel = pd.to_datetime(eleccion, format='%d/%m/%Y')

    fig_ultima = mapa_ultima_semana(df_last, sn_shp, monday_sel)
    st.plotly_chart(fig_ultima, use_container_width=True)

    # ----- Mapa: tiempo promedio por hectárea -----
    st.subheader('Tiempo de riego promedio por hectarea [Hs / Ha]')
    fig_tiempo = mapa_tiempo_promedio(df_riego, sn_shp)
    st.plotly_chart(fig_tiempo, use_container_width=True)

    # ----- Mapa: actividad -----
    st.subheader('Actividad por lote')
    fig_actividad = mapa_actividad(df_riego, sn_shp)
    st.plotly_chart(fig_actividad, use_container_width=True)

    st.divider()

    # ----- Gráfico: superficie semanal -----
    sup_semanal_df = construir_sup_semanal(df_riego_pre, df_chacras, unique_per_week=False)
    st.header('Superficie regada por semana')
    fig_sup = graficar_sup_semanal(sup_semanal_df)
    st.plotly_chart(fig_sup, use_container_width=True)

    st.divider()

    # ----- Gráfico: riegos por regador -----
    st.header('Cantidad de riegos por regador')
    fig_regadores = graficar_riegos_por_regador(df_riego_pre)
    st.plotly_chart(fig_regadores, use_container_width=True)


if __name__ == "__main__":
    run()