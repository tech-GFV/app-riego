import pandas as pd
from koboextractor import KoboExtractor
import datetime
import numpy as np
import geopandas as gpd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit.logger import get_logger
import requests
from lxml import html

LOGGER = get_logger(__name__)

# Preprocesamiento------------------------------------------------------------------------
def cargar_kobo(TOKEN):
  kobo = KoboExtractor(TOKEN, 'https://eu.kobotoolbox.org/api/v2')
  #   Cargar KOBOTOOLBOX
  form_id = 'aM693SUegTpTjVKobB7d2h'
  data = kobo.get_data(form_id, query=None, start=None, limit=None, submitted_after=None)
  df_kobo = pd.json_normalize(data['results'])
  #   Renombrar columna de ID
  df_kobo = df_kobo.rename(columns={"chacra":"ID_chacra"})
  #   Crear columna ID
  df_kobo['ID'] = df_kobo['ID_chacra'].str.strip()
  #   Elimino espacios en columna chacra
  df_kobo['ID_chacra'] = df_kobo['ID_chacra'].str.replace(' ', '')
  #   Eliminar columnas innecesarias
  df_kobo = df_kobo.drop(columns=['_id','formhub/uuid','start','__version__','meta/instanceID','_xform_id_string','_uuid','_attachments','_status','_geolocation','_submission_time','_tags','_notes'])
  #   Convertir timestamp a dateformat
  df_kobo['end'] = pd.to_datetime(df_kobo['end'])
  #   Eliminar registros anteriores al inicio de campaña 30/08/23
  df_kobo = df_kobo.loc[(df_kobo['end'] >= '2023-08-30')]
  return df_kobo

def cargar_chacras():
  df_chacras = pd.read_csv('base_chacras.csv', sep=';', encoding='utf-8')
  #   Renombrar columna de ID
  df_chacras = df_chacras.rename(columns={"ID_QR":"ID_chacra"})
  df_chacras['ID_chacra'] = df_chacras['ID_chacra'].str.replace(' ', '')
  df_chacras['ID_xls'] = df_chacras['ID_xls'].str.replace(' ', '')
  return df_chacras

def cargar_geometria():
  sn_shp = gpd.read_file('PARCELAS_.shp')
  #Renombrar columna de ID
  sn_shp = sn_shp.rename(columns={"ID_xls":"ID_chacra"})
  sn_shp['ID_chacra'] = sn_shp['ID_chacra'].str.replace(' ', '')
  sn_shp = sn_shp.to_crs("WGS84")
  return sn_shp

def crear_riegos(df):
    df_apertura = df[df['Acci_n'] == "apertura"]
    df_cierre = df[df['Acci_n'] == "cierre"]

    merged_df = pd.merge(df_apertura, df_cierre, on='ID_chacra', suffixes=('_ap', '_ci'))
    filtered_df = merged_df[
        (merged_df['end_ci'] > merged_df['end_ap']) &
        (merged_df['end_ci'] < merged_df['end_ap'] + pd.Timedelta(days=1))
    ]

    df_riego_aux = filtered_df[[
        'ID_ap', 'ID_chacra', 'end_ap', 'end_ci', '_submitted_by_ap', '_submitted_by_ci'
    ]].copy()

    df_riego_aux.columns = ['ID', 'ID_chacra', 'time_ap', 'time_ci', 'reg_ap', 'reg_ci']
    df_riego_aux['time_regado'] = df_riego_aux['time_ci'] - df_riego_aux['time_ap']

    return df_riego_aux

def unir_chacra_riego(df_riego_aux, df_chacra):
  #Agregar atributos de chacras a df_riego_aux
  df_riego = df_riego_aux.merge(df_chacra[['ID_chacra', 'ID_xls', 'SUPERFICIE', 'ACTIVIDAD', 'ID_CAMPAÑA']], on='ID_chacra', how='left')
  #Agrupar df agregando columnas de ciclos y t_riego_prom
  df_riego = df_riego.groupby('ID_chacra') \
        .agg(ciclos=('ID_chacra', 'count'), t_riego_prom=('time_regado', 'mean'), superficie=('SUPERFICIE', 'first'), actividad=('ACTIVIDAD', 'first'), time_ci=('time_ci', 'last'), ID=('ID', 'first'), ID_xls=('ID_xls', 'first'), ID_campaña=('ID_CAMPAÑA', 'first')) \
        .reset_index()
  #Agregar semana de riego
  df_riego['fecha_ult_ejec'] = df_riego.time_ci.dt.date
  df_riego['sem_ejec'] = df_riego.time_ci.dt.isocalendar().week.astype('int')
  #df_riego['text_sem'] = df_riego['ID'] + '<br>' + \
  #                      'Semana: ' + str(df_riego['sem_ejec'])
  df_riego['superficie'] = df_riego['superficie'].apply(lambda x: float(x.replace(',', '.').replace('#N/D', '0')) if x else 0)
  return df_riego

# Mapas----------------------------------------------------------------------------------
def mapa_status_compuertas(df_riego, sn_shp):
  fig_status = px.choropleth(
      df_riego,
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
  fig_status.update_layout(
    autosize=False,
    width=800,
    height=800)
  return fig_status

def mapa_sem_riego(df_riego, sn_shp, semana):
  df_riego_copia = df_riego.copy()
  if semana != "TODAS":
    df_riego_copia['sem_ejec'] = df_riego_copia['sem_ejec'].apply(lambda x: x if x == semana else 0)
  fig_sem = px.choropleth(
      df_riego_copia,
      geojson=sn_shp.set_index("ID_chacra").geometry,
      locations="ID_xls",
      color="sem_ejec",
      color_continuous_scale="Bluered_r",
      projection="mercator",
      basemap_visible=True,
      hover_data=['actividad'],
  )
  fig_sem.update_geos(fitbounds="geojson")
  fig_sem.update_layout(
    autosize=False,
    width=800,
    height=800)
  return fig_sem

def mapa_ciclos(df_riego, sn_shp, ciclos):
  df_riego_copia = df_riego.copy()
  if ciclos != "TODOS":
    df_riego_copia['ciclos'] = df_riego_copia['ciclos'].apply(lambda x: x if x == ciclos else 0)
  fig_ciclos = px.choropleth(
      df_riego_copia,
      geojson=sn_shp.set_index("ID_chacra").geometry,
      locations="ID_xls",
      color="ciclos",
      color_continuous_scale="Bluered_r",
      projection="mercator",
      basemap_visible=True,
      hover_data=['actividad'],
  )
  fig_ciclos.update_geos(fitbounds="geojson")
  fig_ciclos.update_layout(
    autosize=False,
    width=800,
    height=800)
  return fig_ciclos

def mapa_actividad(df_riego, sn_shp):
  fig_actividad = px.choropleth(
      df_riego,
      geojson=sn_shp.set_index("ID_chacra").geometry,
      locations="ID_xls",
      color="actividad",
      color_continuous_scale="RdBu",
      projection="mercator",
      basemap_visible=True,
      hover_data=['actividad'],
  )
  fig_actividad.update_geos(fitbounds="geojson")
  fig_actividad.update_layout(
    autosize=False,
    width=800,
    height=800)
  return fig_actividad

def mapa_tiempo_promedio(df_riego, sn_shp):
  df_aux = df_riego.copy()
  df_aux['t_riego_prom'] = df_aux['t_riego_prom'].dt.total_seconds() / 3600
  df_aux['t/sup'] = df_aux['t_riego_prom'] / df_aux['superficie']

  fig_actividad = px.choropleth(
      df_aux,
      geojson=sn_shp.set_index("ID_chacra").geometry,
      locations="ID_xls",
      color="t/sup",
      color_continuous_scale="Bluered",
      projection="mercator",
      basemap_visible=True,
      hover_data=['t_riego_prom', 'superficie', 'actividad'],
  )
  fig_actividad.update_geos(fitbounds="geojson")
  fig_actividad.update_layout(
    autosize=False,
    width=800,
    height=800)
  return fig_actividad

# Graficos-------------------------------------------------------------------------------------------------
def graficar_sup_semanal(df_riego):
  #Agrupar riegos por semana y mostrar superficie regada
  df_sup_semanal = df_riego.groupby('sem_ejec')\
        .agg(sup_regada=('superficie', 'sum'))\
       .reset_index()
  #Round superficie
  df_sup_semanal['sup_regada'] = df_sup_semanal['sup_regada'].round()
  #Graficar
  fig_sup_semanal = px.line(df_sup_semanal, x="sem_ejec", y="sup_regada", text="sup_regada", width=900, height=500)
  fig_sup_semanal.update_traces(textposition="top center")
  return fig_sup_semanal

def graficar_riegos_por_regador(df_riego_pre):
  #Crear df de regadores apertura
  df_reg_ap = df_riego_pre.groupby('reg_ap')\
          .agg(cant_ap=('reg_ap', 'count'))\
        .reset_index()
  #Crear df de regadores cierre
  df_reg_ci = df_riego_pre.groupby('reg_ci')\
          .agg(cant_ci=('reg_ci', 'count'))\
        .reset_index()
  #Merge de ambos df
  df_reg = df_reg_ap.merge(df_reg_ci, left_on='reg_ap', right_on='reg_ci')
  #Total de riegos por regador
  df_reg['total'] = df_reg['cant_ap'] + df_reg['cant_ci']
  #Ordenar de mayor a menor
  df_reg = df_reg.sort_values(by=['total'], ascending=False)
  #Graficar
  fig_riegos_por_regador = px.bar(df_reg, x='reg_ap', y=['cant_ap', 'cant_ci'], width=900, height=500)
  return fig_riegos_por_regador

# Funciones auxiliares-------------------------------------------------------------------------------
def cantidad_compuertas_abiertas(estados_df):
   cantidad_abiertas = estados_df[estados_df['estado'] == 'abierta'].shape[0]
   return cantidad_abiertas

def status_compuertas(df, df_chacra):
    # Filtrar por apertura y cierre
    df_apertura = df[df['Acci_n'] == "apertura"].copy()
    df_cierre = df[df['Acci_n'] == "cierre"].copy()

    # Renombrar columnas
    df_apertura.rename(columns={'end': 'fecha_hora_apertura'}, inplace=True)
    df_cierre.rename(columns={'end': 'fecha_hora_cierre'}, inplace=True)

    # Fusionar ambos DataFrames en uno solo
    df = pd.concat([df_apertura, df_cierre])

    # Agregar columna de fecha
    df['fecha_hora'] = df['fecha_hora_apertura'].fillna(df['fecha_hora_cierre'])

    # Ordenar por fecha
    df.sort_values(by='fecha_hora', inplace=True)

    # Crear una columna 'tipo' para distinguir entre aperturas y cierres
    df['tipo'] = df.groupby('ID').cumcount().astype(str)

    # Inicializar un diccionario para mantener el estado actual de cada compuerta
    estado_compuertas = {}

    # Iterar sobre el DataFrame para simular el estado de las compuertas
    for _, row in df.iterrows():
        compuerta_id = row['ID_chacra']
        accion = row['Acci_n']
        fecha_hora = row['fecha_hora']

        if compuerta_id not in estado_compuertas:
            estado_compuertas[compuerta_id] = {'estado': None, 'fecha_hora': None}

        if accion == 'apertura':
            estado_compuertas[compuerta_id]['estado'] = 'abierta'
            estado_compuertas[compuerta_id]['fecha_hora'] = fecha_hora
        elif accion == 'cierre':
            estado_compuertas[compuerta_id]['estado'] = 'cerrada'
            estado_compuertas[compuerta_id]['fecha_hora'] = fecha_hora

    # Crear DataFrame con estados de las compuertas
    estados_df = pd.DataFrame.from_dict(estado_compuertas, orient='index').reset_index()
    estados_df.columns = ['ID_chacra', 'estado', 'fecha_hora']

    # Merge con df de chacras
    estados_df = estados_df.merge(df_chacra[['ID_chacra', 'ID_xls', 'SUPERFICIE', 'ACTIVIDAD']], on='ID_chacra', how='left')

    # Calcular tiempo transcurrido en horas
    estados_df['fecha_hora'] = pd.to_datetime(estados_df['fecha_hora'])
    estados_df['Tiempo_estado'] = (pd.to_datetime('now', utc=estados_df['fecha_hora'].dt.tz) - estados_df['fecha_hora']).dt.total_seconds() / 3600
    estados_df['Tiempo_estado'] = estados_df['Tiempo_estado'].round(decimals=0)
    # Actualizar estado si tiempo transcurrido es mayor a 24 y estado es 'abierta'
    condicion = (estados_df['Tiempo_estado'] > 24) & (estados_df['estado'] == 'abierta')
    estados_df.loc[condicion, 'estado'] = 's/ cierre'

    return estados_df

def obtener_caudal_casa_piedra():
   # URL del sitio web
  url = 'https://www.coirco.gov.ar/'

  # Realizar la solicitud GET a la página
  response = requests.get(url)

  # Comprobar si la solicitud se realizó con éxito (código de estado 200)
  if response.status_code == 200:
      # Crear un objeto de tipo 'lxml' para analizar el contenido HTML
      tree = html.fromstring(response.content)
      
      # XPath del elemento que contiene el caudal erogado (reemplaza con tu XPath)
      xpath_caudal = '/html/body/div[2]/div[1]/div[3]/div/div/p[2]/span[2]/span[2]/text()'
      xpath_caudal_ayer = '/html/body/div[2]/div[1]/div[3]/div/div/p[2]/span[2]/text()'
      
      # Encontrar el elemento con el XPath proporcionado
      caudal_element = tree.xpath(xpath_caudal)
      caudal_ayer_element = tree.xpath(xpath_caudal_ayer)

      raw_caudal = str(caudal_element[1])
      raw_caudal_ayer = str(caudal_ayer_element[1])

      caudal = raw_caudal.strip()

      # Encontrar el índice del texto ' m³/s'
      indice_caudal = raw_caudal.find(' m³/s')
      indice_caudal_ayer = raw_caudal_ayer.find(' m³/s')

      # Extraer el número antes del texto
      caudal_num = raw_caudal[:indice_caudal].strip()
      caudal_ayer_num = raw_caudal_ayer[:indice_caudal_ayer].strip()

      dif_caudal = int(caudal_num) - int(caudal_ayer_num)

  return caudal, dif_caudal

def obtener_ultimo_registro(df):
   # Ordena el DataFrame por la columna de fecha en orden descendente
  df_riego_sort = df.sort_values('time_ci', ascending=False)

  # Obtiene el último registro después de ordenar
  ultimo_registro = df_riego_sort.head(1)

  # Imprime el último registro
  time_ultimo_registro = list(ultimo_registro['time_ci'])[0].strftime("%d/%m/%y %H:%M")

  return time_ultimo_registro

def calcular_kpis(df):
  caudal_casa_piedra = obtener_caudal_casa_piedra()[0]
  ultimo_registro = obtener_ultimo_registro(df)
  return [caudal_casa_piedra, ultimo_registro]

def mostrar_kpis(kpis, kpi_names, kpis_dif):
    st.header("Parametros")
    for i, (col, (kpi_name, kpi_value, kpi_dif)) in enumerate(zip(st.columns(3), zip(kpi_names, kpis, kpis_dif))):
        col.metric(label=kpi_name, value=kpi_value, delta=kpi_dif)


def run():
  st.set_page_config(
      page_title="APP RIEGO",
      page_icon="💧",
      layout="wide",
      initial_sidebar_state="collapsed",
  )

  st.title('🌱 Santa Nicolasa - Faro verde')
  st.header('💧 APP Riego', divider="green")

  KOBO_TOKEN = 'c7e3cb8f6ae27f4e35148c5e529e473491bfa373'
  df_kobo = cargar_kobo(KOBO_TOKEN)
  df_chacras = cargar_chacras()
  sn_shp = cargar_geometria()
  df_riego_pre = crear_riegos(df_kobo)
  df_riego = unir_chacra_riego(df_riego_pre, df_chacras)
  df_status_compuertas = status_compuertas(df_kobo, df_chacras)

  csv_kobo = df_kobo.to_csv().encode('utf-8')
  csv_riego = df_riego.to_csv().encode('utf-8')

  hoy = datetime.datetime.today()

  col1, col2, col3 = st.columns(3)
  with col1:
    st.download_button(
    "Descargar riegos",
    csv_riego,
    f"status_riego_{hoy}.csv",
    "text/csv",
    key='download-riegos'
    )
  with col2:
    st.download_button(
    "Descargar raw data",
    csv_kobo,
    f"raw_data_{hoy}.csv",
    "text/csv",
    key='download-raw-data'
    )

  kpis = calcular_kpis(df_riego)
  kpis_dif = [f'{obtener_caudal_casa_piedra()[1]} m³/s respecto a ayer', '', '']
  kpi_names = ['Caudal Casa de Piedra', 'Ultimo registro']
  mostrar_kpis(kpis, kpi_names, kpis_dif)

  st.divider()

  st.header('Status compuertas')
  cantidad_abiertas = cantidad_compuertas_abiertas(df_status_compuertas)
  col1, col2 = st.columns(2)
  col1.metric("Compuertas abiertas", cantidad_abiertas)
  grafico_status = mapa_status_compuertas(df_status_compuertas, sn_shp)
  st.plotly_chart(grafico_status, use_container_width=True)

  st.divider()
  st.header('Mapas')

  st.subheader('Cantidad de ciclos de riego ejecutados')
  tipos_ciclo = np.insert(df_riego['ciclos'].unique().astype(object), 0, "TODOS")
  seleccion_ciclo = st.selectbox('Seleccione la cantidad de ciclos', tipos_ciclo)
  grafico_ciclos = mapa_ciclos(df_riego, sn_shp, seleccion_ciclo)
  st.plotly_chart(grafico_ciclos, use_container_width=True)

  semana_actual = datetime.datetime.today().isocalendar().week
  st.subheader('Semana del ultimo riego ejecutado')
  st.text(f'Semana actual: {semana_actual}')
  tipos_semana = np.insert(df_riego['sem_ejec'].unique().astype(object), 0, "TODAS")
  seleccion_semana = st.selectbox('Seleccione una semana', tipos_semana)
  grafico_sem_riego = mapa_sem_riego(df_riego, sn_shp, seleccion_semana)
  st.plotly_chart(grafico_sem_riego, use_container_width=True)

  st.subheader('Tiempo de riego promedio por hectarea [Hs / Ha]')
  grafico_tiempo_promedio = mapa_tiempo_promedio(df_riego, sn_shp)
  st.plotly_chart(grafico_tiempo_promedio, use_container_width=True)

  st.subheader('Actividad por lote')
  grafico_actividad = mapa_actividad(df_riego, sn_shp)
  st.plotly_chart(grafico_actividad, use_container_width=True)

  st.divider()
  st.header('Superficie regada por semana')
  grafico_sup_semanal = graficar_sup_semanal(df_riego)
  st.plotly_chart(grafico_sup_semanal, use_container_width=True)

  st.divider()
  st.header('Cantidad de riegos por regador')
  grafico_riegos_por_regador = graficar_riegos_por_regador(df_riego_pre)
  st.plotly_chart(grafico_riegos_por_regador, use_container_width=True)


if __name__ == "__main__":
    run()
