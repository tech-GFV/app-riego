import pandas as pd
from koboextractor import KoboExtractor
import datetime
import numpy as np
import geopandas as gpd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

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
  df_chacras = pd.read_csv('base_chacras.csv', sep=';', encoding='latin-1')
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
  df_riego['text_sem'] = df_riego['ID'] + '<br>' + \
                        'Semana: ' + str(df_riego['sem_ejec'])
  return df_riego

def mapa_sem_riego(df_riego, sn_shp):
  fig_sem = px.choropleth(
      df_riego,
      geojson=sn_shp.set_index("ID_chacra").geometry,
      locations="ID_xls",
      color="sem_ejec",
      color_continuous_scale="RdBu",
      projection="mercator",
      basemap_visible=True,
  )
  fig_sem.update_geos(fitbounds="geojson")
  fig_sem.update_layout(
    autosize=False,
    width=800,
    height=800)
  return fig_sem

def mapa_ciclos(df_riego, sn_shp):
  fig_ciclos = px.choropleth(
      df_riego,
      geojson=sn_shp.set_index("ID_chacra").geometry,
      locations="ID_xls",
      color="ciclos",
      color_continuous_scale="RdBu",
      projection="mercator",
      basemap_visible=True,
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
  )
  fig_actividad.update_geos(fitbounds="geojson")
  fig_actividad.update_layout(
    autosize=False,
    width=800,
    height=800)
  return fig_actividad

def graficar_sup_semanal(df_riego):
  #Convertir columna superficie a float
  df_riego['superficie'] = df_riego['superficie'].str.replace(',','.')
  df_riego['superficie'] = df_riego['superficie'].str.replace('#N/D','0')
  df_riego['superficie'] = df_riego['superficie'].str.replace('','0')
  df_riego['superficie'] = df_riego['superficie'].astype('float')
  #Agrupar riegos por semana y mostrar superficie regada
  df_sup_semanal = df_riego.groupby('sem_ejec')\
        .agg(sup_regada=('superficie', 'sum'))\
       .reset_index()
  #Round superficie
  df_sup_semanal['sup_regada'] = df_sup_semanal['sup_regada'].round()
  #Graficar
  fig_sup_semanal = px.line(df_sup_semanal, x="sem_ejec", y="sup_regada", text="sup_regada")
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
  fig_riegos_por_regador = px.bar(df_reg, x='reg_ap', y=['cant_ap', 'cant_ci'])
  return fig_riegos_por_regador


def run():
  st.set_page_config(
      page_title="APP RIEGO",
      page_icon="💧",
  )

  st.title('Santa Nicolasa - Faro verde')
  st.header('APP Riego', divider="grey")

  estado_carga_datos = st.text('Actualizando datos...')
  KOBO_TOKEN = 'c7e3cb8f6ae27f4e35148c5e529e473491bfa373'
  df_kobo = cargar_kobo(KOBO_TOKEN)
  df_chacras = cargar_chacras()
  sn_shp = cargar_geometria()
  df_riego_pre = crear_riegos(df_kobo)
  df_riego = unir_chacra_riego(df_riego_pre, df_chacras)
  estado_carga_datos.text('Carga completada correctamente')

  tipo_mapa = st.selectbox('Tipo de mapa', ['Ciclos', 'Semana', 'Actividades'])

  if tipo_mapa == 'Ciclos':
    st.subheader('Cantidad de ciclos de riego ejecutados')
    grafico_ciclos = mapa_ciclos(df_riego, sn_shp)
    st.plotly_chart(grafico_ciclos)

  if tipo_mapa == 'Semana':
    semana_actual = datetime.datetime.today().isocalendar().week
    st.subheader(f'Semana del ultimo riego ejecutado   SEM ACTUAL: {semana_actual}')
    grafico_sem_riego = mapa_sem_riego(df_riego, sn_shp)
    st.plotly_chart(grafico_sem_riego)

  if tipo_mapa == 'Actividades':
    st.subheader('Actividad por lote')
    grafico_actividad = mapa_actividad(df_riego, sn_shp)
    st.plotly_chart(grafico_actividad)

  st.subheader('Superficie regada por semana')
  grafico_sup_semanal = graficar_sup_semanal(df_riego)
  st.plotly_chart(grafico_sup_semanal)

  st.subheader('Cantidad de riegos por regador')
  grafico_riegos_por_regador = graficar_riegos_por_regador(df_riego_pre)
  st.plotly_chart(grafico_riegos_por_regador)


if __name__ == "__main__":
    run()
