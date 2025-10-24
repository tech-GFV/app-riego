"""
Funciones de procesamiento de datos de riego.
"""
import pandas as pd
from datetime import timedelta

def crear_riegos(df):
  """
  Crea un DataFrame con los eventos de riego.
  """
  df = df.sort_values(['id_riego', 'timestamp']).reset_index(drop=True)

  riegos = []

  for chacra, group in df.groupby('id_riego'):
      group = group.reset_index(drop=True)

      i = 0
      while i < len(group):
          if group.loc[i, 'accion'] == 'apertura':
              apertura_timestamp = group.loc[i, 'timestamp']
              usuario_apertura = group.loc[i, 'usuario']


              cierre_encontrado = False
              for j in range(i + 1, len(group)):
                  if group.loc[j, 'accion'] == 'cierre':
                      cierre_timestamp = group.loc[j, 'timestamp']
                      usuario_cierre = group.loc[j, 'usuario']
                      geometria = group.loc[j, 'geometry']
                      superficie_has = group.loc[j, 'Has']
                      id_simple = group.loc[j, 'ID_SIMPLE']
                      lote_albor = group.loc[j, 'lote_albor']
                      if cierre_timestamp - apertura_timestamp <= timedelta(hours=24):
                          riegos.append({
                              'id_riego': chacra,
                              'usuario_apertura': usuario_apertura,
                              'usuario_cierre': usuario_cierre,
                              'inicio': apertura_timestamp,
                              'fin': cierre_timestamp,
                              'duracion_horas': (cierre_timestamp - apertura_timestamp).total_seconds() / 3600,
                              'geometria': geometria,
                              'superficie_has': superficie_has,
                              'id_simple': id_simple,
                              'lote_albor': lote_albor,
                          })
                          cierre_encontrado = True
                          i = j + 1
                          break
                      else:
                          break

              if not cierre_encontrado:
                  i += 1
          else:
              i += 1

  df_riegos = pd.DataFrame(riegos)
  return df_riegos

def agregar_metadatos_riego(df):
  # Agregar lunes de la semana que se regó
  df['semana_lunes'] = (df['fin'] - pd.to_timedelta(df['fin'].dt.weekday, unit='d')).dt.date
  # Agregar numero de semana
  df['num_semana'] = df['fin'].dt.isocalendar().week
  # Agregar numero de mes
  df['mes'] = df['fin'].dt.month
  # Agregar año
  df['año'] = df['fin'].dt.year
  # Agregar campaña
  df['campaña'] = df.apply(
    lambda row: (
        f"{str(row['año']-1)[-2:]}-{str(row['año'])[-2:]}"
        if row['mes'] < 7
        else f"{str(row['año'])[-2:]}-{str(row['año']+1)[-2:]}"
    ),
    axis=1
  )
  # Fecha de inicio de campaña
  df['inicio_campaña'] = df.apply(
      lambda row: pd.Timestamp(year=row['año'] if row['mes'] >= 7 else row['año']-1, month=7, day=1),
      axis=1
  )
  # Calcular número de semana desde inicio de campaña
  df['num_semana_campaña'] = ((df['semana_lunes'].astype('datetime64[ns]') - df['inicio_campaña']).dt.days // 7) + 1
  return df

def unir_riego_chacras(df_riego, df_chacras):
  """
  Une df_riego con df_chacras usando 'id_riego',
  manteniendo todas las filas de df_riego.
  """
  df_riego_chacra = pd.merge(df_riego, df_chacras, on='id_riego', how='left')

  return df_riego_chacra

def crear_resumen_riegos(df):
  """
  Crea un DataFrame con el resumen de riegos por chacra.
  """
  # ordenar df por la columna fin ascendente
  df_sorted = df.sort_values('fin', ascending=True)

  df_resumen = (
        df_sorted.groupby('id_riego', as_index=False)
        .agg(
            id_simple=('id_simple', 'first'),
            geometria=('geometria', 'first'),
            superficie_has=('superficie_has', 'first'),
            ciclos=('id_riego', 'count'),
            duracion_horas_prom=('duracion_horas', 'mean'),
            semana_lunes_ultimo=('semana_lunes', 'last'),
            dia_ultimo_riego=('fin', 'last'),
        )
    )
  # agregar dias que pasaron desde el ultimo riego
  df_resumen['dias_desde_ultimo_riego'] = (pd.to_datetime('now') - df_resumen['dia_ultimo_riego']).dt.days

  return df_resumen

def expandir_planificacion_con_chacras(df_planificacion, df_chacras):
  """
  Expande la planificación de riego uniendo con chacras.

  Cada fila de planificación (semana + lote_albor) se expande a múltiples filas,
  una por cada chacra que pertenece a ese lote_albor.
  """
  # Validar que los dataframes tengan datos
  if df_planificacion.empty or df_chacras.empty:
      return pd.DataFrame(columns=[
          'semana_lunes', 'lote_albor', 'id_riego', 'id_simple',
          'superficie_has', 'geometria'
      ])

  # Unir planificación con chacras por lote_albor
  # Esto crea una fila por cada combinación de (semana, lote_albor, chacra)
  df_planificacion_expandida = pd.merge(
      df_planificacion,
      df_chacras[['id_riego', 'lote_albor', 'geometry', 'Has', 'ID_SIMPLE']],
      on='lote_albor',
      how='inner'  # Solo mantener lotes que existen en ambos dataframes
  )

  # Renombrar columnas para consistencia con otros dataframes
  df_planificacion_expandida = df_planificacion_expandida.rename(columns={
      'geometry': 'geometria',
      'Has': 'superficie_has',
      'ID_SIMPLE': 'id_simple'
  })

  # Ordenar por semana y lote
  df_planificacion_expandida = df_planificacion_expandida.sort_values(
      ['semana_lunes', 'lote_albor', 'id_riego']
  ).reset_index(drop=True)

  return df_planificacion_expandida