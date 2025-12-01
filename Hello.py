"""
Aplicación Streamlit principal.
"""
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from datetime import timedelta

# Imports de nuestros módulos
from config.config import KOBO_TOKEN, KOBO_FORM_ID, PATH_SHAPEFILE, URL_COIRCO, PLANIFICACION_SPREADSHEET_URL
from data.loaders import cargar_datos_kobo, cargar_chacras, cargar_caudales_cdp, cargar_planificacion_riego
from data.processors import crear_riegos, agregar_metadatos_riego, unir_riego_chacras, crear_resumen_riegos, expandir_planificacion_con_chacras

# Configuración de página
st.set_page_config(
    page_title="APP de Riego",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_data(ttl=3600)
def cargar_datos_completos():
    """Carga todos los datos necesarios."""
    # Cargar datos de Kobo
    df_registros = cargar_datos_kobo(KOBO_TOKEN, KOBO_FORM_ID)

    # Cargar chacras
    df_chacras = cargar_chacras(PATH_SHAPEFILE)

    # Cargar caudales
    df_caudales = cargar_caudales_cdp(URL_COIRCO)

    # Cargar planificación de riego
    if PLANIFICACION_SPREADSHEET_URL:
        df_planificacion = cargar_planificacion_riego(PLANIFICACION_SPREADSHEET_URL)
        df_planificacion_expandida = expandir_planificacion_con_chacras(df_planificacion, df_chacras)
    else:
        df_planificacion = pd.DataFrame(columns=['semana_lunes', 'lote_albor'])
        df_planificacion_expandida = pd.DataFrame(columns=['semana_lunes', 'lote_albor', 'id_riego', 'id_simple', 'superficie_has', 'geometria'])

    # Procesar riegos
    df_riegos = unir_riego_chacras(df_registros, df_chacras)
    df_riegos = crear_riegos(df_riegos)
    df_riegos = agregar_metadatos_riego(df_riegos)

    # Crear resumen
    df_resumen = crear_resumen_riegos(df_riegos)

    return df_riegos, df_resumen, df_caudales, df_registros, df_chacras, df_planificacion, df_planificacion_expandida


def main():
    st.title("💧 APP de Riego - Santa Nicolasa")
    st.markdown("---")

    # Cargar datos
    with st.spinner("Cargando datos..."):
        df_riegos_completo, df_resumen_completo, df_caudales, df_registros, df_chacras, df_planificacion, df_planificacion_expandida = cargar_datos_completos()

    # Sidebar con filtros
    with st.sidebar:
        st.header("⚙️ Filtros Globales")

        # Botón para forzar actualización de datos
        if st.button("🔄 Actualizar Datos", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # Obtener campañas disponibles y ordenarlas
        campañas_disponibles = sorted(df_riegos_completo['campaña'].unique())

        # Multiselect para campañas
        filtro_campaña = st.multiselect(
            "Campaña(s)",
            options=campañas_disponibles,
            default=[],
            help="Deja vacío para ver todas las campañas"
        )
    
    # Filtrar datos según campaña seleccionada
    if filtro_campaña:
        df_riegos = df_riegos_completo[df_riegos_completo['campaña'].isin(filtro_campaña)]
        df_resumen = df_resumen_completo[df_resumen_completo['id_riego'].isin(df_riegos['id_riego'].unique())]
    else:
        df_riegos = df_riegos_completo
        df_resumen = df_resumen_completo

    # ========== TABS PARA ORGANIZAR CONTENIDO ==========
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Performance Semanal",
        "Mapas",
        "Análisis Histórico",
        "Desempeño del Equipo",
        "Caudales",
        "Lotes Albor"
    ])

    # ========== TAB 1: PERFORMANCE SEMANAL ==========
    with tab1:
        st.header("Performance Semanal de Riego")
        
        # Obtener semana actual
        fecha_hoy = datetime.datetime.now().date()
        semana_actual = (fecha_hoy - timedelta(days=fecha_hoy.weekday()))
        
        # Obtener todas las semanas disponibles
        semanas_disponibles = sorted(df_riegos['semana_lunes'].unique(), reverse=True)
        
        # Selector de semana
        col1, col2 = st.columns([4, 1])
        with col1:
            semana_seleccionada = st.selectbox(
                "Seleccionar Semana",
                options=semanas_disponibles,
                index=0 if semana_actual in semanas_disponibles else 0,
                format_func=lambda x: f"Semana del {x}" + (" (Actual)" if x == semana_actual else "")
            )
        
        with col2:
            if st.button("Semana Actual", use_container_width=True):
                st.rerun()
        
        # Filtrar riegos de la semana
        df_riegos_semana = df_riegos[df_riegos['semana_lunes'] == semana_seleccionada].copy()
        
        # Métricas de la semana
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Riegos", len(df_riegos_semana))
        
        with col2:
            hectareas_semana = df_riegos_semana['superficie_has'].sum() if len(df_riegos_semana) > 0 else 0
            st.metric("Hectáreas", f"{hectareas_semana:.1f} ha")
        
        with col3:
            st.metric("Chacras", df_riegos_semana['id_riego'].nunique())
        
        with col4:
            duracion_promedio_semana = df_riegos_semana['duracion_horas'].mean() if len(df_riegos_semana) > 0 else 0
            st.metric("Promedio", f"{duracion_promedio_semana:.1f} hs")
        
        # Mapa y gráfico lado a lado
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Mapa de Chacras Regadas")

            # Crear dataframe para el mapa
            df_mapa_semana = df_chacras.copy()

            # Obtener chacras planificadas para esta semana
            chacras_planificadas = set()
            if len(df_planificacion_expandida) > 0:
                df_plan_semana = df_planificacion_expandida[
                    df_planificacion_expandida['semana_lunes'] == semana_seleccionada
                ]
                chacras_planificadas = set(df_plan_semana['id_riego'].unique())

            # Obtener chacras regadas en la semana
            chacras_regadas = set(df_riegos_semana['id_riego'].unique()) if len(df_riegos_semana) > 0 else set()

            # Clasificar cada chacra según su estado
            def clasificar_chacra(id_riego):
                planificada = id_riego in chacras_planificadas
                regada = id_riego in chacras_regadas

                if planificada and regada:
                    return 'Planificada y Regada'
                elif planificada and not regada:
                    return 'Planificada sin Regar'
                elif not planificada and regada:
                    return 'Regada sin Planificar'
                else:
                    return 'No Planificada ni Regada'

            df_mapa_semana['estado'] = df_mapa_semana['id_riego'].apply(clasificar_chacra)

            # Definir colores
            color_map = {
                'Planificada y Regada': '#2ecc71',      # Verde
                'Planificada sin Regar': '#95a5a6',     # Gris
                'Regada sin Planificar': '#3498db',     # Azul
                'No Planificada ni Regada': '#ffffff'   # Blanco
            }

            fig_mapa_semana = px.choropleth(
                df_mapa_semana,
                geojson=df_mapa_semana.set_index("id_riego").geometry,
                locations="id_riego",
                color="estado",
                projection="mercator",
                labels={'estado': 'Estado'},
                hover_data={'ID_SIMPLE': True, 'Has': True, 'estado': True, 'id_riego': False},
                color_discrete_map=color_map,
                category_orders={'estado': ['Planificada y Regada', 'Planificada sin Regar', 'Regada sin Planificar', 'No Planificada ni Regada']}
            )

            fig_mapa_semana.update_geos(fitbounds="geojson", visible=False)
            fig_mapa_semana.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig_mapa_semana, use_container_width=True)

        with col2:
            st.subheader("Distribución por Día")

            if len(df_riegos_semana) > 0:
                df_riegos_semana['dia_semana'] = pd.to_datetime(df_riegos_semana['inicio']).dt.day_name()
                df_riegos_semana['dia_numero'] = pd.to_datetime(df_riegos_semana['inicio']).dt.dayofweek

                dias_espanol = {
                    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                }
                df_riegos_semana['dia_semana_es'] = df_riegos_semana['dia_semana'].map(dias_espanol)

                riegos_por_dia = df_riegos_semana.groupby(['dia_numero', 'dia_semana_es']).agg(
                    cantidad_riegos=('id_riego', 'count'),
                    hectareas=('superficie_has', 'sum')
                ).reset_index().sort_values('dia_numero')

                fig_riegos_dia = px.bar(
                    riegos_por_dia,
                    x='dia_semana_es',
                    y='hectareas',
                    labels={'dia_semana_es': '', 'hectareas': 'Hectáreas'},
                    color='hectareas',
                    color_continuous_scale='Blues'
                )

                fig_riegos_dia.update_layout(
                    showlegend=False,
                    height=500,
                    xaxis={'categoryorder': 'array', 'categoryarray': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']}
                )

                st.plotly_chart(fig_riegos_dia, use_container_width=True)
            else:
                st.info("No hay riegos registrados para esta semana")

        # Tabla detallada
        if len(df_riegos_semana) > 0:
            with st.expander("Ver detalle de riegos"):
                df_detalle = df_riegos_semana[[
                    'id_simple', 'usuario_apertura', 'usuario_cierre',
                    'inicio', 'fin', 'duracion_horas', 'superficie_has'
                ]].sort_values('inicio', ascending=False)

                st.dataframe(df_detalle, use_container_width=True, hide_index=True)

    # ========== TAB 2: MAPAS ==========
    with tab2:
        st.header("Mapas de Análisis")

        if len(df_riegos) > 0:
            fecha_min = pd.to_datetime(df_riegos['inicio']).min().date()
            fecha_max = pd.to_datetime(df_riegos['fin']).max().date()

            # ========== MAPA 1: ESTADO DE RIEGO (REGADO/NO REGADO) ==========
            st.subheader("1. Estado de Riego")

            col1, col2 = st.columns(2)
            with col1:
                fecha_desde_estado = st.date_input(
                    "Desde",
                    value=fecha_min,
                    min_value=fecha_min,
                    max_value=fecha_max,
                    key="fecha_desde_estado"
                )
            with col2:
                fecha_hasta_estado = st.date_input(
                    "Hasta",
                    value=fecha_max,
                    min_value=fecha_min,
                    max_value=fecha_max,
                    key="fecha_hasta_estado"
                )

            if fecha_desde_estado <= fecha_hasta_estado:
                # Filtrar riegos por rango de fechas
                df_riegos_estado = df_riegos.copy()
                df_riegos_estado['fecha_inicio_dt'] = pd.to_datetime(df_riegos_estado['inicio']).dt.date
                df_riegos_estado['fecha_fin_dt'] = pd.to_datetime(df_riegos_estado['fin']).dt.date

                df_riegos_estado = df_riegos_estado[
                    (df_riegos_estado['fecha_inicio_dt'] >= fecha_desde_estado) &
                    (df_riegos_estado['fecha_fin_dt'] <= fecha_hasta_estado)
                ]

                # Crear set de chacras regadas
                chacras_regadas = set(df_riegos_estado['id_riego'].unique()) if len(df_riegos_estado) > 0 else set()

                # Clasificar chacras
                df_mapa_estado = df_chacras.copy()
                df_mapa_estado['estado_riego'] = df_mapa_estado['id_riego'].apply(
                    lambda x: 'Regado' if x in chacras_regadas else 'No Regado'
                )

                # Crear mapa con colores discretos
                color_map = {
                    'Regado': '#2ecc71',        # Verde
                    'No Regado': '#ffffff'      # Blanco
                }

                fig_estado = px.choropleth(
                    df_mapa_estado,
                    geojson=df_mapa_estado.set_index("id_riego").geometry,
                    locations="id_riego",
                    color="estado_riego",
                    color_discrete_map=color_map,
                    projection="mercator",
                    labels={'estado_riego': 'Estado'},
                    hover_data={'ID_SIMPLE': True, 'Has': True, 'estado_riego': True, 'id_riego': False}
                )

                fig_estado.update_geos(fitbounds="geojson", visible=False)
                fig_estado.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

                st.plotly_chart(fig_estado, use_container_width=True)

                with st.expander("Ver tabla detallada"):
                    df_tabla_estado = df_mapa_estado[['ID_SIMPLE', 'Has', 'estado_riego']].copy()
                    df_tabla_estado = df_tabla_estado.sort_values('estado_riego')
                    df_tabla_estado = df_tabla_estado.rename(columns={
                        'ID_SIMPLE': 'ID Chacra',
                        'Has': 'Hectáreas',
                        'estado_riego': 'Estado'
                    })
                    st.dataframe(df_tabla_estado, use_container_width=True, hide_index=True)
            else:
                st.error("La fecha 'Desde' debe ser anterior o igual a la fecha 'Hasta'")

            st.markdown("---")

            # ========== MAPA 2: DÍAS DESDE ÚLTIMO RIEGO ==========
            st.subheader("2. Días desde Último Riego")

            col1, col2 = st.columns(2)
            with col1:
                fecha_desde_dias = st.date_input(
                    "Desde",
                    value=fecha_min,
                    min_value=fecha_min,
                    max_value=fecha_max,
                    key="fecha_desde_dias"
                )
            with col2:
                fecha_hasta_dias = st.date_input(
                    "Hasta",
                    value=fecha_max,
                    min_value=fecha_min,
                    max_value=fecha_max,
                    key="fecha_hasta_dias"
                )

            if fecha_desde_dias <= fecha_hasta_dias:
                # Filtrar riegos por rango de fechas
                df_riegos_dias = df_riegos.copy()
                df_riegos_dias['fecha_inicio_dt'] = pd.to_datetime(df_riegos_dias['inicio']).dt.date
                df_riegos_dias['fecha_fin_dt'] = pd.to_datetime(df_riegos_dias['fin']).dt.date

                df_riegos_dias = df_riegos_dias[
                    (df_riegos_dias['fecha_inicio_dt'] >= fecha_desde_dias) &
                    (df_riegos_dias['fecha_fin_dt'] <= fecha_hasta_dias)
                ]

                if len(df_riegos_dias) > 0:
                    df_resumen_filtrado = df_riegos_dias.groupby('id_riego', as_index=False).agg(
                        id_simple=('id_simple', 'first'),
                        superficie_has=('superficie_has', 'first'),
                        dia_ultimo_riego=('fin', 'last')
                    )
                    df_resumen_filtrado['dias_desde_ultimo_riego'] = (
                        pd.to_datetime('now') - df_resumen_filtrado['dia_ultimo_riego']
                    ).dt.days

                    df_mapa_dias = df_chacras.copy()
                    df_mapa_dias = df_mapa_dias.merge(
                        df_resumen_filtrado[['id_riego', 'dias_desde_ultimo_riego']],
                        on='id_riego',
                        how='left'
                    )
                    df_mapa_dias['dias_desde_ultimo_riego'] = df_mapa_dias['dias_desde_ultimo_riego'].fillna(999)

                    fig_dias = px.choropleth(
                        df_mapa_dias,
                        geojson=df_mapa_dias.set_index("id_riego").geometry,
                        locations="id_riego",
                        color="dias_desde_ultimo_riego",
                        color_continuous_scale="RdYlGn_r",
                        projection="mercator",
                        labels={'dias_desde_ultimo_riego': 'Días sin riego'},
                        hover_data={'ID_SIMPLE': True, 'Has': True, 'id_riego': False}
                    )

                    fig_dias.update_geos(fitbounds="geojson", visible=False)
                    fig_dias.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

                    st.plotly_chart(fig_dias, use_container_width=True)

                    with st.expander("Ver tabla detallada"):
                        # Usar df_mapa_dias para incluir todas las parcelas, incluyendo las con 999 días
                        df_tabla_dias = df_mapa_dias[['ID_SIMPLE', 'Has', 'dias_desde_ultimo_riego']].copy()
                        # Agregar columna de último riego desde df_resumen_filtrado
                        df_tabla_dias = df_tabla_dias.merge(
                            df_resumen_filtrado[['id_simple', 'dia_ultimo_riego']],
                            left_on='ID_SIMPLE',
                            right_on='id_simple',
                            how='left'
                        )
                        df_tabla_dias = df_tabla_dias.drop(columns=['id_simple'])
                        df_tabla_dias = df_tabla_dias.sort_values('dias_desde_ultimo_riego', ascending=False)
                        df_tabla_dias = df_tabla_dias.rename(columns={
                            'ID_SIMPLE': 'ID Chacra',
                            'Has': 'Hectáreas',
                            'dia_ultimo_riego': 'Último Riego',
                            'dias_desde_ultimo_riego': 'Días Transcurridos'
                        })
                        st.dataframe(df_tabla_dias, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay riegos en el rango de fechas seleccionado")
            else:
                st.error("La fecha 'Desde' debe ser anterior o igual a la fecha 'Hasta'")

            st.markdown("---")

            # ========== MAPA 3: TIEMPO DE RIEGO PROMEDIO (SIN FILTRO - ESTÁTICO) ==========
            st.subheader("3. Tiempo de Riego Promedio Histórico")

            # Usar todos los riegos históricos sin filtro
            df_duracion_historico = df_riegos.groupby('id_riego', as_index=False).agg(
                id_simple=('id_simple', 'first'),
                superficie_has=('superficie_has', 'first'),
                duracion_horas_prom=('duracion_horas', 'mean'),
                cantidad_riegos=('duracion_horas', 'count')
            )

            df_mapa_duracion = df_chacras.copy()
            df_mapa_duracion = df_mapa_duracion.merge(
                df_duracion_historico[['id_riego', 'duracion_horas_prom']],
                on='id_riego',
                how='left'
            )

            fig_duracion = px.choropleth(
                df_mapa_duracion,
                geojson=df_mapa_duracion.set_index("id_riego").geometry,
                locations="id_riego",
                color="duracion_horas_prom",
                color_continuous_scale="YlOrRd",
                projection="mercator",
                labels={'duracion_horas_prom': 'Horas promedio'},
                hover_data={'ID_SIMPLE': True, 'Has': True, 'duracion_horas_prom': ':.1f', 'id_riego': False}
            )

            fig_duracion.update_geos(fitbounds="geojson", visible=False)
            fig_duracion.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

            st.plotly_chart(fig_duracion, use_container_width=True)

            with st.expander("Ver tabla detallada"):
                df_tabla_duracion = df_duracion_historico[['id_simple', 'superficie_has', 'duracion_horas_prom', 'cantidad_riegos']].copy()
                df_tabla_duracion = df_tabla_duracion.sort_values('duracion_horas_prom', ascending=False)
                df_tabla_duracion['duracion_horas_prom'] = df_tabla_duracion['duracion_horas_prom'].round(1)
                df_tabla_duracion = df_tabla_duracion.rename(columns={
                    'id_simple': 'ID Chacra',
                    'superficie_has': 'Hectáreas',
                    'duracion_horas_prom': 'Duración Promedio (hs)',
                    'cantidad_riegos': 'Cantidad de Riegos'
                })
                st.dataframe(df_tabla_duracion, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ========== MAPA 4: CICLOS DE RIEGO ==========
            st.subheader("4. Ciclos de Riego")

            col1, col2 = st.columns(2)
            with col1:
                fecha_desde_ciclos = st.date_input(
                    "Desde",
                    value=fecha_min,
                    min_value=fecha_min,
                    max_value=fecha_max,
                    key="fecha_desde_ciclos"
                )
            with col2:
                fecha_hasta_ciclos = st.date_input(
                    "Hasta",
                    value=fecha_max,
                    min_value=fecha_min,
                    max_value=fecha_max,
                    key="fecha_hasta_ciclos"
                )

            if fecha_desde_ciclos <= fecha_hasta_ciclos:
                # Filtrar riegos por rango de fechas
                df_riegos_ciclos = df_riegos.copy()
                df_riegos_ciclos['fecha_inicio_dt'] = pd.to_datetime(df_riegos_ciclos['inicio']).dt.date
                df_riegos_ciclos['fecha_fin_dt'] = pd.to_datetime(df_riegos_ciclos['fin']).dt.date

                df_riegos_ciclos = df_riegos_ciclos[
                    (df_riegos_ciclos['fecha_inicio_dt'] >= fecha_desde_ciclos) &
                    (df_riegos_ciclos['fecha_fin_dt'] <= fecha_hasta_ciclos)
                ]

                if len(df_riegos_ciclos) > 0:
                    ciclos_por_chacra = df_riegos_ciclos.groupby('id_riego').size().reset_index(name='ciclos_riego')

                    df_mapa_ciclos = df_chacras.copy()
                    df_mapa_ciclos = df_mapa_ciclos.merge(
                        ciclos_por_chacra,
                        on='id_riego',
                        how='left'
                    )
                    df_mapa_ciclos['ciclos_riego'] = df_mapa_ciclos['ciclos_riego'].fillna(0).astype(int)

                    fig_ciclos = px.choropleth(
                        df_mapa_ciclos,
                        geojson=df_mapa_ciclos.set_index("id_riego").geometry,
                        locations="id_riego",
                        color="ciclos_riego",
                        color_continuous_scale="Blues",
                        projection="mercator",
                        labels={'ciclos_riego': 'Ciclos de riego'},
                        hover_data={'ID_SIMPLE': True, 'Has': True, 'ciclos_riego': True, 'id_riego': False}
                    )

                    fig_ciclos.update_geos(fitbounds="geojson", visible=False)
                    fig_ciclos.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

                    st.plotly_chart(fig_ciclos, use_container_width=True)

                    with st.expander("Ver tabla detallada"):
                        df_tabla_ciclos = df_mapa_ciclos[df_mapa_ciclos['ciclos_riego'] > 0][['ID_SIMPLE', 'Has', 'ciclos_riego']].copy()
                        df_tabla_ciclos = df_tabla_ciclos.sort_values('ciclos_riego', ascending=False)
                        df_tabla_ciclos = df_tabla_ciclos.rename(columns={
                            'ID_SIMPLE': 'ID Chacra',
                            'Has': 'Hectáreas',
                            'ciclos_riego': 'Ciclos de Riego'
                        })
                        st.dataframe(df_tabla_ciclos, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay riegos en el rango de fechas seleccionado")
            else:
                st.error("La fecha 'Desde' debe ser anterior o igual a la fecha 'Hasta'")
        else:
            st.info("No hay datos de riegos disponibles")

    # ========== TAB 3: ANÁLISIS HISTÓRICO ==========
    with tab3:
        st.header("Análisis Histórico")
        
        # Gráfico de hectáreas por semana
        st.subheader("Hectáreas Regadas por Semana")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            campañas_disponibles = sorted(df_riegos['campaña'].unique())
            campañas_seleccionadas = st.multiselect(
                "Comparar campañas",
                options=campañas_disponibles,
                default=campañas_disponibles[-2:] if len(campañas_disponibles) >= 2 else campañas_disponibles
            )
        
        if campañas_seleccionadas:
            df_filtrado = df_riegos[df_riegos['campaña'].isin(campañas_seleccionadas)].copy()
            
            df_filtrado['num_semana'] = pd.to_numeric(df_filtrado['num_semana'], errors='coerce')
            df_filtrado = df_filtrado.dropna(subset=['num_semana'])
            df_filtrado['num_semana'] = df_filtrado['num_semana'].astype(int)
            
            hectareas_semanales = df_filtrado.groupby(['num_semana', 'campaña']).agg(
                hectareas_regadas=pd.NamedAgg(column='superficie_has', aggfunc='sum')
            ).reset_index()
            
            fig_hectareas = px.line(
                hectareas_semanales,
                x='num_semana',
                y='hectareas_regadas',
                color='campaña',
                markers=True,
                labels={'num_semana': 'Semana del Año', 'hectareas_regadas': 'Hectáreas', 'campaña': 'Campaña'}
            )
            
            fig_hectareas.update_layout(
                hovermode='x unified',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_hectareas, use_container_width=True)
        else:
            st.warning("Selecciona al menos una campaña")
        
        # Historial de riegos
        st.subheader("Historial de Riegos")
        
        with st.expander("Filtros", expanded=False):
            col1, col2 = st.columns(2)
            filtro_id_simple_riegos = col1.text_input("Buscar por ID", key="filtro_id_simple_riegos")
            semana_seleccionada_hist = col2.multiselect(
                "Semana",
                options=sorted(df_riegos['semana_lunes'].unique(), reverse=True)
            )
        
        df_mostrar_riegos = df_riegos.copy().sort_values(by='inicio', ascending=False)
        
        if filtro_id_simple_riegos or semana_seleccionada_hist:
            if filtro_id_simple_riegos:
                df_mostrar_riegos = df_mostrar_riegos[
                    df_mostrar_riegos['id_simple'].str.contains(filtro_id_simple_riegos, case=False, na=False)
                ]
            if semana_seleccionada_hist:
                df_mostrar_riegos = df_mostrar_riegos[
                    df_mostrar_riegos['semana_lunes'].isin(semana_seleccionada_hist)
                ]
        
        st.dataframe(
            df_mostrar_riegos[[
                'id_simple', 'usuario_apertura', 'usuario_cierre',
                'inicio', 'fin', 'duracion_horas', 'superficie_has'
            ]].head(1000),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Mostrando {min(100, len(df_mostrar_riegos))} de {len(df_mostrar_riegos)} registros")

    # ========== TAB 4: DESEMPEÑO DEL EQUIPO ==========
    with tab4:
        st.header("Desempeño del Equipo")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Usuarios Activos", df_riegos['usuario_apertura'].nunique())
        with col2:
            st.metric("Total Riegos", len(df_riegos))
        with col3:
            promedio_por_usuario = len(df_riegos) / df_riegos['usuario_apertura'].nunique() if df_riegos['usuario_apertura'].nunique() > 0 else 0
            st.metric("Promedio/Usuario", f"{promedio_por_usuario:.1f}")
        
        st.subheader("Ranking de Regadores")
        
        campana_seleccionada = st.multiselect(
            "Seleccionar Campaña",
            options=sorted(df_riegos['campaña'].unique(), reverse=True)
        )
        
        df_riegos_campana = df_riegos[df_riegos['campaña'].isin(campana_seleccionada)] if campana_seleccionada else df_riegos.copy()
        
        ranking_usuarios = df_riegos_campana.groupby('usuario_apertura').agg(
            riegos_realizados=pd.NamedAgg(column='id_riego', aggfunc='count'),
            hectareas_regadas=pd.NamedAgg(column='superficie_has', aggfunc='sum')
        ).reset_index().sort_values(by='riegos_realizados', ascending=False)
        
        ranking_usuarios['hectareas_regadas'] = ranking_usuarios['hectareas_regadas'].round(1)
        ranking_usuarios = ranking_usuarios.rename(columns={
            'usuario_apertura': 'Usuario',
            'riegos_realizados': 'Riegos',
            'hectareas_regadas': 'Hectáreas'
        })
        
        st.dataframe(ranking_usuarios, use_container_width=True, hide_index=True)

    # ========== TAB 5: CAUDALES ==========
    with tab5:
        st.header("Caudales Río Colorado - COIRCO")
        
        fecha_hoy = datetime.datetime.now().date()
        df_caudales['fecha'] = pd.to_datetime(df_caudales['fecha']).dt.date
        caudal_actual = df_caudales.loc[df_caudales['fecha'] == fecha_hoy, 'caudal'].values
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if len(caudal_actual) > 0:
                st.metric("Caudal Actual", f"{caudal_actual[0]:.1f} m³/s")
            else:
                st.metric("Caudal Actual", "Sin datos")
        
        with col2:
            promedio_mes = df_caudales[df_caudales['fecha'].apply(lambda x: x.month) == fecha_hoy.month]['caudal'].mean()
            st.metric("Promedio del Mes", f"{promedio_mes:.1f} m³/s" if not pd.isna(promedio_mes) else "Sin datos")

        st.markdown("---")
        
        # Agregar año y día del año
        df_caudales['año'] = df_caudales['fecha'].apply(lambda x: x.year)
        df_caudales['fecha_dt'] = pd.to_datetime(df_caudales['fecha'])
        df_caudales['dia_año'] = df_caudales['fecha_dt'].dt.dayofyear

        st.subheader("Comparativa de Caudales por Año")
        
        # Multiselect para seleccionar años a comparar
        años_disponibles = sorted(df_caudales['año'].unique(), reverse=True)
        años_seleccionados = st.multiselect(
            "Seleccionar años para comparar",
            options=años_disponibles,
            default=años_disponibles[:2] if len(años_disponibles) >= 2 else años_disponibles,
            key="filtro_años_caudal"
        )
        
        if años_seleccionados:
            # Filtrar por años seleccionados
            df_caudales_filtrado = df_caudales[df_caudales['año'].isin(años_seleccionados)].copy()
            
            # Agrupar por día del año y año
            df_caudales_grafico = df_caudales_filtrado.groupby(['dia_año', 'año']).agg({
                'caudal': 'mean'
            }).reset_index()
            
            # Ordenar
            df_caudales_grafico = df_caudales_grafico.sort_values('dia_año')
            
            # Convertir año a string para mejor visualización
            df_caudales_grafico['año_str'] = df_caudales_grafico['año'].astype(str)
            
            # Crear gráfico de líneas
            fig_caudales = px.line(
                df_caudales_grafico,
                x='dia_año',
                y='caudal',
                color='año_str',
                labels={
                    'dia_año': 'Día del Año', 
                    'caudal': 'Caudal (m³/s)',
                    'año_str': 'Año'
                },
                markers=False
            )
            
            fig_caudales.update_layout(
                height=500,
                hovermode='x unified',
                legend=dict(
                    title='Año',
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            fig_caudales.update_traces(line=dict(width=2))
            
            st.plotly_chart(fig_caudales, use_container_width=True)
            
            # Tabla comparativa de estadísticas
            with st.expander("Ver estadísticas por año"):
                stats_años = df_caudales_filtrado.groupby('año').agg({
                    'caudal': ['mean', 'min', 'max', 'std']
                }).round(2)
                
                stats_años.columns = ['Promedio (m³/s)', 'Mínimo (m³/s)', 'Máximo (m³/s)', 'Desv. Est.']
                stats_años = stats_años.reset_index()
                stats_años = stats_años.rename(columns={'año': 'Año'})
                
                st.dataframe(stats_años, use_container_width=True, hide_index=True)
        else:
            st.warning("Selecciona al menos un año para visualizar")

    # ========== TAB 6: LOTES ALBOR ==========
    with tab6:
        st.header("Mapa de Lotes Albor")

        if len(df_chacras) > 0 and 'lote_albor' in df_chacras.columns:
            st.subheader("Distribución de Lotes Albor")

            # Verificar si hay datos en lote_albor
            df_chacras_albor = df_chacras.copy()

            # Crear figura choropleth
            fig_albor = px.choropleth(
                df_chacras_albor,
                geojson=df_chacras_albor.set_index("id_riego").geometry,
                locations="id_riego",
                color="lote_albor",
                projection="mercator",
                labels={'lote_albor': 'Lote Albor'},
                hover_data={'ID_SIMPLE': False, 'Has': False, 'lote_albor': True, 'id_riego': False},
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            fig_albor.update_geos(fitbounds="geojson", visible=False)
            fig_albor.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=700
            )

            st.plotly_chart(fig_albor, use_container_width=True)

        else:
            st.info("No hay datos de lotes Albor disponibles")


if __name__ == "__main__":
    main()
