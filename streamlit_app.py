# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import clustering_analysis as ca
from pathlib import Path
from loader_modelos import load_bundle, predict_record
import time
import base64
from supabase_client import supabase

def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo64 = load_image_base64("img/logoupeu.png")
guardian = load_image_base64("img/logoguardian.png")
icono_usuario = load_image_base64("img/usr.jpg")


def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles/style.css")

def nav_button(label, key):
    active = "nav-active" if st.session_state.page == key else ""
    if st.button(label, key=key):
        st.session_state.page = key
        st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "rna"

colB, colC = st.columns(2)

with colB: nav_button("RNA: Guardian", "rna")
with colC: nav_button("Clustering: Edu-Insight 360", "cluster")

res = supabase.table("estudiantes").select("*").execute()
datos = res.data
total_estudiantes = len(datos)
def extraer_categoria(valor):
    if not valor:
        return "-"
    return valor.split(" (")[0]
riesgos = [extraer_categoria(fila["riesgo_des"]) for fila in datos]
rendimientos = [extraer_categoria(fila["rendimiento"]) for fila in datos]
bienestares = [extraer_categoria(fila["bienestar_est"]) for fila in datos]
map_des = {
    "Bajo": 1,
    "Medio": 2,
    "Alto": 3,
    "Muy_Alto": 4
}

map_rend = {
    "Inicio": 1,
    "En_Proceso": 2,
    "Previsto": 3,
    "Destacado": 4
}

map_bien = {
    "Crítico": 1,
    "Regular": 2,
    "Bueno": 3,
    "Excelente": 4
}

def promedio_categoria(valores, mapa):
    nums = [mapa[v] for v in valores if v in mapa]
    if not nums:
        return "-"
    prom = sum(nums) / len(nums)
    # redondear al valor más cercano
    valor_red = round(prom)
    # obtener categoria por valor
    for cat, num in mapa.items():
        if num == valor_red:
            return cat
    return "-"
riesgo_prom = promedio_categoria(riesgos, map_des)
rend_prom = promedio_categoria(rendimientos, map_rend)
bien_prom = promedio_categoria(bienestares, map_bien)
#RNAAAA

if st.session_state.page == "rna":
    st.markdown(f"""
    <div class="custom-header">
        <img src="data:image/png;base64,{logo64}" class="logo">
        <input class="search-bar" placeholder="Buscar Guardian">
        <div class="header-icons">
            <span class="bell">🔔</span>
            <img src="data:image/jpeg;base64,{icono_usuario}" class="user-img">
            <span class="menu">☰</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-title">Total de estudiantes</p>
                <h2 class="metric-value">{total_estudiantes}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-title">Riesgo de Deserción Promedio</p>
                <h2 class="metric-value">{riesgo_prom}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-title">Rendimiento Promedio</p>
                <h2 class="metric-value">{rend_prom}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-title">Bienestar Estudiantil Promedio</p>
                <h2 class="metric-value">{bien_prom}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <img src="data:image/png;base64,{guardian}" class="logo-guardian">
        """, unsafe_allow_html=True)




    formateado = []
    for fila in datos:
        formateado.append([
            fila["codigo_est"],
            fila["nombre"],
            fila["carrera"],
            fila.get("riesgo_des", "-") or "-",
            fila.get("rendimiento", "-") or "-",
            fila.get("bienestar_est", "-") or "-"
        ])


    if "tabla_data" not in st.session_state:
        st.session_state.tabla_data = formateado

    html = """
    <div class="custom-table">
    <h3 class="table-title"
        style="
            color:#002D62;
            font-size:24px;
            font-weight:700;
        "
    >
        Tabla predictiva de estudiantes
    </h3>
    <table>
    <thead>
    <tr>
    <th>Código</th>
    <th>Nombre</th>
    <th>Carrera</th>
    <th>Riesgo de Deserción</th>
    <th>Rendimiento</th>
    <th>Bienestar Estudiantil</th>
    <th>Opciones</th>
    </tr>
    </thead>
    <tbody>
    """

    for i, (codigo, nombre, carrera, des, rend, bien) in enumerate(st.session_state.tabla_data):
        html += f"""
    <tr>
    <td>{codigo}</td>
    <td>{nombre}</td>
    <td>{carrera}</td>
    <td>{des}</td>
    <td>{rend}</td>
    <td>{bien}</td>
    <td><form action="" method="get" style="margin:0;"><input type="hidden" name="row" value="{i}"><button class="action-btn" type="submit">Acciones</button></form></td>
    </tr>
    """

    html += """
    </tbody>
    </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


    @st.dialog("Perfil del estudiante")
    def abrir_modal(codigo, nombre, carrera, des, rend, bien):

        st.markdown(f"""
        <h1 class='modal-title'>{nombre}</h1>
        <p class='subtitle'><b>Código:</b> {codigo}</p>
        <p class='subtitle'><b>Carrera:</b> {carrera}</p>
        """, unsafe_allow_html=True)

        st.markdown("<p class='modal-subtitle'>Datos importantes</p>", unsafe_allow_html=True)

        nota_num = st.slider("PPS actual (0–20)", 0, 20, 12)

        def clasificar_nota_pps(nota):
            if nota < 11:
                return "Inicio"
            elif nota < 14:
                return "En_Proceso"
            elif nota < 17:
                return "Previsto"
            else:
                return "Destacado"

        pps_cat = clasificar_nota_pps(nota_num)

        col1, col2, col3 = st.columns(3)
        with col1:
            est = st.selectbox("Nivel de estrés", ["Bajo", "Moderado", "Alto", "Muy_Alto"])
            apoyo = st.selectbox("Soporte social", ["Insuficiente", "Moderado", "Bueno", "Excelente"])
        with col2:
            horas_estudio = st.selectbox("Horas de estudio", ["Insuficiente", "Limitado", "Adecuado", "Intensivo"])
            procrast = st.selectbox("Procrastinación", ["Minimo", "Ocasional", "Frecuente", "Constante"])
        with col3:
            horas_trabajo = st.selectbox("Horas trabajo/semana", ["Ninguna", "Parcial_Baja", "Parcial_Alta", "Tiempo_Completo"])

        entrada = {
            "pps_actual_20": pps_cat,
            "estres": est,
            "apoyo_social": apoyo,
            "horas_de_estudio_semana": horas_estudio,
            "indice_procrastinacion": procrast,
            "horas_trabajo_semana": horas_trabajo
        }

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.button("Predecir", type="primary", use_container_width=True):
            pred_r = predict_record(model_r, pre_r, schema_r, map_r, entrada)
            pred_d = predict_record(model_d, pre_d, schema_d, map_d, entrada)
            pred_b = predict_record(model_b, pre_b, schema_b, map_b, entrada)

            st.markdown("<p class='modal-subtitle'>Resultados</p>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                label_r = pred_r["pred_label"]
                prob_r = pred_r["probabilidades"][label_r]

                st.markdown(
                    f"""
                    <div class='result-card'>
                        <div class='result-title'>Prob. Rendimiento</div>
                        <div class='result-value'>{label_r}</div>
                        <div class='result-value'>({prob_r:.2%})</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                label_d = pred_d["pred_label"]
                prob_d = pred_d["probabilidades"][label_d]

                st.markdown(
                    f"""
                    <div class='result-card'>
                        <div class='result-title'>Prob. Desercion</div>
                        <div class='result-value'>{label_d}</div>
                        <div class='result-value'>({prob_d:.2%})</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:
                label_b = pred_b["pred_label"]
                prob_b = pred_b["probabilidades"][label_b]

                st.markdown(
                    f"""
                    <div class='result-card'>
                        <div class='result-title'>Prob. Bienestar</div>
                        <div class='result-value'>{label_b}</div>
                        <div class='result-value'>({prob_b:.2%})</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            porc_r = f"{prob_r*100:.1f}%"
            porc_d = f"{prob_d*100:.1f}%"
            porc_b = f"{prob_b*100:.1f}%"
            st.session_state.tabla_data[i][3] = f'{pred_d["pred_label"]} ({porc_d})'   # Deserción
            st.session_state.tabla_data[i][4] = f'{pred_r["pred_label"]} ({porc_r})'   # Rendimiento
            st.session_state.tabla_data[i][5] = f'{pred_b["pred_label"]} ({porc_b})'   # Bienestar

            st.session_state.show_save = True

        if st.session_state.get("show_save", False):
            if st.button("Guardar", type="primary", use_container_width=True):
                pred_r = predict_record(model_r, pre_r, schema_r, map_r, entrada)
                pred_d = predict_record(model_d, pre_d, schema_d, map_d, entrada)
                pred_b = predict_record(model_b, pre_b, schema_b, map_b, entrada)
                label_r = pred_r["pred_label"]
                prob_r = pred_r["probabilidades"][label_r]
                label_d = pred_d["pred_label"]
                prob_d = pred_d["probabilidades"][label_d]
                label_b = pred_b["pred_label"]
                prob_b = pred_b["probabilidades"][label_b]
                porc_r = f"{prob_r*100:.1f}%"
                porc_d = f"{prob_d*100:.1f}%"
                porc_b = f"{prob_b*100:.1f}%"
                nuevo_des = f'{pred_d["pred_label"]} ({porc_d})'
                nuevo_rend = f'{pred_r["pred_label"]} ({porc_r})'
                nuevo_bien = f'{pred_b["pred_label"]} ({porc_b})'

                supabase.table("estudiantes").update({
                    "riesgo_des": nuevo_des,
                    "rendimiento": nuevo_rend,
                    "bienestar_est": nuevo_bien
                }).eq("codigo_est", codigo).execute()

                st.session_state.show_results = True
                st.session_state.show_save = True
                st.session_state.modal_closed = True
                time.sleep(2)
                st.rerun()


    if "row" in st.query_params:
        i = int(st.query_params["row"])
        codigo, nombre, carrera, des, rend, bien = st.session_state.tabla_data[i]

        abrir_modal(codigo, nombre, carrera, des, rend, bien)

        st.query_params.clear()




    #1 === Cargar los 3 modelos ===
    model_r, pre_r, schema_r, map_r = load_bundle("artefactos/modelo_rendimiento/v1")
    model_d, pre_d, schema_d, map_d = load_bundle("artefactos/modelo_desercion/v1")
    model_b, pre_b, schema_b, map_b = load_bundle("artefactos/modelo_bienestar/v1")




#Cluster

if st.session_state.page == "cluster":
    # Cargar dataset clusterizado si existe
    clustered_path = Path("data/dataset_clustered.csv")
    if clustered_path.exists():
        df = pd.read_csv(clustered_path)
    else:
        st.warning("No se encontró 'data/dataset_clustered.csv'. Ejecuta el análisis de clustering primero.")
        st.stop()

    # Cargar artefactos (si existen)
    arte_dir = Path("artefactos/clustering")
    kmeans, scaler, pca = None, None, None
    try:
        kmeans = joblib.load(arte_dir / "kmeans_model.joblib")
        scaler = joblib.load(arte_dir / "scaler.joblib")
        pca = joblib.load(arte_dir / "pca_model.joblib")
    except Exception:
        # artefactos no disponibles: seguir con df
        pass

    st.markdown("""
    <div class="custom-header">
        <h2 style='color:#002D62;margin:0'>Clustering — Edu‑Insight 360</h2>
    </div>
    """, unsafe_allow_html=True)

    # Métricas superiores
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Total estudiantes")
        st.metric("Total", f"{len(df)}")
    with col2:
        largest = df['cluster'].value_counts().idxmax() if 'cluster' in df.columns else None
        st.subheader("Grupo más grande")
        st.metric("Cluster", f"{int(largest)+1}" if largest is not None else "-")
    with col3:
        st.subheader("Horas estudio (prom)")
        st.metric("Horas", f"{df['study_hours_wk'].mean():.1f}")
    with col4:
        st.subheader("Estrés (prom)")
        st.metric("Estrés", f"{df['stress'].mean():.1f}")

    # PCA scatter
    if 'cluster' in df.columns and pca is not None:
        pca_coords = pca.transform(scaler.transform(df[[c for c in df.columns if c in ['study_hours_wk','sleep_hours','lms_activity_rate','attendance_rate','assignments_on_time_rate','procrastination_index','self_efficacy','stress']]])) if scaler is not None else None
    else:
        pca_coords = None

    if pca_coords is None:
        # Fallback: try to load pca coords from columns if present
        if {'pca_0','pca_1'}.issubset(set(df.columns)):
            df['pca_0'] = df['pca_0']
            df['pca_1'] = df['pca_1']
        else:
            # compute a simple PCA on selected numeric cols
            numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in ['student_id','cluster'] if c in df.columns], errors='ignore')
            from sklearn.decomposition import PCA as skPCA
            scaler_tmp = None
            try:
                from sklearn.preprocessing import StandardScaler as _SS
                scaler_tmp = _SS()
                numeric_s = scaler_tmp.fit_transform(numeric.fillna(numeric.mean()))
            except Exception:
                numeric_s = numeric.fillna(0).values
            p = skPCA(n_components=2)
            coords = p.fit_transform(numeric_s)
            df['pca_0'] = coords[:,0]
            df['pca_1'] = coords[:,1]

    fig = px.scatter(df, x='pca_0', y='pca_1', color=df['cluster'].astype(str) if 'cluster' in df.columns else None,
                     hover_data=['student_id','major'] if 'student_id' in df.columns else None,
                     title='PCA 2D del dataset por cluster')
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Radar: comparar medias de variables seleccionadas por cluster
    radar_features = ['study_hours_wk','stress','attendance_rate','procrastination_index','social_support','self_efficacy']
    if 'cluster' in df.columns:
        cluster_means = df.groupby('cluster')[radar_features].mean()
        # normalizar entre 0 y 1 por característica para radar
        norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
        # permitir seleccionar clusters a mostrar
        clusters_available = sorted(cluster_means.index.tolist())
        selected = st.multiselect('Seleccionar clusters para comparar', options=[int(c) for c in clusters_available], default=[int(c) for c in clusters_available])
        if selected:
            radar_df = norm.loc[selected]
            # preparar para plotly (long format)
            radar_long = radar_df.reset_index().melt(id_vars='cluster', value_vars=radar_features, var_name='feature', value_name='value')
            fig_radar = px.line_polar(radar_long, r='value', theta='feature', color='cluster', line_close=True, title='Comparación por cluster (normalizado)')
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)

    # Tabla interactiva
    st.subheader('Muestra de estudiantes')
    table_cols = ['student_id','major','cluster','study_hours_wk','stress']
    available_cols = [c for c in table_cols if c in df.columns]
    st.dataframe(df[available_cols].head(200))

    # Permitir seleccionar un estudiante para ver detalles
    if 'student_id' in df.columns:
        sid = st.selectbox('Ver perfil del estudiante (ID)', options=[None] + df['student_id'].astype(str).tolist())
        if sid:
            student = df[df['student_id'].astype(str) == sid].iloc[0]
            st.markdown(f"**ID:** {student.get('student_id')}  \n                         **Carrera:** {student.get('major')}  \n                         **Cluster:** {int(student.get('cluster'))+1 if 'cluster' in student.index else '-'}")
            cols = st.columns(3)
            with cols[0]:
                st.metric('Horas estudio', f"{student.get('study_hours_wk'):.1f}")
            with cols[1]:
                st.metric('Estrés', f"{student.get('stress'):.1f}")
            with cols[2]:
                st.metric('Asistencia', f"{student.get('attendance_rate'):.1f}")

    st.markdown("---")
    st.markdown("Si necesitas que ajuste visuales al diseño de Figma (colores, layout o texto), dime qué partes quieres priorizar.")

    # Botón para recalcular clustering (se ejecuta dentro del contenedor)
    if st.button("Recalcular clustering", type="primary"):
        with st.spinner("Recalculando clustering — esto puede tardar unos segundos..."):
            try:
                X, df_new, features = ca.load_and_preprocess_data("data/dataset.csv")
                kmeans, scaler, labels, Xs = ca.perform_clustering(X, n_clusters=4)
                pca, coords = ca.apply_pca(Xs, n_components=2)
                stats, df_clustered = ca.get_cluster_statistics(df_new, labels)
                ca.save_clustering_artifacts(kmeans, scaler, pca)
                df_clustered.to_csv("data/dataset_clustered.csv", index=False)
                st.success("Recalculo completado — artefactos y dataset actualizados.")
                # recargar página para mostrar cambios
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error al recalcular clustering: {e}")
