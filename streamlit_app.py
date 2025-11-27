# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
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
        st.markdown("""
            <div class="metric-card">
                <p class="metric-title">Total de estudiantes</p>
                <h2 class="metric-value">28</h2>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card">
                <p class="metric-title">Riesgo de Deserción Promedio</p>
                <h2 class="metric-value">Alto</h2>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <p class="metric-title">Rendimiento Promedio</p>
                <h2 class="metric-value">Previsto</h2>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="metric-card">
                <p class="metric-title">Bienestar Estudiantil Promedio</p>
                <h2 class="metric-value">Regular</h2>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <img src="data:image/png;base64,{guardian}" class="logo-guardian">
        """, unsafe_allow_html=True)


    res = supabase.table("estudiantes").select("*").execute()
    datos = res.data

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
            

            st.session_state.show_save = True

        if st.session_state.get("show_save", False):   
            if st.button("Guardar", type="primary", use_container_width=True):
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
    st.write("Hola abel")
