# # streamlit_app.py
# import streamlit as st
# import pandas as pd
# import joblib
# from pathlib import Path

# # 1. Cargar modelo
# MODEL_PATH = Path("artefactos") / "modelo_pima.pkl"
# modelo = joblib.load(MODEL_PATH)

# st.title("🤖 Predicción de Diabetes ")
# st.write("(Pima Dataset)")

# # Tabs
# tab1, tab2, tab3 = st.tabs(["🧪 Predicción", "📊 Análisis del modelo", "📈 Gráficos interactivos"])
# # --- TAB 1: Entrada y predicción ---
# with tab1:
#     # 2. Ingreso de datos del paciente
#     st.subheader("Predicción de Diabetes")
#     st.write("Ingrese los valores clínicos para predecir si la paciente probablemente tiene diabetes.")
#     data = {
#         'npreg': st.slider("Número de embarazos", 0, 20, 2),
#         'glu':   st.slider("Nivel de glucosa (mg/dl)", 50, 200, 100),
#         'bp':    st.slider("Presión arterial (mmHg)", 40, 130, 70),
#         'skin':  st.slider("Espesor del pliegue cutáneo (mm)", 7, 100, 20),
#         'bmi':   st.slider("IMC", 10.0, 50.0, 25.0),
#         'ped':   st.slider("Pedigree de diabetes", 0.0, 2.5, 0.5),
#         'age':   st.slider("Edad (años)", 18, 90, 35)
#     }

#     # 3. Predicción
#     if st.button("Predecir"):
#         entrada = pd.DataFrame([data])
#         pred = modelo.predict(entrada)[0]
#         prob = modelo.predict_proba(entrada)[0][1]
#         resultado = "Diabética" if pred == 1 else "No diabética"
#         st.write(f"Resultado: **{resultado}**")
#         st.write(f"Probabilidad estimada: **{prob:.2f}**")

# # --- TAB 2: Análisis del modelo ---
# with tab2:
#     # 4. Importancia de cada variable en la predicción
#     # Coeficientes del modelo
#     coef_df = pd.DataFrame({
#         'Variable': modelo.feature_names_in_,
#         'Peso': modelo.coef_[0]
#     }).sort_values(by='Peso', key=abs, ascending=False)

#     st.subheader("Importancia de cada variable en la predicción")
#     st.bar_chart(coef_df.set_index("Variable"))

#     # 5. Correlación
#     import statsmodels.api as sm
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     df = sm.datasets.get_rdataset("Pima.tr", "MASS").data
#     df['type'] = df['type'].map({'Yes': 1, 'No': 0})

#     st.subheader("Correlación entre variables")
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# # --- TAB 3: Gráficos interactivos ---
# with tab3:
#     # 6. Gráficos interactivos
#     st.subheader("📈 Gráficos interactivos")

#     # Copiamos y preparamos los datos
#     df_plot = df.copy()
#     df_plot['type'] = df_plot['type'].map({1: 'Diabética', 0: 'No diabética'})

#     # Variables disponibles para graficar
#     variables = ['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age']

#     # Selección de variables para los ejes
#     col1, col2 = st.columns(2)
#     x_var = col1.selectbox("Elige variable para eje X", variables, index=0)
#     y_var = col2.selectbox("Elige variable para eje Y", variables, index=1)

#     # Crear gráfico interactivo
#     import plotly.express as px
#     fig_plotly = px.scatter(
#         df_plot,
#         x=x_var,
#         y=y_var,
#         color="type",
#         title=f"{x_var} vs {y_var} según diagnóstico",
#         labels={"type": "Diagnóstico", x_var: x_var, y_var: y_var},
#         hover_data=['npreg', 'bmi', 'glu', 'age'],
#         width=900,
#         height=600
#     )

#     # Mostrar gráfico
#     st.plotly_chart(fig_plotly)

#     st.subheader("Distribución de Glucosa")
#     st.plotly_chart(px.histogram(df, x="glu", color="type", barmode="overlay", nbins=40, labels={"type": "Diabetes (1=Sí)"}), use_container_width=True)




##################

'''
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. Cargar modelo
MODEL_PATH = Path("artefactos") / "modelo_pima.pkl"
modelo = joblib.load(MODEL_PATH)

st.title("🤖 Predicción de Diabetes (Pima Dataset)")

# 2. Ingreso de datos del paciente
st.write("Ingrese los valores clínicos para predecir si la paciente probablemente tiene diabetes.")
data = {
    'npreg': st.slider("Número de embarazos", 0, 20, 2),
    'glu':   st.slider("Nivel de glucosa (mg/dl)", 50, 200, 100),
    'bp':    st.slider("Presión arterial (mmHg)", 40, 130, 70),
    'skin':  st.slider("Espesor del pliegue cutáneo (mm)", 7, 100, 20),
    'bmi':   st.slider("IMC", 10.0, 50.0, 25.0),
    'ped':   st.slider("Pedigree de diabetes", 0.0, 2.5, 0.5),
    'age':   st.slider("Edad (años)", 18, 90, 35)
}

# 3. Predicción
if st.button("Predecir"):
    entrada = pd.DataFrame([data])
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]
    resultado = "Diabética" if pred == 1 else "No diabética"
    st.write(f"Resultado: **{resultado}**")
    st.write(f"Probabilidad estimada: **{prob:.2f}**")




# --- agregamos mas

# Coeficientes del modelo
import pandas as pd
coef_df = pd.DataFrame({
    'Variable': modelo.feature_names_in_,
    'Peso': modelo.coef_[0]
}).sort_values(by='Peso', key=abs, ascending=False)

st.subheader("Importancia de cada variable en la predicción")
st.bar_chart(coef_df.set_index("Variable"))

# EDA

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

df = sm.datasets.get_rdataset("Pima.tr", "MASS").data
df['type'] = df['type'].map({'Yes': 1, 'No': 0})

st.subheader("Correlación entre variables")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
'''


##
##MODEL_PATH = Path("artefactos") / "modelo_pima.pkl"
    ##modelo = joblib.load(MODEL_PATH)

    # st.title("🎓 Predicción de Estudiantes (Rendimiento, Deserción y Bienestar)")

    # # Tabs
    # tab1, tab2, tab3 = st.tabs(["📘 Predicción", "📊 Análisis", "⚙️ Artefactos"])

    # with tab1:
    #     st.subheader("Ingrese los datos del estudiante")

    #     # 🔢 Nota → categoría
    #     nota_num = st.slider("PPS actual (nota 0–20)", 0, 20, 12)

    #     # Convertir a categoría
    #     def clasificar_nota_pps(nota):
    #         if nota < 11:
    #             return "Inicio"
    #         elif nota < 14:
    #             return "En_Proceso"
    #         elif nota < 17:
    #             return "Previsto"
    #         else:
    #             return "Destacado"

    #     pps_cat = clasificar_nota_pps(nota_num)

    #     # 🟦 CATEGÓRICAS (con tus valores REALES)
    #     est = st.selectbox("Estrés", ["Bajo", "Moderado", "Alto", "Muy_Alto"])
    #     apoyo = st.selectbox("Apoyo social", ["Insuficiente", "Moderado", "Bueno", "Excelente"])
    #     horas_estudio = st.selectbox("Horas de estudio / semana", ["Insuficiente", "Limitado", "Adecuado", "Intensivo"])
    #     indproc = st.selectbox("Procrastinación", ["Minimo", "Ocasional", "Frecuente", "Constante"])
    #     horas_trabajo = st.selectbox("Horas de trabajo / semana", ["Ninguna", "Parcial_Baja", "Parcial_Alta", "Tiempo_Completo"])

    #     # 🔧 Unificar entrada final
    #     entrada = {
    #         "pps_actual_20": pps_cat,
    #         "estres": est,
    #         "apoyo_social": apoyo,
    #         "horas_de_estudio_semana": horas_estudio,
    #         "indice_procrastinacion": indproc,
    #         "horas_trabajo_semana": horas_trabajo
    #     }

    #     st.write("📥 Datos enviados al modelo:")
    #     st.json(entrada)

    #     # Botón de predicción
    #     if st.button(" Predecir"):
    #         pred_rend = predict_record(model_r, pre_r, schema_r, map_r, entrada)
    #         pred_des = predict_record(model_d, pre_d, schema_d, map_d, entrada)
    #         pred_bien = predict_record(model_b, pre_b, schema_b, map_b, entrada)

    #         st.subheader("📘 Resultados")
    #         st.write("🎓 **Rendimiento:**", pred_rend["pred_label"])
    #         st.json(pred_rend["probabilidades"])

    #         st.write("📉 **Deserción:**", pred_des["pred_label"])
    #         st.json(pred_des["probabilidades"])

    #         st.write("💚 **Bienestar Estudiantil:**", pred_bien["pred_label"])
    #         st.json(pred_bien["probabilidades"])