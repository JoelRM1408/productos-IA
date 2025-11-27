
# Model Card — modelo_desercion

**Versión:** v1  
**Framework:** TensorFlow/Keras 2.16.2  
**Python:** 3.12.7

## 🧠 Modelo
Red neuronal multicapa (RNA) para clasificar el **riesgo de deserción**.
Clases: ['Bajo', 'Medio', 'Alto'].

## 📊 Datos
Total filas: 10000  
Train: 8000 — Test: 2000

Variables de entrada:
['horas_trabajo_semana', 'estres', 'apoyo_social']

## 📈 Resultados en Test
Accuracy: 0.647  
Balanced Accuracy: 0.549

Matriz de confusión:
[[966, 117, 41], [265, 147, 99], [89, 94, 182]]

## ⚙️ Preprocesamiento
- OneHotEncoder(handle_unknown='ignore')
- ColumnTransformer

## 📁 Artefactos exportados
- model_des.keras
- preprocess_des.joblib
- input_schema.json
- label_map.json
- decision_policy.json
- sample_inputs.json
