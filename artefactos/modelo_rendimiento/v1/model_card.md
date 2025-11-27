
# Model Card — modelo_rendimiento

**Versión:** v1  
**Framework:** TensorFlow/Keras 2.16.2  
**Python:** 3.12.7

## 🧠 Modelo
Red neuronal multicapa (RNA) para clasificar el rendimiento estudiantil:
- Entrada: OneHotEncoder de 3 variables categóricas
- Arquitectura:
  - Dense(16, relu)
  - Dense(8, relu)
  - Dense(4, softmax)

## 📊 Datos
Total filas: 10000  
Train: 8000 — Test: 2000  
Variables de entrada:
['pps_actual_20', 'horas_de_estudio_semana', 'indice_procrastinacion']

Variable objetivo: `rendimiento`  
Clases: ['Destacado', 'Previsto', 'En_Proceso', 'Inicio']

## 📈 Resultados en Test
- Accuracy = 0.535
- Balanced Accuracy = 0.489

Matriz de confusión:
[[569, 62, 74, 10], [119, 92, 119, 70], [94, 57, 161, 138], [27, 16, 143, 249]]

## ⚙️ Preprocesamiento
- OneHotEncoder(handle_unknown='ignore')
- ColumnTransformer

## 📁 Artefactos exportados
- model_rend.keras  
- preprocess_rend.joblib  
- input_schema.json  
- label_map.json  
- decision_policy.json  
- sample_inputs.json  
