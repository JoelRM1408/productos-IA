
# Model Card — modelo_bienestar

**Versión:** v1  
**Framework:** TensorFlow/Keras 2.16.2  
**Python:** 3.12.7

## 🧠 Modelo
Red neuronal multicapa (RNA) para clasificar el **Bienestar Estudiantil**.
Clases: ['Critico', 'Regular', 'Excelente']

## 📊 Datos
Total filas: 10000  
Train: 8000 — Test: 2000

Variables de entrada:
['estres', 'apoyo_social']

## 📈 Resultados en Test
Accuracy: 0.590  
Balanced Accuracy: 0.531

Matriz de confusión:
[[205, 107, 105], [90, 177, 316], [54, 149, 797]]

## ⚙️ Preprocesamiento
- OneHotEncoder(handle_unknown='ignore')
- ColumnTransformer

## 📁 Artefactos exportados
- model_bien.keras
- preprocess_bien.joblib
- input_schema.json
- label_map.json
- decision_policy.json
- sample_inputs.json
