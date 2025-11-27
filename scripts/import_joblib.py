import joblib
import pandas as pd

print("Cargando scaler...")
scaler = joblib.load("artefactos/clustering/scaler.joblib")
print("Scaler cargado.")

print("Cargando dataset...")
df = pd.read_csv("data/dataset.csv")
print(f"Dataset cargado con {len(df)} filas y {len(df.columns)} columnas.")

# Obtén las columnas esperadas por el scaler
expected = scaler.feature_names_in_
actual = df.columns

print(f"Columnas esperadas por scaler: {len(expected)}")
print(f"Columnas en dataset: {len(actual)}")

# Muestra las columnas faltantes
missing = [col for col in expected if col not in actual]
print("Columnas faltantes:", missing)

if not missing:
    print("¡Todas las columnas están presentes!")
else:
    print("Faltan las siguientes columnas. Agregalas al dataset.csv:")
    for col in missing:
        print(f"  - {col}")
