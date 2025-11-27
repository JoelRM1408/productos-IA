# clustering_analysis.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


def load_and_preprocess_data(csv_path):
    """Carga y preprocesa el dataset"""
    df = pd.read_csv(csv_path)

    # Seleccionar características relevantes para clustering
    features_numeric = [
        'study_hours_wk', 'sleep_hours', 'lms_activity_rate', 'attendance_rate',
        'assignments_on_time_rate', 'procrastination_index', 'self_efficacy',
        'stress', 'internet_quality', 'part_time_job_hours', 'commute_min',
        'social_support', 'screen_time_nonstudy_hr', 'extracurricular_hours',
        'habit_consistency', 'age', 'credits_enrolled', 'year_of_study',
        'lms_quiz_rate', 'lms_forum_rate', 'lms_resource_rate', 'lms_submission_rate',
        'attendance_lecture', 'attendance_lab'
    ]

    # Obtener columnas de calificaciones
    grade_cols = [col for col in df.columns if col.startswith('grade_')]
    features_numeric.extend(grade_cols)

    # Crear DataFrame con características
    X = df[features_numeric].copy()

    # Manejar valores faltantes
    X = X.fillna(X.mean())

    # Codificar variables categóricas para información adicional
    le_major = LabelEncoder()
    le_device = LabelEncoder()
    le_internet = LabelEncoder()

    df['major_encoded'] = le_major.fit_transform(df['major'].astype(str))
    df['primary_device_encoded'] = le_device.fit_transform(df['primary_device'].astype(str))
    df['internet_type_encoded'] = le_internet.fit_transform(df['internet_type'].astype(str))

    return X, df, features_numeric


def perform_clustering(X, n_clusters=4, random_state=42):
    """Aplica clustering KMeans"""
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    return kmeans, scaler, cluster_labels, X_scaled


def apply_pca(X_scaled, n_components=2):
    """Aplica PCA para reducción dimensional"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca


def get_cluster_statistics(df, cluster_labels):
    """Obtiene estadísticas de cada cluster"""
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels

    cluster_stats = {}
    for i in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == i]

        stats = {
            'count': int(len(cluster_data)),
            'mean_stress': float(cluster_data['stress'].mean()),
            'mean_study_hours': float(cluster_data['study_hours_wk'].mean()),
            'mean_attendance': float(cluster_data['attendance_rate'].mean()),
            'mean_procrastination': float(cluster_data['procrastination_index'].mean()),
            'mean_social_support': float(cluster_data['social_support'].mean()),
            'mean_self_efficacy': float(cluster_data['self_efficacy'].mean()),
            'most_common_major': cluster_data['major'].mode().iat[0] if not cluster_data['major'].mode().empty else 'N/A'
        }

        cluster_stats[f'Cluster {i+1}'] = stats

    return cluster_stats, df_clustered


def generate_cluster_descriptions(cluster_stats):
    """Genera descripciones de clusters basadas en estadísticas"""
    descriptions = {}

    for cluster_name, stats in cluster_stats.items():
        stress_level = "alto" if stats['mean_stress'] > 45 else "moderado" if stats['mean_stress'] > 35 else "bajo"
        study_level = "altas" if stats['mean_study_hours'] > 14 else "moderadas" if stats['mean_study_hours'] > 12 else "bajas"
        procrastination_level = "alta" if stats['mean_procrastination'] > 48 else "moderada" if stats['mean_procrastination'] > 45 else "baja"

        description = f"Estudiantes con estrés {stress_level}, horas de estudio {study_level} y procrastinación {procrastination_level}. "
        description += f"Asistencia promedio: {stats['mean_attendance']:.1f}%. "
        description += f"Carrera predominante: {stats['most_common_major']}."

        descriptions[cluster_name] = description

    return descriptions


def save_clustering_artifacts(kmeans, scaler, pca, output_dir="artefactos/clustering"):
    """Guarda los artefactos del clustering"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(kmeans, f"{output_dir}/kmeans_model.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    joblib.dump(pca, f"{output_dir}/pca_model.joblib")

    print(f"Artefactos guardados en {output_dir}")


def load_clustering_artifacts(artifacts_dir="artefactos/clustering"):
    """Carga los artefactos del clustering"""
    kmeans = joblib.load(f"{artifacts_dir}/kmeans_model.joblib")
    scaler = joblib.load(f"{artifacts_dir}/scaler.joblib")
    pca = joblib.load(f"{artifacts_dir}/pca_model.joblib")

    return kmeans, scaler, pca


if __name__ == "__main__":
    csv_path = "data/dataset.csv"
    print("Cargando datos desde:", csv_path)
    X, df, features_numeric = load_and_preprocess_data(csv_path)

    print("Aplicando clustering KMeans (k=4)...")
    kmeans, scaler, cluster_labels, X_scaled = perform_clustering(X, n_clusters=4)

    print("Aplicando PCA (2 componentes)...")
    pca, X_pca = apply_pca(X_scaled, n_components=2)

    print("Calculando estadísticas por cluster...")
    cluster_stats, df_clustered = get_cluster_statistics(df, cluster_labels)
    descriptions = generate_cluster_descriptions(cluster_stats)

    for k, v in cluster_stats.items():
        print(f"\n{k}: {v}")

    print("Guardando artefactos y dataset con clusters...")
    save_clustering_artifacts(kmeans, scaler, pca)
    df_clustered.to_csv("data/dataset_clustered.csv", index=False)
    print("Listo: data/dataset_clustered.csv y artefactos en artefactos/clustering")
