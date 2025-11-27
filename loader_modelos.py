import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

def load_bundle(artefact_dir):
    """
    Carga modelo, preprocesador, schema e invert label_map.
    """
    model = tf.keras.models.load_model(os.path.join(artefact_dir, "model.keras"))
    preproc = joblib.load(os.path.join(artefact_dir, "preprocess.joblib"))
    schema = json.load(open(os.path.join(artefact_dir, "input_schema.json"), "r"))
    label_map = json.load(open(os.path.join(artefact_dir, "label_map.json"), "r"))
    reverse_map = {v: k for k, v in label_map.items()}
    return model, preproc, schema, reverse_map


def predict_record(model, preproc, schema, reverse_map, record: dict):
    """
    Recibe un dict → aplica schema → preprocesa → predice → devuelve labels y probabilidades.
    """
    df = pd.DataFrame([record])

    # asegurar que estén todas las columnas
    for col in schema.keys():
        if col not in df.columns:
            df[col] = np.nan

    df = df[list(schema.keys())]

    X_proc = preproc.transform(df)
    pred_probs = model.predict(X_proc)[0]

    pred_int = int(np.argmax(pred_probs))
    pred_label = reverse_map[pred_int]

    prob_dict = {
        reverse_map[i]: float(prob)
        for i, prob in enumerate(pred_probs)
    }

    return {
        "pred_int": pred_int,
        "pred_label": pred_label,
        "probabilidades": prob_dict
    }