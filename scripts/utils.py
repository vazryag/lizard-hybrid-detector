import os
import re
import pickle
import typing as t

import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML_FILE_PATH = os.path.join(CURRENT_DIR, "..", "project_config.yaml")


def read_config_yaml() -> dict:
    """Reads the project config YAML file."""
    with open(CONFIG_YAML_FILE_PATH, "r") as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    return data

def write_config_yaml(updated_config: dict) -> None:
    """Writes an updated version of the config YAML file."""
    with open(CONFIG_YAML_FILE_PATH, "w") as file:
        yaml.dump(updated_config, file)
    file.close()

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> t.Tuple[float, float]:
    """
    Evaluates both accuracy and f1-score for the given predictions.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    
    return float(accuracy), float(f1)

def save_pipeline(pipeline: Pipeline) -> None:
    """
    Saves a trained pipeline overwriting the
    existing one.
    """
    PROJECT_CONFIG = read_config_yaml()
    PIPELINE_FILE_PATH = os.path.join(
        CURRENT_DIR,
        PROJECT_CONFIG["pipeline_metadata"]["pipe_path"]
    )

    with open(PIPELINE_FILE_PATH, "wb") as f:
        pickle.dump(pipeline, f)


class FeatureExtractor():
    """
    Implements a feature extractor using a pre-trained
    MobileNetV3 from Keras.
    """
    PROJECT_CONFIG = read_config_yaml()
    KERAS_MODEL_PATH = os.path.join(CURRENT_DIR, PROJECT_CONFIG["extractor"]["file_path"])

    def __init__(self) -> None:
        # handling stringfied tuples from YAML config file with PyYAML
        self.model_ranked_features = [
            int(re.sub(r'[^0-9]', '', idx)) for
            idx in
            self.PROJECT_CONFIG["extractor"]["ranked_features"].split(",")
        ]

        self.mobilenet_model = tf.keras.models.load_model(self.KERAS_MODEL_PATH)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the incoming data, extracting the most
        important features.
        """
        # checking/handling if data is a sample or a dataset
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=0)
        
        # running feature extraction process
        preprocessed_data = tf.keras.applications.mobilenet_v3.preprocess_input(data)
        features = self.mobilenet_model.predict(preprocessed_data)
        selected_features = pd.DataFrame(features)[self.model_ranked_features]
        
        return selected_features.to_numpy()