import os
import joblib

import bentoml
from sklearn.pipeline import Pipeline

from utils import read_config_yaml


config = read_config_yaml()


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_FILE_PATH = os.path.join(CURRENT_DIR, config["pipeline_metadata"]["pipe_path"])


if __name__ == "__main__":
    # loading trained pipeline
    pipeline = joblib.load(PIPELINE_FILE_PATH)

    # saving model to bentoml local storage
    saved_model = bentoml.sklearn.save_model(
        config["pipeline_metadata"]["bento_model_name"],
        pipeline,
        labels={
            "owner": "arthur_g",
            "stage": "dev"
        },
        metadata={
            "accuracy": config["pipeline_metadata"]["accuracy"],
            "f1_score": config["pipeline_metadata"]["f1_score"]
        }
    )

    print(f"Model saved: {saved_model}")