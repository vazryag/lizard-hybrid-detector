import os
import pickle
import logging
import warnings
import typing as t

import bentoml
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning

from utils import FeatureExtractor
from pipeline import inference_pipeline
from utils import read_config_yaml, write_config_yaml
from utils import save_pipeline, evaluate_model_performance

logging.basicConfig(level=logging.WARN)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="skopt")


config = read_config_yaml()


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATASET_COLLECTION_PATH = os.path.join(CURRENT_DIR, config["data"]["image_datasets_collection"])


if __name__ == "__main__":
    # loading dataset npz object
    print("Loading image dataset...")
    dataset = np.load(IMAGE_DATASET_COLLECTION_PATH)

    # loading feature extractor
    print("Setting up feature extractor...")
    feature_extractor = FeatureExtractor()

    # splitting train predictors and target
    x_train = dataset[config["data"]["train_images_set"]]
    y_train = dataset[config["data"]["train_target_column"]]

    # splitting test predictors and target
    x_test = dataset[config["data"]["test_images_set"]]
    y_test = dataset[config["data"]["test_target_column"]]

    # extracting image features
    print("Extracting features...")
    x_train_features = feature_extractor.transform(x_train)
    x_test_features = feature_extractor.transform(x_test)

    # fitting inference pipeline
    print("Fitting the pipeline...")
    inference_pipeline.fit(x_train_features, y_train)

    # scoring trained model on test data
    print("Evaluating the pipeline...")
    y_preds = inference_pipeline.predict(x_test_features)
    new_accuracy, new_f1 = evaluate_model_performance(y_test, y_preds)

    # updating config file performance metrics
    print("Updating project config file...")
    config["pipeline_metadata"]["accuracy"] = new_accuracy
    config["pipeline_metadata"]["f1_score"] = new_f1
    write_config_yaml(updated_config=config)

    # serializing the trained pipeline
    print("Serializing the trained pipeline and finishing...")
    save_pipeline(pipeline=inference_pipeline)