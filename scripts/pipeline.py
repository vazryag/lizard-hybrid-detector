from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from utils import read_config_yaml


config = read_config_yaml()


inference_pipeline = Pipeline(
    [
        ("minmax_rescaler", MinMaxScaler(feature_range=(0, 1))),
        ("linear_svm_classifier", LinearSVC(**config["classifier"]))
    ]
)