import bentoml
import numpy as np
from sklearn.pipeline import Pipeline
from bentoml.io import Image, JSON
from scripts.utils import FeatureExtractor


# initializing feature extractor
feature_extractor = FeatureExtractor()

# retrieving classifier from local storage
lizards_clf_runner = bentoml.sklearn.get("lizards_clf:latest").to_runner()

# running prediction service
svc = bentoml.Service("lizards_classifier", runners=[lizards_clf_runner])


# API endpoints
@svc.api(input=Image(), output=JSON(), route="/api/v1/lizards/predict")
def predict(image: Image) -> JSON:
    """
    This endpoint receives a 3D (RGB) image comprised of a
    dorsal, a lateral and a ventral photos taken from an amazonian
    lizard and then predicts to what specie that specimen belongs. 
    """
    image_array = np.array(image)
    features = feature_extractor.transform(data=image_array)

    result = lizards_clf_runner.predict.run(features)
    result = "anolis" if result == 0 else "hoplocercus" if result == 1 else "polychrus"

    return {"prediction": result, "api_version": "v1"}