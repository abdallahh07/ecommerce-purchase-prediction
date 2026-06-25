import pandas as pd
from ecommerce_model.processing.data_manager import load_pipeline
from ecommerce_model.processing.features import build_session_features

SAVE_PATH = "ecommerce_model/trained_models/lgbm_pipeline.pkl"

def make_prediction(df: pd.DataFrame) -> pd.DataFrame:
    # Build features
    session_features = build_session_features(df)
    X = session_features.drop(columns=["user_session", "purchased", "n_events"])

    # Load model and predict
    pipeline = load_pipeline(SAVE_PATH)
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    session_features["predicted_purchase"] = predictions
    session_features["purchase_probability"] = probabilities

    return session_features[["user_session", "predicted_purchase", "purchase_probability"]]


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/2019-Oct.csv").sample(1000, random_state=42)
    results = make_prediction(df)
    print(results.head())