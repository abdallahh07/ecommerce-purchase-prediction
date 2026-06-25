import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from ecommerce_model.processing.data_manager import load_raw_data, save_pipeline
from ecommerce_model.processing.features import build_session_features
from ecommerce_model.pipeline import create_pipeline

DATA_PATH = "data/2019-Oct.csv"
SAVE_PATH = "ecommerce_model/trained_models/lgbm_pipeline.pkl"

def run_training():
    # Load data
    print("Loading data...")
    df = load_raw_data(DATA_PATH, sample_frac=0.1, random_state=42)

    # Build features
    print("Building features...")
    session_features = build_session_features(df)

    # Split
    X = session_features.drop(columns=["user_session", "purchased", "n_events"])
    y = session_features["purchased"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    print("Training model...")
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))

    # Save
    save_pipeline(pipeline, SAVE_PATH)


if __name__ == "__main__":
    run_training()