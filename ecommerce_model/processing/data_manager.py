import pandas as pd
import joblib
from pathlib import Path

def load_raw_data(data_path: str, sample_frac: float, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df.sample(frac=sample_frac, random_state=random_state)


def save_pipeline(pipeline, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Model saved to {save_path}")


def load_pipeline(save_path: str):
    return joblib.load(save_path)