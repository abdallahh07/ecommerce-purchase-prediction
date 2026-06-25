import pandas as pd

def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    session_features = df.groupby("user_session").agg(
        n_views=("event_type", lambda x: (x == "view").sum()),
        n_carts=("event_type", lambda x: (x == "cart").sum()),
        n_events=("event_type", "count"),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
        n_unique_products=("product_id", "nunique"),
        purchased=("event_type", lambda x: int((x == "purchase").any()))
    ).reset_index()

    return session_features