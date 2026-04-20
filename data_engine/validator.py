import pandas as pd
import numpy as np
from typing import Tuple

class DataValidator:
    @staticmethod
    def fill_gaps(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
        df = df.set_index("timestamp")
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=freq))
        df = df.ffill().bfill()
        df = df.reset_index().rename(columns={"index": "timestamp"})
        return df
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, atr: pd.Series, threshold: float = 5.0) -> pd.DataFrame:
        df = df.copy()
        returns = df["close"].pct_change()
        outlier_mask = abs(returns) > (threshold * atr / df["close"])
        df.loc[outlier_mask, ["open", "high", "low", "close"]] = np.nan
        df = df.ffill().bfill()
        return df
    
    @staticmethod
    def validate_schema(df: pd.DataFrame) -> bool:
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        return all(col in df.columns for col in required)
    
    @staticmethod
    def tag_sessions(df: pd.DataFrame, timezone: str = "UTC") -> pd.DataFrame:
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(timezone).dt.tz_convert("Europe/London").dt.hour
        conditions = [
            (df["hour"] >= 0) & (df["hour"] < 7),
            (df["hour"] >= 7) & (df["hour"] < 12),
            (df["hour"] >= 12) & (df["hour"] < 17),
            (df["hour"] >= 17) & (df["hour"] < 24),
        ]
        choices = [0, 1, 2, 3]
        df["session_code"] = np.select(conditions, choices, default=0)
        return df
