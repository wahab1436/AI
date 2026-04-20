import pandas as pd
import numpy as np
from typing import Optional

class LabelEngine:
    def __init__(self, atr_period: int = 14, lookahead: int = 8, threshold_multiplier: float = 1.2):
        self.atr_period = atr_period
        self.lookahead = lookahead
        self.threshold_multiplier = threshold_multiplier
        
    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        return tr.rolling(self.atr_period).mean()
    
    def label(self, df: pd.DataFrame, htf_bias: Optional[pd.Series] = None) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = self.compute_atr(df)
        
        future_close = df["close"].shift(-self.lookahead)
        current_close = df["close"]
        future_move = future_close - current_close
        
        threshold = self.threshold_multiplier * df["atr"]
        
        conditions = [
            future_move > threshold,
            future_move < -threshold,
        ]
        choices = [2, 1]
        df["label"] = np.select(conditions, choices, default=0)
        
        if htf_bias is not None:
            df["label"] = np.where(
                (df["label"] == 1) & (htf_bias == 1), 0, df["label"]
            )
            df["label"] = np.where(
                (df["label"] == 2) & (htf_bias == -1), 0, df["label"]
            )
            
        df = df.dropna(subset=["label"])
        return df
