import pandas as pd
import numpy as np

class ImpulseAnalyzer:
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        
    def _compute_atr(self, df: pd.DataFrame) -> float:
        high, low, close = df["high"], df["low"], df["close"]
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(self.atr_period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 1.0
    
    def compute_strength(self, df: pd.DataFrame) -> float:
        atr = self._compute_atr(df)
        last_candle = df.iloc[-1]
        body = abs(last_candle["close"] - last_candle["open"])
        return body / atr if atr > 0 else 0.0
