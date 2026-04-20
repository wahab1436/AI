import pandas as pd
import numpy as np

class OrderBlockDetector:
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        
    def _compute_atr(self, df: pd.DataFrame) -> np.ndarray:
        high, low, close = df["high"], df["low"], df["close"]
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        return tr.rolling(self.atr_period).mean().values
    
    def detect_bullish_ob(self, df: pd.DataFrame) -> dict:
        atr = self._compute_atr(df)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 1.0
        
        for i in range(len(df) - 1, 0, -1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                if i > 0 and df["close"].iloc[i-1] < df["open"].iloc[i-1]:
                    ob_low = df["low"].iloc[i]
                    ob_high = df["high"].iloc[i]
                    dist = (df["close"].iloc[-1] - ob_high) / current_atr if current_atr > 0 else 1.0
                    return {"detected": True, "level": (ob_low + ob_high) / 2, "distance_atr": dist}
        return {"detected": False, "level": None, "distance_atr": 999}
    
    def detect_bearish_ob(self, df: pd.DataFrame) -> dict:
        atr = self._compute_atr(df)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 1.0
        
        for i in range(len(df) - 1, 0, -1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i > 0 and df["close"].iloc[i-1] > df["open"].iloc[i-1]:
                    ob_low = df["low"].iloc[i]
                    ob_high = df["high"].iloc[i]
                    dist = (ob_low - df["close"].iloc[-1]) / current_atr if current_atr > 0 else 1.0
                    return {"detected": True, "level": (ob_low + ob_high) / 2, "distance_atr": dist}
        return {"detected": False, "level": None, "distance_atr": 999}
