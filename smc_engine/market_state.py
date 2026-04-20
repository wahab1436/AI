import pandas as pd
import numpy as np

class MarketStateClassifier:
    def __init__(self, adx_period: int = 14, adx_threshold: float = 25):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
    def _compute_adx(self, df: pd.DataFrame) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        
    def classify(self, df: pd.DataFrame) -> int:
        adx = self._compute_adx(df)
        
        if adx < self.adx_threshold:
            return 0
            
        returns = df["close"].pct_change().tail(20)
        if returns.mean() > 0:
            return 1
        return 2
