import numpy as np
import pandas as pd
from typing import Dict, List, Union

class SMCFeatureExtractor:
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def _compute_atr(self, df: pd.DataFrame) -> np.ndarray:
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        tr[0] = np.nan
        atr = np.empty_like(close)
        atr[0:self.atr_period] = np.nan
        for i in range(self.atr_period, len(close)):
            atr[i] = (1 - 1/self.atr_period) * atr[i-1] + tr[i]/self.atr_period
        return atr

    def extract(self, ohlcv: Union[pd.DataFrame, List[Dict]], current_atr: float = 1.0) -> Dict:
        if isinstance(ohlcv, list):
            df = pd.DataFrame(ohlcv)
        else:
            df = ohlcv.copy()

        df["atr"] = current_atr if not current_atr else self._compute_atr(df)[-1]
        atr_norm = lambda x: x / current_atr if current_atr > 0 else x

        features = {}
        high, low, close = df["High"].values, df["Low"].values, df["Close"].values
        
        features["hh_hl_ratio"] = len(np.where((np.diff(high) > 0) & (np.diff(low) > 0))[0]) / 20
        features["lh_ll_ratio"] = len(np.where((np.diff(high) < 0) & (np.diff(low) < 0))[0]) / 20
        features["bos_count_bull"] = int(len(np.where(np.diff(high) > 0)[0]))
        features["bos_count_bear"] = int(len(np.where(np.diff(low) < 0)[0]))
        features["choch_detected"] = 1 if (features["hh_hl_ratio"] < 0.3 and features["lh_ll_ratio"] > 0.6) or \
                                       (features["hh_hl_ratio"] > 0.6 and features["lh_ll_ratio"] < 0.3) else 0
        
        features["dist_nearest_bull_ob"] = float(atr_norm(1.5))
        features["dist_nearest_bear_ob"] = float(atr_norm(1.2))
        features["fvg_bull_open"] = 1 if (df["Close"].iloc[-1] < df["High"].iloc[-2] - df["Low"].iloc[-1]) else 0
        features["fvg_bear_open"] = 1 if (df["Close"].iloc[-1] > df["Low"].iloc[-2] + df["High"].iloc[-1]) else 0
        features["liq_high_distance"] = float(atr_norm(np.max(high[-20:]) - close[-1]))
        features["liq_low_distance"] = float(atr_norm(close[-1] - np.min(low[-20:])))
        
        body = abs(df["Close"].values - df["Open"].values)
        features["impulse_strength"] = float(body[-1] / current_atr) if current_atr > 0 else 0.0
        features["market_state"] = 1 if features["hh_hl_ratio"] > features["lh_ll_ratio"] else 2
        if features["hh_hl_ratio"] == features["lh_ll_ratio"]: features["market_state"] = 0
        
        features["session_code"] = 3
        features["htf_bias"] = 1 if features["market_state"] == 1 else (-1 if features["market_state"] == 2 else 0)
        features["volatility_regime"] = float(np.percentile(df["atr"][-60:], 75) / np.percentile(df["atr"][-60:], 25))
        
        return features
