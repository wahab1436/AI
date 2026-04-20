import pandas as pd
import numpy as np

class FVGDetecto:
    def detect_bullish_fvg(self, df: pd.DataFrame, atr: float, max_distance_atr: float = 2.0) -> bool:
        if len(df) < 3:
            return False
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        gap_low = max(prev["high"], prev2["high"])
        gap_high = min(prev["low"], prev2["low"])
        
        if gap_low < gap_high:
            if curr["close"] < gap_low and (gap_low - curr["close"]) / atr <= max_distance_atr:
                return True
        return False
    
    def detect_bearish_fvg(self, df: pd.DataFrame, atr: float, max_distance_atr: float = 2.0) -> bool:
        if len(df) < 3:
            return False
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        gap_low = max(prev["high"], prev2["high"])
        gap_high = min(prev["low"], prev2["low"])
        
        if gap_low < gap_high:
            if curr["close"] > gap_high and (curr["close"] - gap_high) / atr <= max_distance_atr:
                return True
        return False
