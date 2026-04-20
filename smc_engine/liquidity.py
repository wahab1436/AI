import pandas as pd
import numpy as np

class LiquidityMapper:
    def find_equal_highs(self, df: pd.DataFrame, tolerance_pct: float = 0.001) -> list:
        highs = df["high"].values
        levels = []
        for i in range(len(highs)):
            for j in range(i + 1, min(i + 10, len(highs))):
                if abs(highs[i] - highs[j]) / highs[i] < tolerance_pct:
                    if highs[i] not in levels:
                        levels.append(highs[i])
                    break
        return levels
    
    def find_equal_lows(self, df: pd.DataFrame, tolerance_pct: float = 0.001) -> list:
        lows = df["low"].values
        levels = []
        for i in range(len(lows)):
            for j in range(i + 1, min(i + 10, len(lows))):
                if abs(lows[i] - lows[j]) / lows[i] < tolerance_pct:
                    if lows[i] not in levels:
                        levels.append(lows[i])
                    break
        return levels
    
    def compute_distances(self, df: pd.DataFrame, atr: float) -> dict:
        current = df["close"].iloc[-1]
        eq_highs = self.find_equal_highs(df.tail(30))
        eq_lows = self.find_equal_lows(df.tail(30))
        
        liq_high_dist = min((h - current) / atr for h in eq_highs) / atr if eq_highs and atr > 0 else 999
        liq_low_dist = min((current - l) / atr for l in eq_lows) / atr if eq_lows and atr > 0 else 999
        
        return {
            "liq_high_distance": liq_high_dist if liq_high_dist != 999 else 5.0,
            "liq_low_distance": liq_low_dist if liq_low_dist != 999 else 5.0
        }
