import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class StructureAnalyzer:
    def __init__(self, order: int = 5):
        self.order = order
        
    def detect_swings(self, df: pd.DataFrame) -> dict:
        highs = df["high"].values
        lows = df["low"].values
        
        peak_idx = argrelextrema(highs, np.greater, order=self.order)[0]
        trough_idx = argrelextrema(lows, np.less, order=self.order)[0]
        
        peaks = [(i, highs[i]) for i in peak_idx]
        troughs = [(i, lows[i]) for i in trough_idx]
        
        return {"peaks": peaks, "troughs": troughs}
    
    def compute_structure_ratios(self, df: pd.DataFrame, lookback: int = 20) -> dict:
        swings = self.detect_swings(df.tail(lookback * 2))
        peaks = swings["peaks"]
        troughs = swings["troughs"]
        
        if len(peaks) < 2 or len(troughs) < 2:
            return {"hh_hl_ratio": 0.5, "lh_ll_ratio": 0.5}
            
        hh = sum(1 for i in range(1, len(peaks)) if peaks[i][1] > peaks[i-1][1])
        hl = sum(1 for i in range(1, len(troughs)) if troughs[i][1] > troughs[i-1][1])
        lh = sum(1 for i in range(1, len(peaks)) if peaks[i][1] < peaks[i-1][1])
        ll = sum(1 for i in range(1, len(troughs)) if troughs[i][1] < troughs[i-1][1])
        
        total_bull = hh + hl + 1e-6
        total_bear = lh + ll + 1e-6
        
        return {
            "hh_hl_ratio": (hh + hl) / total_bull,
            "lh_ll_ratio": (lh + ll) / total_bear
        }
    
    def detect_bos(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        swings = self.detect_swings(df.tail(lookback))
        peaks = swings["peaks"]
        troughs = swings["troughs"]
        
        bos_bull = 0
        bos_bear = 0
        
        for i in range(1, len(troughs)):
            if troughs[i][1] > peaks[i-1][1] if i-1 < len(peaks) else False:
                bos_bull += 1
                
        for i in range(1, len(peaks)):
            if peaks[i][1] < troughs[i-1][1] if i-1 < len(troughs) else False:
                bos_bear += 1
                
        return {"bos_count_bull": bos_bull, "bos_count_bear": bos_bear}
    
    def detect_choch(self, df: pd.DataFrame, lookback: int = 20) -> int:
        struct = self.compute_structure_ratios(df, lookback)
        if struct["hh_hl_ratio"] < 0.3 and struct["lh_ll_ratio"] > 0.6:
            return 1
        if struct["hh_hl_ratio"] > 0.6 and struct["lh_ll_ratio"] < 0.3:
            return 1
        return 0
