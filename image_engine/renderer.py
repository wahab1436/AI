import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import numpy as np
import io
from typing import Optional

class ChartRenderer:
    def __init__(
        self,
        width: int = 380,
        height: int = 380,
        candle_window: int = 50,
        bg_color: str = "#000000",
        ema_periods: list = [20, 50]
    ):
        self.width = width
        self.height = height
        self.candle_window = candle_window
        self.bg_color = bg_color
        self.ema_periods = ema_periods
        
    def _compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    def render(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Image.Image:
        df = df.tail(self.candle_window).copy()
        df = df.reset_index(drop=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(self.width/100, self.height/100), dpi=100)
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        x = np.arange(len(df))
        up = df["close"] >= df["open"]
        down = ~up
        
        ax.bar(x[up], df["close"][up] - df["open"][up], 
               bottom=df["open"][up], color="#00ff00", width=0.6)
        ax.bar(x[down], df["close"][down] - df["open"][down],
               bottom=df["open"][down], color="#ff0000", width=0.6)
        
        ax.vlines(x, df["low"], df["high"], color="#888888", linewidth=0.5)
        
        for period in self.ema_periods:
            ema = self._compute_ema(df["close"], period)
            color = "#0000ff" if period == 20 else "#ffa500"
            ax.plot(x, ema, color=color, linewidth=1, label=f"EMA{period}")
        
        ax.set_xlim(-1, len(df))
        ax.set_ylim(df["low"].min() * 0.999, df["high"].max() * 1.001)
        ax.axis("off")
        
        vol_height = int(self.height * 0.15)
        vol_ax = fig.add_axes([0, 0, 1, 0.15])
        vol_ax.bar(x, df["volume"], color="#444444", width=0.6)
        vol_ax.axis("off")
        vol_ax.set_facecolor(self.bg_color)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=self.bg_color)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)
        
        if output_path:
            img.save(output_path)
            
        return img
