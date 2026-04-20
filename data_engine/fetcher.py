import pandas as pd
import yfinance as yf
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, source: str = "yahoo"):
        self.source = source
        
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        interval = tf_map.get(timeframe, "1h")
        
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data fetched for {symbol} {timeframe}")
            
        df = df.reset_index()
        if "Datetime" in df.columns:
            df["timestamp"] = df["Datetime"]
        elif "Date" in df.columns:
            df["timestamp"] = df["Date"]
            
        df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        
        if output_path:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved data to {output_path}")
            
        return df
