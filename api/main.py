import os
import tempfile
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

from cnn_model.model import CNNFeatureExtractor
from smc_engine.features import SMCFeatureExtractor
from fusion_model.predict import FusionPredictor

app = FastAPI(title="Trading AI System", version="1.0.0")

cnn_extractor = CNNFeatureExtractor()
smc_extractor = SMCFeatureExtractor()
predictor = FusionPredictor()

class PredictionResponse(BaseModel):
    signal: str
    confidence: float
    buy_probability: float
    sell_probability: float
    notrade_probability: float
    pattern_detected: str
    pattern_reliability: float
    smc_analysis: dict
    recommendation: str
    timestamp: str

def get_pattern_from_prediction(probabilities: list, labels: list) -> str:
    max_idx = np.argmax(probabilities)
    if labels[max_idx] == "BUY":
        return "Bullish Engulfing"
    elif labels[max_idx] == "SELL":
        return "Bearish Engulfing"
    return "Indecision / Consolidation"

def format_smc_output(smc_feats: dict) -> dict:
    return {
        "market_structure": "HH/HL Uptrend" if smc_feats["market_state"] == 1 else ("LL/LH Downtrend" if smc_feats["market_state"] == 2 else "Range Bound"),
        "order_block": f"{'Bullish' if smc_feats['market_state']==1 else 'Bearish'} OB at calculated level",
        "fair_value_gap": "Bullish FVG" if smc_feats["fvg_bull_open"] else ("Bearish FVG" if smc_feats["fvg_bear_open"] else "No active FVG"),
        "liquidity_level": f"Nearest liquidity {'above' if smc_feats['liq_high_distance'] > smc_feats['liq_low_distance'] else 'below'}",
        "impulse": "Strong momentum" if smc_feats["impulse_strength"] > 1.5 else "Weak/Normal",
        "market_state": "Trending" if smc_feats["market_state"] != 0 else "Ranging",
        "volatility": "High" if smc_feats["volatility_regime"] > 1.2 else ("Low" if smc_feats["volatility_regime"] < 0.8 else "Medium")
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_chart(
    chart: UploadFile = File(...),
    ohlcv_data: Optional[str] = Form(None)
):
    if not chart.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type. Upload PNG, JPG, or JPEG.")
    
    tmp_path = f"/tmp/{uuid.uuid4()}_{chart.filename}"
    try:
        content = await chart.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        
        cnn_embedding = cnn_extractor.extract(tmp_path)
        
        smc_data = None
        if ohlcv_data:
            df = pd.read_json(ohlcv_data, typ="series")
            smc_data = smc_extractor.extract(df)
        else:
            smc_data = {"hh_hl_ratio": 0.5, "lh_ll_ratio": 0.5, "bos_count_bull": 2, "bos_count_bear": 1,
                        "choch_detected": 0, "dist_nearest_bull_ob": 1.0, "dist_nearest_bear_ob": 1.0,
                        "fvg_bull_open": 0, "fvg_bear_open": 0, "liq_high_distance": 1.5, "liq_low_distance": 1.2,
                        "impulse_strength": 1.1, "market_state": 0, "session_code": 3, "htf_bias": 0, "volatility_regime": 1.0}
        
        smc_vec = np.array([
            smc_data["hh_hl_ratio"], smc_data["lh_ll_ratio"], smc_data["bos_count_bull"],
            smc_data["bos_count_bear"], smc_data["choch_detected"], smc_data["dist_nearest_bull_ob"],
            smc_data["dist_nearest_bear_ob"], smc_data["fvg_bull_open"], smc_data["fvg_bear_open"],
            smc_data["liq_high_distance"], smc_data["liq_low_distance"], smc_data["impulse_strength"],
            float(smc_data["market_state"]), float(smc_data["session_code"]), float(smc_data["htf_bias"]),
            float(smc_data["volatility_regime"])
        ]).reshape(1, 16)
        
        pred = predictor.predict(cnn_embedding, smc_vec)
        pattern = get_pattern_from_prediction(
            [pred["notrade_probability"], pred["sell_probability"], pred["buy_probability"]],
            ["NO_TRADE", "SELL", "BUY"]
        )
        reliability = 0.85 if "Engulfing" in pattern else 0.75 if "Star" in pattern else 0.60
        smc_analysis = format_smc_output(smc_data)
        
        rec = f"Consider {'LONG' if pred['signal']=='BUY' else 'SHORT' if pred['signal']=='SELL' else 'WAIT'} position with stop loss at nearest S/R level"
        
        return PredictionResponse(
            signal=pred["signal"],
            confidence=pred["confidence"],
            buy_probability=pred["buy_probability"],
            sell_probability=pred["sell_probability"],
            notrade_probability=pred["notrade_probability"],
            pattern_detected=pattern,
            pattern_reliability=reliability,
            smc_analysis=smc_analysis,
            recommendation=rec,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
