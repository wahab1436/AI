# Trading AI System

CNN + SMC + XGBoost pipeline for chart image analysis.

## Architecture
Image (380x380) -> EfficientNet-B3 -> 1536-dim embedding -> PCA(128) + SMC(16) -> XGBoost -> BUY/SELL/NO_TRADE

## Quick Start
```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
