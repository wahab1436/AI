import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yaml
import logging

logger = logging.getLogger(__name__)

class FusionPredictor:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.class_labels = self.config["model"]["class_labels"]
        self.scaler = joblib.load(self.config["paths"]["scaler"])
        self.pca = joblib.load(self.config["paths"]["pca_model"])
        self.model = joblib.load(self.config["paths"]["fusion_model"])
        self.min_conf = self.config["inference"]["min_confidence_threshold"]

    def predict(self, cnn_embedding: np.ndarray, smc_features: np.ndarray) -> dict:
        cnn_pca = self.pca.transform(cnn_embedding.reshape(1, -1))
        
        combined = np.column_stack((cnn_pca, smc_features))
        x_scaled = self.scaler.transform(combined)
        
        probabilities = self.model.predict_proba(x_scaled)[0]
        signal_idx = np.argmax(probabilities)
        confidence = probabilities[signal_idx]
        signal = self.class_labels[signal_idx]
        
        if confidence < self.min_conf:
            signal = "NO_TRADE"
            confidence = float(probabilities[0])
            
        return {
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "buy_probability": round(float(probabilities[self.class_labels.index("BUY")]), 2),
            "sell_probability": round(float(probabilities[self.class_labels.index("SELL")]), 2),
            "notrade_probability": round(float(probabilities[self.class_labels.index("NO_TRADE")]), 2)
        }
