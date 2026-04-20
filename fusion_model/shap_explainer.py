import numpy as np
import shap
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        
    def explain(self, X: np.ndarray, top_k: int = 10) -> Dict:
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)
            
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        
        return {
            "top_features": [
                {"name": self.feature_names[i], "importance": float(mean_abs_shap[i])}
                for i in top_indices
            ],
            "shap_values": shap_values.tolist() if X.shape[0] == 1 else None
        }
    
    def detect_drift(self, X_baseline: np.ndarray, X_current: np.ndarray, threshold: float = 0.1) -> bool:
        shap_base = np.abs(self.explainer.shap_values(X_baseline)).mean(axis=0)
        shap_curr = np.abs(self.explainer.shap_values(X_current)).mean(axis=0)
        
        drift = np.abs(shap_base - shap_curr).mean() / (np.abs(shap_base).mean() + 1e-6)
        return drift > threshold
