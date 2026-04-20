import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FusionTrainer:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.pca = PCA(n_components=self.config["model"]["pca_components"])
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42
        )
        
    def train(
        self,
        cnn_embeddings: np.ndarray,
        smc_features: np.ndarray,
        labels: np.ndarray,
        output_dir: str = "models"
    ):
        cnn_pca = self.pca.fit_transform(cnn_embeddings)
        X = np.column_stack((cnn_pca, smc_features))
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=5, gap=50)
        best_score = 0
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = labels[train_idx], labels[val_idx]
            
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            preds = self.model.predict(X_val)
            score = f1_score(y_val, preds, average="weighted")
            
            if score > best_score:
                best_score = score
                
        logger.info(f"Best validation F1: {best_score:.4f}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, f"{output_dir}/fusion_model.pkl")
        joblib.dump(self.pca, f"{output_dir}/pca.pkl")
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        
        return best_score
