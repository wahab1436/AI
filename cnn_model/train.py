import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ChartDataset(Dataset):
    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms as T
        
        img = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        else:
            preprocess = T.Compose([
                T.Resize([380, 380]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = preprocess(img)
            
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

class CNNTrainer:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(in_features=1536, out_features=3)
        self.model = self.model.to(self.device)
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 7
    ):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                
            val_f1 = self._evaluate(val_loader)
            scheduler.step(val_f1)
            
            logger.info(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, F1={val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                Path("cnn_model/weights").mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    "cnn_model/weights/cnn_model.pt"
                )
                logger.info("Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
                    
        return best_f1
    
    def _evaluate(self, loader: DataLoader) -> float:
        from sklearn.metrics import f1_score
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        return f1_score(all_labels, all_preds, average="weighted")
