import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .model import CNNFeatureExtractor
from pathlib import Path

class EmbeddingExtractor:
    def __init__(self, weights_path: str = "cnn_model/weights/cnn_model.pt"):
        self.extractor = CNNFeatureExtractor()
        if Path(weights_path).exists():
            self.extractor.backbone.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )
        self.extractor.backbone.eval()
        
    def extract_batch(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        from PIL import Image
        import torchvision.transforms as T
        
        preprocess = T.Compose([
            T.Resize([380, 380]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(preprocess(img))
                
            batch = torch.stack(batch_tensors)
            with torch.no_grad():
                feats = self.extractor.backbone(batch)
                embeddings.append(feats.cpu().numpy())
                
        return np.vstack(embeddings)
