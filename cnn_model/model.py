import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import yaml
import logging

logger = logging.getLogger(__name__)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, config_path: str = "config/model_config.yaml"):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.image_size = self.config["paths"]["image_size"]
        self.embedding_dim = self.config["model"]["embedding_dim"]
        weights_path = self.config["paths"]["cnn_weights"]
        
        weights = EfficientNet_B3_Weights.DEFAULT
        self.backbone = efficientnet_b3(weights=weights)
        self.backbone.classifier = nn.Identity()
        
        if os.path.exists(weights_path):
            self.backbone.load_state_dict(torch.load(weights_path, map_location="cpu"))
            logger.info("Loaded trained CNN weights")
        else:
            logger.warning("No trained CNN weights found. Using pretrained ImageNet.")
            
        self.backbone.eval()
        self.preprocess = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.backbone(x).squeeze(0)
        return features.cpu().numpy()
