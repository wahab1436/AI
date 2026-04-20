"""
Trading AI System - Main Entry Point
Single file to run, train, or test the entire pipeline.
"""

import argparse
import logging
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("trading_ai_main")

CONFIG_PATH = "config/model_config.yaml"

def load_config(config_path: str = CONFIG_PATH) -> dict:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def verify_environment(config: dict):
    logger.info("Verifying environment and model files...")
    required_paths = [
        config["paths"]["cnn_weights"],
        config["paths"]["fusion_model"],
        config["paths"]["pca_model"],
        config["paths"]["scaler"]
    ]
    missing = [p for p in required_paths if not Path(p).exists()]
    if missing:
        logger.warning("Missing model files (inference will fail until trained):")
        for m in missing:
            logger.warning(f"  - {m}")
    else:
        logger.info("All model files found. System ready.")
    return len(missing) == 0

# =============================================================================
# COMMAND: SERVE API
# =============================================================================
def cmd_serve(args):
    import uvicorn
    from api.main import app
    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

# =============================================================================
# COMMAND: TRAIN CNN
# =============================================================================
def cmd_train_cnn(args):
    from cnn_model.train import CNNTrainer, ChartDataset
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from torch.utils.data import DataLoader
    import torch

    config = load_config(args.config)
    trainer = CNNTrainer(config_path=args.config)
    
    # Build transforms
    transform = Compose([
        Resize(config["paths"]["image_size"]),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = ChartDataset(args.train_images, args.train_labels, transform=transform)
    val_dataset = ChartDataset(args.val_images, args.val_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    logger.info("Starting CNN training...")
    best_f1 = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    logger.info(f"CNN training complete. Best Validation F1: {best_f1:.4f}")

# =============================================================================
# COMMAND: TRAIN FUSION (XGBoost)
# =============================================================================
def cmd_train_fusion(args):
    from fusion_model.train import FusionTrainer
    import numpy as np
    
    config = load_config(args.config)
    trainer = FusionTrainer(config_path=args.config)
    
    cnn_emb = np.load(args.cnn_embeddings)
    smc_feat = np.load(args.smc_features)
    labels = np.load(args.labels)
    
    logger.info("Starting Fusion model training...")
    score = trainer.train(cnn_emb, smc_feat, labels, output_dir=args.output_dir)
    logger.info(f"Fusion training complete. Best Validation F1: {score:.4f}")

# =============================================================================
# COMMAND: EXTRACT FEATURES (CLI)
# =============================================================================
def cmd_extract(args):
    from cnn_model.extract import EmbeddingExtractor
    from smc_engine.features import SMCFeatureExtractor
    import numpy as np
    
    extractor = EmbeddingExtractor(weights_path=args.weights)
    smc = SMCFeatureExtractor()
    
    logger.info(f"Extracting embeddings from {args.input}...")
    embeddings = extractor.extract_batch(args.input, batch_size=32)
    
    # Dummy SMC features if OHLCV not provided
    smc_features = np.zeros((len(embeddings), 16))
    
    np.save(args.output_cnn, embeddings)
    np.save(args.output_smc, smc_features)
    logger.info(f"Features saved to {args.output_cnn} and {args.output_smc}")

# =============================================================================
# COMMAND: PREDICT (CLI INFERENCE)
# =============================================================================
def cmd_predict(args):
    from cnn_model.model import CNNFeatureExtractor
    from smc_engine.features import SMCFeatureExtractor
    from fusion_model.predict import FusionPredictor
    import numpy as np
    
    if not verify_environment(load_config(args.config)):
        logger.error("Cannot run inference without trained model files.")
        sys.exit(1)
        
    cnn = CNNFeatureExtractor(config_path=args.config)
    smc = SMCFeatureExtractor()
    predictor = FusionPredictor(config_path=args.config)
    
    logger.info(f"Analyzing image: {args.image}")
    embedding = cnn.extract(args.image)
    
    # Placeholder SMC vector (replace with actual OHLCV if available)
    smc_vec = np.array([0.5, 0.5, 2, 1, 0, 1.5, 1.2, 0, 0, 1.5, 1.2, 1.1, 0.0, 3.0, 0.0, 1.0]).reshape(1, -1)
    
    result = predictor.predict(embedding, smc_vec)
    
    print("\n" + "="*50)
    print("TRADING AI PREDICTION")
    print("="*50)
    print(json.dumps(result, indent=2))
    print("="*50)

# =============================================================================
# CLI ARGUMENT PARSER
# =============================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog="trading-ai",
        description="Trading AI System - CNN + SMC + XGBoost Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # SERVE
    p_serve = subparsers.add_parser("serve", help="Start FastAPI inference server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Server host")
    p_serve.add_argument("--port", type=int, default=8000, help="Server port")
    p_serve.add_argument("--no-reload", dest="reload", action="store_false", help="Disable auto-reload")
    p_serve.set_defaults(func=cmd_serve)

    # TRAIN CNN
    p_cnn = subparsers.add_parser("train-cnn", help="Train EfficientNet-B3 feature extractor")
    p_cnn.add_argument("--config", default=CONFIG_PATH)
    p_cnn.add_argument("--train-images", required=True, nargs="+", help="Paths to training images")
    p_cnn.add_argument("--train-labels", required=True, help="Path to training labels (.npy)")
    p_cnn.add_argument("--val-images", required=True, nargs="+", help="Paths to validation images")
    p_cnn.add_argument("--val-labels", required=True, help="Path to validation labels (.npy)")
    p_cnn.add_argument("--epochs", type=int, default=50)
    p_cnn.add_argument("--lr", type=float, default=1e-4)
    p_cnn.add_argument("--weight-decay", type=float, default=1e-4)
    p_cnn.add_argument("--patience", type=int, default=7)
    p_cnn.set_defaults(func=cmd_train_cnn)

    # TRAIN FUSION
    p_fusion = subparsers.add_parser("train-fusion", help="Train XGBoost fusion classifier")
    p_fusion.add_argument("--config", default=CONFIG_PATH)
    p_fusion.add_argument("--cnn-embeddings", required=True, help="Path to CNN embeddings (.npy)")
    p_fusion.add_argument("--smc-features", required=True, help="Path to SMC features (.npy)")
    p_fusion.add_argument("--labels", required=True, help="Path to labels (.npy)")
    p_fusion.add_argument("--output-dir", default="models")
    p_fusion.set_defaults(func=cmd_train_fusion)

    # EXTRACT
    p_extract = subparsers.add_parser("extract", help="Extract CNN embeddings and SMC features")
    p_extract.add_argument("--weights", default="cnn_model/weights/cnn_model.pt")
    p_extract.add_argument("--input", nargs="+", required=True, help="Image files or directories")
    p_extract.add_argument("--output-cnn", default="data/features/cnn_embeddings.npy")
    p_extract.add_argument("--output-smc", default="data/features/smc_features.npy")
    p_extract.set_defaults(func=cmd_extract)

    # PREDICT
    p_predict = subparsers.add_parser("predict", help="Run inference on a chart image")
    p_predict.add_argument("image", help="Path to chart image (PNG/JPG)")
    p_predict.add_argument("--config", default=CONFIG_PATH)
    p_predict.set_defaults(func=cmd_predict)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
