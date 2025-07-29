# src/main.py

import argparse
import logging
import os
from typing import Dict, Any

from utils.logger_setup import setup_logger
from utils.config_loader import load_config
from utils.registry import DATA_PROCESSOR_REGISTRY
from data_handler import * # Import all processors to register them
from model_handler import * # Import all strategies to register them
from model_handler.model_factory import ModelFactory
from core.trainer import ModelTrainer
from core.predictor import Predictor

def run_training(config: Dict[str, Any]):
    """
    Main function to run the training pipeline.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting new training pipeline run.")
    logger.info(f"Configuration: {config}")

    # 1. Setup Tokenizer for Data Processor
    # We need a tokenizer to process data, but the final model/tokenizer pair is created by the factory later.
    # This is a temporary tokenizer for data processing purposes.
    try:
        temp_tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load temporary tokenizer for data processing: {e}")
        raise

    # 2. Data Processing
    try:
        data_processor_class = DATA_PROCESSOR_REGISTRY.get(config['data']['type'])
        data_processor = data_processor_class(config, temp_tokenizer)
        datasets = data_processor.load_and_preprocess()
        label_info = data_processor.get_label_info()
        logger.info(f"Data processed. Datasets created: {datasets.keys()}")
        logger.info(f"Label info: {label_info}")
    except Exception as e:
        logger.error(f"Error during data processing: {e}", exc_info=True)
        raise

    # 3. Model Creation
    try:
        model, tokenizer = ModelFactory.create(config, label_info)
    except Exception as e:
        logger.error(f"Error during model creation: {e}", exc_info=True)
        raise

    # 4. Model Training
    trainer = ModelTrainer(config, model, tokenizer, datasets)
    trainer.train()

    logger.info("Training pipeline finished successfully.")


def run_inference(config: Dict[str, Any], texts: list[str]):
    """
    Function to run inference using a trained model.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting inference run.")

    output_dir = os.path.join(config['training']['output_dir'], config['project']['name'])
    model_path = os.path.join(output_dir, "best_model")

    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Please run training first.")
        return

    try:
        predictor = Predictor(
            model_path=model_path,
            task_type=config['data']['type'],
            multi_label_threshold=config['training'].get('multi_label_threshold', 0.5)
        )

        predictions = predictor.predict(texts)
        logger.info(f"Predictions: {predictions}")
        print("Inference Results:")
        for text, pred in zip(texts, predictions):
            print(f"  Text: '{text}'")
            print(f"  Prediction: {pred}\\n")
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="LLM Text Classification Pipeline")
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "infer"],
        help="The mode to run the script in: 'train' or 'infer'."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs='+', # Allows multiple text inputs
        help="Text to classify (only for 'infer' mode)."
    )
    args = parser.parse_args()

    # Setup logger first
    setup_logger()

    config = load_config(args.config)

    if args.mode == "train":
        run_training(config)
    elif args.mode == "infer":
        if not args.text:
            print("Error: --text argument is required for inference mode.")
            return
        run_inference(config, args.text)


if __name__ == "__main__":
    main()
