# llm-text-classification/main.py

import argparse
import yaml
import logging
from src.utils.logging_utils import setup_logger
from src.train import run_training
from src.evaluate import run_evaluation
from src.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description="LLM-based Text Classification Pipeline")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'infer'],
        help="The mode to run the pipeline in: 'train', 'evaluate', or 'infer'."
    )
    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return

    # 设置日志
    logger = setup_logger(config['paths']['log_file'])
    logger.info(f"Successfully loaded configuration from {args.config}")
    logger.info(f"Running in '{args.mode}' mode.")

    # 根据模式调用相应的功能
    try:
        if args.mode == 'train':
            run_training(config)
        elif args.mode == 'evaluate':
            run_evaluation(config)
        elif args.mode == 'infer':
            run_inference(config)
    except Exception as e:
        logger.critical(f"A critical error occurred in '{args.mode}' mode: {e}", exc_info=True)


if __name__ == '__main__':
    main()