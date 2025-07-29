import argparse
import yaml
import logging
from src.training_pipeline import train_model

def main():
    parser = argparse.ArgumentParser(description="LLM Text Classification Finetuning")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
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
        print(f"Error loading YAML file: {e}")
        return

    # 启动训练
    train_model(config)

if __name__ == "__main__":
    main()