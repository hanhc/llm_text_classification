# src/model_handler/model_factory.py

import logging
from transformers import AutoTokenizer, PreTrainedModel
from typing import Dict, Any, Tuple

from ..utils.registry import TUNING_STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class to create tokenizer and model based on configuration.
    """
    @staticmethod
    def create(config: Dict[str, Any], label_info: Dict[str, Any]) -> Tuple[PreTrainedModel, AutoTokenizer]:
        """
        Creates and returns the model and tokenizer.
        """
        model_name = config['model']['name']
        tuning_strategy_name = config['tuning']['strategy']

        # 1. Create Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{model_name}': {e}")
            raise

        # 2. Get Tuning Strategy from Registry
        try:
            tuning_strategy_class = TUNING_STRATEGY_REGISTRY.get(tuning_strategy_name)
            tuning_strategy = tuning_strategy_class(config)
        except KeyError:
            logger.error(f"Tuning strategy '{tuning_strategy_name}' is not registered.")
            raise

        # 3. Create Model using the strategy
        try:
            model = tuning_strategy.get_model(
                num_labels=label_info['num_labels'],
                label2id=label_info['label2id'],
                id2label=label_info['id2label']
            )
        except Exception as e:
            logger.error(f"Failed to create model using '{tuning_strategy_name}' strategy: {e}")
            raise

        return model, tokenizer

