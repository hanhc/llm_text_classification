# src/model_handler/tuning_strategies/sft.py

import logging
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from typing import Dict, Any

from .base_strategy import BaseTuningStrategy
from ...utils.registry import TUNING_STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


@TUNING_STRATEGY_REGISTRY.register("sft")
class SFTStrategy(BaseTuningStrategy):
    """
    Standard Fine-Tuning (SFT) strategy.
    """

    def get_model(self, num_labels: int, label2id: Dict, id2label: Dict) -> PreTrainedModel:
        logger.info(f"Loading model for SFT: {self.model_config['name']}")

        problem_type = "multi_label_classification" if self.config['data'][
                                                           'type'] == 'multi_label' else "single_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['name'],
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            problem_type=problem_type,
            trust_remote_code=True  # Needed for some models
        )
        logger.info("Model loaded successfully for SFT.")
        return model