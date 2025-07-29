# src/model_handler/tuning_strategies/lora.py

import logging
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any

from .base_strategy import BaseTuningStrategy
from ...utils.registry import TUNING_STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


@TUNING_STRATEGY_REGISTRY.register("lora")
class LoRAStrategy(BaseTuningStrategy):
    """
    LoRA (Low-Rank Adaptation) fine-tuning strategy.
    """

    def get_model(self, num_labels: int, label2id: Dict, id2label: Dict) -> PreTrainedModel:
        logger.info(f"Loading model for LoRA: {self.model_config['name']}")

        problem_type = "multi_label_classification" if self.config['data'][
                                                           'type'] == 'multi_label' else "single_label_classification"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['name'],
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            problem_type=problem_type,
            trust_remote_code=True
        )

        lora_params = self.tuning_config['parameters']
        logger.info(f"Applying LoRA with parameters: {lora_params}")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_params['r'],
            lora_alpha=lora_params['lora_alpha'],
            lora_dropout=lora_params['lora_dropout'],
            bias="none",
            target_modules=lora_params.get('target_modules', ["query", "value"])  # Default target modules
        )

        model = get_peft_model(base_model, peft_config)
        logger.info("LoRA configured model created.")
        model.print_trainable_parameters()

        return model