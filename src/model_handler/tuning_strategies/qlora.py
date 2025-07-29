# src/model_handler/tuning_strategies/qlora.py

import logging
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from typing import Dict, Any

from .base_strategy import BaseTuningStrategy
from ...utils.registry import TUNING_STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


@TUNING_STRATEGY_REGISTRY.register("qlora")
class QLoRAStrategy(BaseTuningStrategy):
    """
    QLoRA (Quantized Low-Rank Adaptation) fine-tuning strategy.
    """

    def get_model(self, num_labels: int, label2id: Dict, id2label: Dict) -> PreTrainedModel:
        logger.info(f"Loading model for QLoRA: {self.model_config['name']}")

        qlora_params = self.tuning_config['parameters']
        logger.info(f"Using QLoRA with quantization parameters: {qlora_params['quantization']}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_params['quantization'].get('load_in_4bit', True),
            bnb_4bit_quant_type=qlora_params['quantization'].get('bnb_4bit_quant_type', "nf4"),
            bnb_4bit_compute_dtype=getattr(torch,
                                           qlora_params['quantization'].get('bnb_4bit_compute_dtype', "float16")),
            bnb_4bit_use_double_quant=qlora_params['quantization'].get('bnb_4bit_use_double_quant', True),
        )

        problem_type = "multi_label_classification" if self.config['data'][
                                                           'type'] == 'multi_label' else "single_label_classification"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['name'],
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            problem_type=problem_type,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map={"": 0}  # Automatically map to GPU
        )

        # Prepare model for k-bit training
        base_model = prepare_model_for_kbit_training(base_model)

        logger.info(f"Applying LoRA with parameters: {qlora_params['lora']}")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=qlora_params['lora']['r'],
            lora_alpha=qlora_params['lora']['lora_alpha'],
            lora_dropout=qlora_params['lora']['lora_dropout'],
            bias="none",
            target_modules=qlora_params['lora'].get('target_modules', ["query", "key", "value"])
        )

        model = get_peft_model(base_model, peft_config)
        logger.info("QLoRA configured model created.")
        model.print_trainable_parameters()

        return model