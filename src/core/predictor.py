# src/core/predictor.py

import torch
import logging
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model_path: str, task_type: str, multi_label_threshold: float = 0.5):
        self.model_path = model_path
        self.task_type = task_type
        self.multi_label_threshold = multi_label_threshold
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model for prediction from {self.model_path}")
        try:
            # Check if it's a PEFT model
            is_peft = "adapter_config.json" in [f for f in __import__("os").listdir(self.model_path)]

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            if is_peft:
                logger.info("Detected PEFT model (LoRA/QLoRA). Loading with PeftModel.")
                config = __import__("peft").PeftConfig.from_pretrained(self.model_path)
                base_model_name = config.base_model_name_or_path
                base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=len(config.id2label), trust_remote_code=True)
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                logger.info("Detected standard SFT model. Loading with AutoModelForSequenceClassification.")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

            self.model.eval()
            if torch.cuda.is_available():
                self.model.to('cuda')
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict]:
        if not isinstance(texts, list):
            texts = [texts]

        pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True if self.task_type == 'multi_label' else False
        )

        results = pipe(texts)

        if self.task_type == 'multi_label':
            processed_results = []
            for result_list in results:
                labels = [
                    res['label'] for res in result_list if res['score'] > self.multi_label_threshold
                ]
                scores = [
                    res['score'] for res in result_list if res['score'] > self.multi_label_threshold
                ]
                processed_results.append({"labels": labels, "scores": scores})
            return processed_results
        else: # single_label
            return results