# src/model_handler/tuning_strategies/base_strategy.py

from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Any

class BaseTuningStrategy(ABC):
    """
    Abstract base class for model fine-tuning strategies.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.tuning_config = config['tuning']

    @abstractmethod
    def get_model(self, num_labels: int, label2id: Dict, id2label: Dict) -> PreTrainedModel:
        """
        Loads and configures the model according to the specific tuning strategy.
        """
        pass