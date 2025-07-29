# src/data_handler/base_processor.py

from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Any

class BaseDataProcessor(ABC):
    """
    Abstract base class for data processing.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.text_column = config['data']['text_column']
        self.label_column = config['data']['label_column']

    @abstractmethod
    def load_and_preprocess(self) -> Dict[str, Dataset]:
        """
        Loads data from file and preprocesses it into a tokenized Dataset.
        Should return a dictionary containing 'train', 'eval', and 'test' datasets.
        """
        pass

    @abstractmethod
    def get_label_info(self) -> Dict[str, Any]:
        """
        Returns information about labels, like number of labels and id-to-label mapping.
        """
        pass