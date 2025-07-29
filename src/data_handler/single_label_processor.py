# src/data_handler/single_label_processor.py

import logging
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from typing import Dict, Any

from .base_processor import BaseDataProcessor
from ..utils.registry import DATA_PROCESSOR_REGISTRY

logger = logging.getLogger(__name__)


@DATA_PROCESSOR_REGISTRY.register("single_label")
class SingleLabelDataProcessor(BaseDataProcessor):
    """
    Data processor for single-label text classification.
    """

    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        super().__init__(config, tokenizer)
        self.label2id = None
        self.id2label = None

    def load_and_preprocess(self) -> DatasetDict:
        logger.info(f"Loading single-label data from {self.config['data']['path']}")
        try:
            df = pd.read_csv(self.config['data']['path'])
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.config['data']['path']}")
            raise

        # Create label mappings
        unique_labels = df[self.label_column].unique()
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        logger.info(f"Found {len(unique_labels)} unique labels.")

        # Map labels to integers
        df['label'] = df[self.label_column].map(self.label2id)

        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=self.config['data']['test_size'],
            random_state=self.config['project']['seed'],
            stratify=df['label']
        )
        train_df, eval_df = train_test_split(
            train_df,
            test_size=self.config['data']['eval_size'],
            random_state=self.config['project']['seed'],
            stratify=train_df['label']
        )

        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        test_dataset = Dataset.from_pandas(test_df)

        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.text_column],
                padding="max_length",
                truncation=True,
                max_length=self.config['model']['max_length']
            )

        tokenized_datasets = DatasetDict({
            "train": train_dataset.map(tokenize_function, batched=True),
            "eval": eval_dataset.map(tokenize_function, batched=True),
            "test": test_dataset.map(tokenize_function, batched=True)
        })

        return tokenized_datasets

    def get_label_info(self) -> Dict[str, Any]:
        if self.label2id is None or self.id2label is None:
            raise ValueError("Label info not available. Run `load_and_preprocess` first.")
        return {
            "num_labels": len(self.label2id),
            "label2id": self.label2id,
            "id2label": self.id2label
        }