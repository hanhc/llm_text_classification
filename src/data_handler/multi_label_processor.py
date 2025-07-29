# src/data_handler/multi_label_processor.py

import logging
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from typing import Dict, Any, List

from .base_processor import BaseDataProcessor
from ..utils.registry import DATA_PROCESSOR_REGISTRY

logger = logging.getLogger(__name__)


@DATA_PROCESSOR_REGISTRY.register("multi_label")
class MultiLabelDataProcessor(BaseDataProcessor):
    """
    Data processor for multi-label text classification.
    Assumes labels in the CSV are in a string format like "label1,label2,label3".
    """

    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        super().__init__(config, tokenizer)
        self.label2id = None
        self.id2label = None
        self.all_labels = []

    def load_and_preprocess(self) -> DatasetDict:
        logger.info(f"Loading multi-label data from {self.config['data']['path']}")
        try:
            df = pd.read_csv(self.config['data']['path'])
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.config['data']['path']}")
            raise

        # Process labels
        df[self.label_column] = df[self.label_column].apply(lambda x: x.split(','))

        # Create label mappings
        self.all_labels = sorted(list(set(label for sublist in df[self.label_column] for label in sublist)))
        self.label2id = {label: i for i, label in enumerate(self.all_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        logger.info(f"Found {len(self.all_labels)} unique labels.")

        def binarize_labels(examples):
            labels = [0.0] * len(self.all_labels)
            for label in examples[self.label_column]:
                if label in self.label2id:
                    labels[self.label2id[label]] = 1.0
            return {"labels": labels}

        # Apply binarization
        processed_df = df.copy()
        processed_df['labels'] = processed_df.apply(binarize_labels, axis=1)['labels']

        # Split data
        train_df, test_df = train_test_split(
            processed_df,
            test_size=self.config['data']['test_size'],
            random_state=self.config['project']['seed']
        )
        train_df, eval_df = train_test_split(
            train_df,
            test_size=self.config['data']['eval_size'],
            random_state=self.config['project']['seed']
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
            "train": train_dataset.map(tokenize_function, batched=True, remove_columns=df.columns.tolist()),
            "eval": eval_dataset.map(tokenize_function, batched=True, remove_columns=df.columns.tolist()),
            "test": test_dataset.map(tokenize_function, batched=True, remove_columns=df.columns.tolist())
        })

        return tokenized_datasets

    def get_label_info(self) -> Dict[str, Any]:
        if not self.all_labels:
            raise ValueError("Label info not available. Run `load_and_preprocess` first.")
        return {
            "num_labels": len(self.all_labels),
            "label2id": self.label2id,
            "id2label": self.id2label
        }