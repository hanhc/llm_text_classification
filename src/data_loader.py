# llm-text-classification/src/data_loader.py

import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """数据加载器抽象基类"""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.text_col = config['data_processing']['text_column']
        self.label_col = config['data_processing']['label_column']

    @abstractmethod
    def load_and_preprocess(self):
        """加载和预处理数据"""
        pass

    def _load_data(self):
        try:
            data_path = self.config['paths']['data_path']
            logger.info(f"Loading data from: {data_path}")
            return pd.read_csv(data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found at: {data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise


class SingleLabelLoader(BaseDataLoader):
    """单标签数据加载器"""

    def load_and_preprocess(self):
        df = self._load_data()

        le = LabelEncoder()
        df['label_id'] = le.fit_transform(df[self.label_col])

        self.num_labels = len(le.classes_)
        self.label2id = {label: i for i, label in enumerate(le.classes_)}
        self.id2label = {i: label for i, label in enumerate(le.classes_)}

        logger.info(f"Found {self.num_labels} unique labels for single-label classification.")
        logger.info(f"Label mapping: {self.label2id}")

        dataset = Dataset.from_pandas(df)

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples[self.text_col],
                padding="max_length",
                truncation=True,
                max_length=self.config['data_processing']['max_length']
            )
            tokenized["labels"] = examples["label_id"]
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=df.columns.tolist()  # 移除原始DataFrame中的所有列
        )
        return tokenized_dataset


class MultiLabelLoader(BaseDataLoader):
    """多标签数据加载器"""

    def load_and_preprocess(self):
        df = self._load_data()
        delimiter = self.config['data_processing']['label_delimiter']

        # 将标签字符串分割成列表
        df['labels_list'] = df[self.label_col].astype(str).apply(lambda x: x.split(delimiter))

        mlb = MultiLabelBinarizer()
        encoded_labels = mlb.fit_transform(df['labels_list'])

        self.num_labels = len(mlb.classes_)
        self.label2id = {label: i for i, label in enumerate(mlb.classes_)}
        self.id2label = {i: label for i, label in enumerate(mlb.classes_)}

        logger.info(f"Found {self.num_labels} unique labels for multi-label classification.")
        logger.info(f"Label mapping: {self.label2id}")

        # 将 one-hot 编码的标签列表添加到DataFrame
        df['labels_one_hot'] = list(encoded_labels)
        dataset = Dataset.from_pandas(df)

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples[self.text_col],
                padding="max_length",
                truncation=True,
                max_length=self.config['data_processing']['max_length']
            )
            # 标签需要是 float 类型以计算 BCEWithLogitsLoss
            tokenized["labels"] = [list(map(float, labels)) for labels in examples['labels_one_hot']]
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=df.columns.tolist()  # 移除原始DataFrame中的所有列
        )
        return tokenized_dataset


class DataLoaderFactory:
    """数据加载器工厂"""

    @staticmethod
    def create_data_loader(config, tokenizer) -> BaseDataLoader:
        task_type = config['project']['task_type']
        logger.info(f"Creating data loader for task type: {task_type}")

        if task_type == 'single_label':
            return SingleLabelLoader(config, tokenizer)
        elif task_type == 'multi_label':
            return MultiLabelLoader(config, tokenizer)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")