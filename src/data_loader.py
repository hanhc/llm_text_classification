# llm-text-classification/src/data_loader.py (最终重构版)

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
        self.approach = config['model']['approach']
        self.text_col = config['data_processing']['text_column']
        self.label_col = config['data_processing']['label_column']

    def load_and_preprocess(self):
        """根据 'approach' 配置，选择不同的数据处理流程。"""
        if self.approach == 'classification_head':
            logger.info("Approach is 'classification_head'. Loading and tokenizing data.")
            return self.load_and_tokenize_for_classification()
        elif self.approach == 'generative':
            logger.info("Approach is 'generative'. Loading raw text data for SFT.")
            return self.load_raw_text_for_generation()
        else:
            raise ValueError(f"Unsupported model approach: {self.approach}")

    @abstractmethod
    def load_and_tokenize_for_classification(self):
        """为 classification_head 方法加载并分词数据。"""
        pass

    @abstractmethod
    def load_raw_text_for_generation(self):
        """为 generative 方法加载原始文本数据。"""
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

    def load_raw_text_for_generation(self):
        df = self._load_data()
        # 对于生成式任务，我们只需要原始文本和标签即可。
        # 不需要编码或分词，SFTTrainer会处理。
        self.num_labels = len(df[self.label_col].unique())
        self.label2id, self.id2label = {}, {}  # 生成式任务不需要
        logger.info(f"Loaded raw text dataset with {len(df)} records.")
        return Dataset.from_pandas(df)

    def load_and_tokenize_for_classification(self):
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
            remove_columns=df.columns.tolist()  # 移除所有原始列
        )
        return tokenized_dataset


# MultiLabelLoader 的修改与 SingleLabelLoader 类似
class MultiLabelLoader(BaseDataLoader):
    """多标签数据加载器"""

    def load_raw_text_for_generation(self):
        # 注意: 生成式多标签分类是一个更复杂的任务。
        # SFTTrainer 通常期望一个单一的文本标签。
        # 这里我们假设标签被 delimiter 分隔成一个字符串，例如 "科幻;惊悚"
        logger.warning("Generative multi-label classification is an advanced task. "
                       "Ensure your prompt and model can handle delimited string labels.")
        df = self._load_data()
        self.num_labels = -1  # 在生成式任务中不适用
        self.label2id, self.id2label = {}, {}
        logger.info(f"Loaded raw text dataset for multi-label generation with {len(df)} records.")
        return Dataset.from_pandas(df)

    def load_and_tokenize_for_classification(self):
        df = self._load_data()
        delimiter = self.config['data_processing']['label_delimiter']

        df['labels_list'] = df[self.label_col].astype(str).apply(lambda x: x.split(delimiter))

        mlb = MultiLabelBinarizer()
        encoded_labels = mlb.fit_transform(df['labels_list'])

        self.num_labels = len(mlb.classes_)
        self.label2id = {label: i for i, label in enumerate(mlb.classes_)}
        self.id2label = {i: label for i, label in enumerate(mlb.classes_)}

        logger.info(f"Found {self.num_labels} unique labels for multi-label classification.")
        logger.info(f"Label mapping: {self.label2id}")

        df['labels_one_hot'] = list(encoded_labels)
        dataset = Dataset.from_pandas(df)

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples[self.text_col],
                padding="max_length",
                truncation=True,
                max_length=self.config['data_processing']['max_length']
            )
            tokenized["labels"] = [list(map(float, labels)) for labels in examples['labels_one_hot']]
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=df.columns.tolist()  # 移除所有原始列
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