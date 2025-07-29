# llm-text-classification/src/model_loader.py (Refactored)

import torch
import logging
from typing import Dict, Any, Tuple, Optional, Callable
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    一个解耦的、负责加载和配置模型与分词器的类。

    这个类将模型加载的复杂逻辑分解为独立的步骤：
    1. 加载分词器 (_load_tokenizer)
    2. 创建量化配置 (_create_quantization_config)
    3. 根据指定的策略加载基础模型 (_create_base_model)
    4. 创建PEFT配置 (_create_peft_config)
    5. 应用PEFT包装器 (load 方法)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化加载器。

        Args:
            config (Dict[str, Any]): 项目的完整配置字典。
        """
        self.config = config
        self.model_config = config['model']
        self.model_name = self.model_config['model_name_or_path']

        # 使用分发器模式代替 if/elif
        self.model_creators: Dict[str, Callable] = {
            "classification_head": self._create_classification_model,
            "generative": self._create_generative_model,
        }

    def load(self, num_labels: int, label2id: Dict, id2label: Dict) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        执行完整的模型和分词器加载与配置流程。

        Args:
            num_labels (int): 分类任务的标签数量。
            label2id (Dict): 标签到ID的映射。
            id2label (Dict): ID到标签的映射。

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: 配置完成的模型和分词器。
        """
        # 1. 加载分词器
        tokenizer = self._load_tokenizer()

        # 2. 创建量化配置 (如果需要)
        quantization_config = self._create_quantization_config()

        # 3. 加载基础模型
        base_model = self._create_base_model(
            num_labels, label2id, id2label, quantization_config
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # 4. 应用PEFT (如果需要)
        finetuning_type = self.model_config.get('finetuning_type', 'sft')
        if finetuning_type in ['lora', 'qlora']:
            logger.info(f"Applying {finetuning_type.upper()}...")
            peft_config = self._create_peft_config()
            model = get_peft_model(base_model, peft_config)
            logger.info(f"{finetuning_type.upper()} applied successfully.")
            model.print_trainable_parameters()
            return model, tokenizer

        logger.info("SFT (full finetuning) mode selected. No PEFT wrapping.")
        return base_model, tokenizer

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """加载并配置分词器。"""
        logger.info(f"Loading tokenizer for '{self.model_name}'")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer `pad_token` was None, set to `eos_token`.")
        return tokenizer

    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """根据配置创建QLoRA的量化配置。"""
        if self.model_config.get('finetuning_type') != 'qlora':
            return None

        qlora_config = self.model_config.get('qlora')
        if not qlora_config:
            raise ValueError("QLoRA finetuning selected, but 'qlora' config section is missing.")

        logger.info(f"Applying QLoRA quantization with config: {qlora_config}")
        return BitsAndBytesConfig(
            load_in_4bit=qlora_config.get('load_in_4bit', True),
            bnb_4bit_quant_type=qlora_config.get('bnb_4bit_quant_type', "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, qlora_config.get('bnb_4bit_compute_dtype', "bfloat16")),
            bnb_4bit_use_double_quant=qlora_config.get('bnb_4bit_use_double_quant', True),
        )

    def _create_base_model(self, num_labels: int, label2id: Dict, id2label: Dict,
                           quant_config: Optional[BitsAndBytesConfig]) -> PreTrainedModel:
        """使用分发器动态创建基础模型。"""
        approach = self.model_config.get('approach', 'classification_head')
        creator = self.model_creators.get(approach)

        if not creator:
            raise ValueError(f"Unsupported model approach: {approach}")

        logger.info(f"Creating base model using '{approach}' approach.")
        return creator(num_labels, label2id, id2label, quant_config)

    def _create_classification_model(self, num_labels: int, label2id: Dict, id2label: Dict,
                                     quant_config: Optional[BitsAndBytesConfig]) -> PreTrainedModel:
        """加载用于序列分类的模型。"""
        model_kwargs = {
            "num_labels": num_labels,
            "id2label": id2label,
            "label2id": label2id,
            "quantization_config": quant_config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.config['project']['task_type'] == 'multi_label':
            model_kwargs['problem_type'] = "multi_label_classification"

        return AutoModelForSequenceClassification.from_pretrained(self.model_name, **model_kwargs)

    def _create_generative_model(self, num_labels: int, label2id: Dict, id2label: Dict,
                                 quant_config: Optional[BitsAndBytesConfig]) -> PreTrainedModel:
        """加载用于因果语言建模（生成式）的模型。"""
        # num_labels, label2id, id2label 在这里不直接使用，但为了保持接口一致性而保留
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )

    def _create_peft_config(self) -> LoraConfig:
        """根据配置创建LoRA配置。"""
        lora_config_data = self.model_config.get('lora')
        if not lora_config_data:
            raise ValueError("LoRA/QLoRA finetuning selected, but 'lora' config section is missing.")

        approach = self.model_config.get('approach', 'classification_head')
        task_type = TaskType.SEQ_CLS if approach == 'classification_head' else TaskType.CAUSAL_LM

        return LoraConfig(
            task_type=task_type,
            r=lora_config_data['r'],
            lora_alpha=lora_config_data['lora_alpha'],
            lora_dropout=lora_config_data['lora_dropout'],
            target_modules=lora_config_data['target_modules'],
            bias="none",
        )


class ModelLoaderFactory:
    """
    模型加载器工厂 (兼容层)。

    这个工厂类提供了一个静态方法，作为与项目其他部分（如 train.py）的统一接口。
    它内部实例化并使用解耦的 `ModelLoader` 类来完成实际工作。
    """

    @staticmethod
    def create_model_and_tokenizer(config: Dict[str, Any], num_labels: int,
                                   label2id: Dict, id2label: Dict) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        创建并返回模型和分词器。

        这是一个兼容性接口，委托给新的 `ModelLoader` 类。
        """
        logger.info("Initializing ModelLoader...")
        loader = ModelLoader(config)
        model, tokenizer = loader.load(num_labels, label2id, id2label)
        return model, tokenizer