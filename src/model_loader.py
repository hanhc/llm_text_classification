# llm-text-classification/src/model_loader.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import logging

logger = logging.getLogger(__name__)


class ModelLoaderFactory:
    """模型加载器工厂"""

    @staticmethod
    def create_model_and_tokenizer(config, num_labels, label2id, id2label):
        model_config = config['model']
        model_name = model_config['model_name_or_path']
        finetuning_type = model_config.get('finetuning_type', 'sft')
        approach = model_config.get('approach', 'classification_head')

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Fine-tuning approach: {approach}")
        logger.info(f"Fine-tuning type: {finetuning_type}")

        # 1. 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. 配置量化 (QLoRA)
        quantization_config = None
        if finetuning_type == 'qlora':
            qlora_config = model_config['qlora']
            logger.info(f"Applying QLoRA quantization with config: {qlora_config}")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=qlora_config.get('load_in_4bit', True),
                bnb_4bit_quant_type=qlora_config.get('bnb_4bit_quant_type', "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, qlora_config.get('bnb_4bit_compute_dtype', "bfloat16")),
                bnb_4bit_use_double_quant=qlora_config.get('bnb_4bit_use_double_quant', True),
            )

        # 3. 加载模型
        device_map = "auto"
        if approach == 'classification_head':
            model_class = AutoModelForSequenceClassification
            model_kwargs = {
                "num_labels": num_labels,
                "id2label": id2label,
                "label2id": label2id,
                "quantization_config": quantization_config,
                "device_map": device_map,
                "trust_remote_code": True,
            }
            # 对于多标签分类，需要修改problem_type
            if config['project']['task_type'] == 'multi_label':
                model_kwargs['problem_type'] = "multi_label_classification"

            model = model_class.from_pretrained(model_name, **model_kwargs)
            model.config.pad_token_id = tokenizer.pad_token_id

        elif approach == 'generative':
            model_class = AutoModelForCausalLM
            model = model_class.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise ValueError(f"Unsupported model approach: {approach}")

        # 4. 应用PEFT (LoRA/QLoRA)
        if finetuning_type in ['lora', 'qlora']:
            logger.info("Applying PEFT (LoRA/QLoRA)...")
            lora_config_data = model_config['lora']

            peft_task_type = TaskType.SEQ_CLS if approach == 'classification_head' else TaskType.CAUSAL_LM

            peft_config = LoraConfig(
                task_type=peft_task_type,
                r=lora_config_data['r'],
                lora_alpha=lora_config_data['lora_alpha'],
                lora_dropout=lora_config_data['lora_dropout'],
                target_modules=lora_config_data['target_modules'],
                bias="none",
            )
            model = get_peft_model(model, peft_config)
            logger.info("PEFT model created successfully.")
            model.print_trainable_parameters()

        return model, tokenizer