# llm-text-classification/src/train.py

import logging
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from trl import SFTTrainer
from .data_loader import DataLoaderFactory
from .model_loader import ModelLoaderFactory
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
import numpy as np
import os

logger = logging.getLogger(__name__)


def compute_metrics_single_label(p):
    """单标签分类评估指标"""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def compute_metrics_multi_label(p):
    """多标签分类评估指标"""
    logits = p.predictions
    labels = p.label_ids

    # 应用sigmoid获取概率
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))

    # 使用0.5作为阈值
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= 0.5)] = 1

    # 计算指标
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    roc_auc = roc_auc_score(labels, preds, average='micro')

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
    }


def run_training(config):
    """执行训练的主函数"""
    try:
        logger.info("===================================")
        logger.info("       STARTING TRAINING           ")
        logger.info("===================================")

        # 1. 设置WandB (如果配置了)
        if config['project']['experiment_tracker'] == 'wandb':
            os.environ['WANDB_PROJECT'] = config['project']['wandb_project']
            logger.info(f"WandB tracking enabled. Project: {config['project']['wandb_project']}")

        # 2. 加载tokenizer (用于数据预处理)
        # 仅加载tokenizer以初始化data_loader
        temp_tokenizer, _ = ModelLoaderFactory.create_model_and_tokenizer(config, 0, {}, {})

        # 3. 加载和预处理数据
        data_loader_factory = DataLoaderFactory.create_data_loader(config, temp_tokenizer)
        tokenized_dataset = data_loader_factory.load_and_preprocess()

        # 分割数据集
        if 'train' in tokenized_dataset.column_names:
            # 假设数据集已经分割好
            train_dataset = tokenized_dataset['train']
            eval_dataset = tokenized_dataset['validation']
        else:
            # 否则手动分割
            split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        # 4. 加载模型
        model, tokenizer = ModelLoaderFactory.create_model_and_tokenizer(
            config,
            num_labels=data_loader_factory.num_labels,
            label2id=data_loader_factory.label2id,
            id2label=data_loader_factory.id2label
        )

        # 5. 显式创建数据整理器
        # 这个整理器会负责将批处理数据正确地填充并转换为Tensor
        logger.info("Initializing DataCollatorWithPadding.")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # 6. 配置训练参数 (这部分不变)
        training_args_config = config['training_args']
        training_args = TrainingArguments(
            output_dir=config['paths']['output_dir'],
            report_to=config['project']['experiment_tracker'],
            **training_args_config
        )

        # 7. 初始化 Trainer (注意，我们只对标准 Trainer 添加 data_collator)
        approach = config['model']['approach']
        task_type = config['project']['task_type']

        trainer = None
        if approach == 'classification_head':
            compute_metrics_fn = compute_metrics_single_label if task_type == 'single_label' else compute_metrics_multi_label
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn
            )
        elif approach == 'generative':
            # 注意：生成式微调需要特定的文本格式，这里是一个简单的例子
            # 您可能需要根据您的任务和模型进行调整
            # TRL的SFTTrainer期望一个'text'列或者dataset_text_field参数
            # 我们在这里将输入和输出拼接成一个文本
            def format_dataset_for_sft(dataset):
                # 这里假设SFTTrainer会处理label，我们需要一个text字段
                # 这是一个简化的例子，实际应用中可能需要更复杂的prompt工程
                # SFTTrainer将 text 和 label 拼接训练
                return dataset.map(lambda x: {
                    'text': f"文本: {x[config['data_processing']['text_column']]}\n分类: {x[config['data_processing']['label_column']]}"})

            formatted_train_dataset = format_dataset_for_sft(train_dataset)
            formatted_eval_dataset = format_dataset_for_sft(eval_dataset)

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=formatted_train_dataset,
                eval_dataset=formatted_eval_dataset,
                dataset_text_field="text",  # 指定包含完整prompt的字段
                max_seq_length=config['data_processing']['max_length'],
                tokenizer=tokenizer,
            )

        # 7. 开始训练
        logger.info("Starting model training...")
        trainer.train()
        logger.info("Training finished.")

        # 8. 保存模型和tokenizer
        logger.info(f"Saving model to {config['paths']['output_dir']}")
        trainer.save_model()
        tokenizer.save_pretrained(config['paths']['output_dir'])
        logger.info("Model and tokenizer saved successfully.")

        # 9. 评估模型
        logger.info("Evaluating model on the evaluation set...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")

        logger.info("===================================")
        logger.info("        TRAINING COMPLETED         ")
        logger.info("===================================")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise