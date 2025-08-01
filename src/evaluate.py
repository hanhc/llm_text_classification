# llm-text-classification/src/evaluate.py
# 这个脚本可以用来在独立的测试集上评估已训练好的模型。
# 它的结构会和 train.py 非常相似。

import logging
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from .data_loader import DataLoaderFactory
from .train import compute_metrics_single_label, compute_metrics_multi_label

logger = logging.getLogger(__name__)


def run_evaluation(config):
    try:
        logger.info("===================================")
        logger.info("      STARTING EVALUATION          ")
        logger.info("===================================")

        model_checkpoint = config['paths']['model_checkpoint']

        # 1. 加载Tokenizer和模型
        logger.info(f"Loading model from checkpoint: {model_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        # 评估通常针对 classification_head 方法
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, trust_remote_code=True)

        # 2. 加载评估数据
        # 假设配置文件中的 data_path 指向测试集
        data_loader = DataLoaderFactory.create_data_loader(config, tokenizer)
        eval_dataset = data_loader.load_and_preprocess()
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        # 3. 配置Trainer
        training_args = TrainingArguments(
            output_dir=f"{model_checkpoint}/eval_results",
            per_device_eval_batch_size=config['training_args']['per_device_eval_batch_size'],
            report_to="none",  # 评估时通常不需要上报
        )

        compute_metrics_fn = compute_metrics_single_label if config['project'][
                                                                 'task_type'] == 'single_label' else compute_metrics_multi_label

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
        )

        # 4. 运行评估
        results = trainer.evaluate()
        logger.info("Evaluation results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

        logger.info("===================================")
        logger.info("      EVALUATION COMPLETED         ")
        logger.info("===================================")

    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {model_checkpoint}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
        raise