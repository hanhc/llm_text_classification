# src/core/trainer.py

import logging
import os
from transformers import TrainingArguments, Trainer
from typing import Dict, Any
from datasets import Dataset

from .evaluator import compute_metrics_factory

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any], model, tokenizer, datasets: Dict[str, Dataset]):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.training_args_config = config['training']
        self.output_dir = os.path.join(self.training_args_config['output_dir'], config['project']['name'])

    def train(self):
        logger.info("Initializing Trainer...")

        # Setup report_to from config
        report_to = self.training_args_config.get('report_to', 'none')
        if report_to == 'wandb':
            if 'WANDB_PROJECT' not in os.environ:
                 os.environ['WANDB_PROJECT'] = self.config['project']['name']
            logger.info(f"Reporting to wandb project: {os.environ['WANDB_PROJECT']}")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.training_args_config['num_train_epochs'],
            per_device_train_batch_size=self.training_args_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_args_config['per_device_eval_batch_size'],
            warmup_steps=self.training_args_config['warmup_steps'],
            weight_decay=self.training_args_config['weight_decay'],
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=self.training_args_config['logging_steps'],
            evaluation_strategy=self.training_args_config['evaluation_strategy'],
            save_strategy=self.training_args_config['save_strategy'],
            load_best_model_at_end=self.training_args_config['load_best_model_at_end'],
            metric_for_best_model=self.training_args_config['metric_for_best_model'],
            greater_is_better=self.training_args_config['greater_is_better'],
            fp16=self.training_args_config.get('fp16', False),
            report_to=report_to,
            learning_rate=self.training_args_config['learning_rate'],
            gradient_accumulation_steps=self.training_args_config['gradient_accumulation_steps'],
        )

        compute_metrics = compute_metrics_factory(
            self.config['data']['type'],
            self.config['training'].get('multi_label_threshold', 0.5)
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['eval'],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        try:
            logger.info("Starting model training...")
            trainer.train()
            logger.info("Model training completed.")

            logger.info("Saving the best model...")
            trainer.save_model(os.path.join(self.output_dir, "best_model"))
            self.tokenizer.save_pretrained(os.path.join(self.output_dir, "best_model"))
            logger.info(f"Best model saved to {os.path.join(self.output_dir, 'best_model')}")

        except Exception as e:
            logger.error(f"An error occurred during training: {e}", exc_info=True)
            raise

        # Evaluate on test set
        logger.info("Evaluating on the test set...")
        test_results = trainer.evaluate(self.datasets['test'])
        logger.info(f"Test Set Evaluation Results: {test_results}")

        return trainer
