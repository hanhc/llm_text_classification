# llm-text-classification/src/inference.py

import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import numpy as np
import json

logger = logging.getLogger(__name__)


def run_inference(config):
    """执行推理的主函数"""
    try:
        logger.info("===================================")
        logger.info("       STARTING INFERENCE          ")
        logger.info("===================================")

        model_checkpoint = config['paths']['model_checkpoint']
        inference_text = config['inference']['inference_text']
        approach = config['model']['approach']
        task_type = config['project']['task_type']

        logger.info(f"Loading model from checkpoint: {model_checkpoint}")
        logger.info(f"Performing inference on text: '{inference_text}'")

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

        if approach == 'classification_head':
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, trust_remote_code=True)
            model.eval()

            inputs = tokenizer(inference_text, return_tensors="pt", padding=True, truncation=True,
                               max_length=config['data_processing']['max_length'])

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            if task_type == 'single_label':
                predicted_class_id = torch.argmax(logits, dim=1).item()
                predicted_label = model.config.id2label[predicted_class_id]
                logger.info(f"Predicted Label: {predicted_label}")

            elif task_type == 'multi_label':
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits)
                preds = (probs > 0.5).int()
                predicted_labels = [model.config.id2label[i] for i, label in enumerate(preds[0]) if label == 1]
                logger.info(f"Predicted Labels: {', '.join(predicted_labels) if predicted_labels else 'None'}")

        elif approach == 'generative':
            model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True)
            # 使用pipeline简化生成任务
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                            device=0 if torch.cuda.is_available() else -1)

            # 从配置中读取完全相同的Prompt模板
            prompt_template = config['model'].get('prompt_template', "文本: {text}\n分类:")
            if '{text}' not in prompt_template:
                raise ValueError("`prompt_template` in config must contain the placeholder '{text}'.")

            # 使用模板构建推理时的输入Prompt
            prompt = prompt_template.format(text=inference_text)

            logger.info(f"Constructed Inference Prompt:\n{prompt}")

            # `max_new_tokens` 应该设置得比较小，因为我们只需要标签
            # `pad_token_id` 对于开放送式生成很重要
            raw_output = pipe(prompt, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

            generated_text = raw_output[0]['generated_text']
            # 从生成文本中解析出标签
            # 这是一个简单的解析逻辑，可能需要根据实际输出来优化
            prediction = generated_text[len(prompt):].strip()

            logger.info(f"Generated Text: {generated_text}")
            logger.info(f"Parsed Prediction: {prediction}")

        logger.info("===================================")
        logger.info("        INFERENCE COMPLETED        ")
        logger.info("===================================")

    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at: {model_checkpoint}. Please train a model first.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}", exc_info=True)
        raise

# 评估脚本可以类似地实现，这里不再赘述以保持简洁
# src/evaluate.py 会很像 train.py 的评估部分，但它会加载一个已存在的模型
# 并只在测试集上运行 trainer.evaluate()