# 如何运行项目
## 1.安装依赖

```bash
pip install -r requirements.txt
```

注意：bitsandbytes 在 Windows 上安装可能需要特定步骤。在 Linux 上通常更直接。

## 2.配置实验

打开 configs/config.yaml 文件，根据您的需求进行修改。
- 要训练单标签分类，使用 task_type: "single_label" 并指向 data/sample_single_label.csv。
- 要训练多标签分类，使用 task_type: "multi_label" 并指向 data/sample_multi_label.csv。
- 选择 finetuning_type (sft, lora, qlora) 和 approach (classification_head, generative)。
- 设置 output_dir 来保存您的模型。

## 3.开始训练

```bash
python main.py --config configs/config.yaml --mode train
```

训练日志将保存在 logs/app.log，模型文件将保存在 outputs/your_output_dir。

## 4.进行推理
训练完成后，更新 config.yaml 中的 model_checkpoint 为你的模型输出路径 (outputs/your_output_dir)。然后运行：

```bash
python main.py --config configs/config.yaml --mode infer
```

## 5.进行评估
要在一个新的数据集上评估，请更新 config.yaml 中的 data_path 指向你的测试集，并运行：

```bash
python main.py --config configs/config.yaml --mode evaluate
```

