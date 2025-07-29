
### **如何运行项目**

1. **安装依赖**:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
2. **(可选) 登录Wandb**: 如果你要使用`wandb`进行实验跟踪，请先登录。
    
    ```bash
    wandb login
    
    ```
    
3. **运行训练**:
选择一个配置文件并运行 `main.py`。
    - **训练一个单标签SFT模型**:
        
        ```bash
        python src/main.py train --config configs/single_label_sft.yaml
        
        ```
        
    - **训练一个多标签LoRA模型**:
        
        ```bash
        python src/main.py train --config configs/multi_label_lora.yaml
        
        ```
        
    
    训练过程中的日志会保存在 `logs/app.log`，训练结果（模型、tokenizer、检查点）会保存在 `outputs/` 目录下对应的项目文件夹中。你可以在 `outputs/` 目录或Wandb界面查看TensorBoard日志。
    
4. **运行推理**:
训练完成后，使用训练时相同的配置文件来加载模型进行推理。
    
    ```bash
    python src/main.py infer --config configs/single_label_sft.yaml --text "这台电脑性能如何" "这个菜真好吃"
    
    ```
    
    或者对多标签模型进行推理：
    
    ```bash
    python src/main.py infer --config configs/multi_label_lora.yaml --text "电影院音效很好，但服务员态度一般"
    
    ```