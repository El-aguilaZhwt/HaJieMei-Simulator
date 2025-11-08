#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小红书风格评论生成模型完整训练脚本
包含：数据集加载、预处理、训练监控、模型保存
"""

import torch
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import (
    TrainingArguments, 
    TrainerCallback,
    AutoTokenizer
)
import json
import pandas as pd
import os
from time import time
import numpy as np
import psutil
from sklearn.model_selection import train_test_split

# 1. 监控回调类（增强版）
class XiaohongshuTrainerCallback(TrainerCallback):
    def __init__(self):
        self.metrics = {
            'loss': [], 'learning_rate': [], 
            'grad_norm': [], 'gpu_mem': [],
            'epoch': [], 'steps_per_second': []
        }
        self.start_time = time()
        self.last_log_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        # 确保log_history不为空
        if not state.log_history:
            return
            
        # 避免重复记录同一step
        current_step = state.log_history[-1].get('step', 0)
        if current_step <= self.last_log_step:
            return
            
        self.last_log_step = current_step
        
        # 安全获取指标
        current_metrics = {
            'loss': state.log_history[-1].get('loss', None),
            'learning_rate': state.log_history[-1].get('learning_rate', None),
            'grad_norm': state.log_history[-1].get('grad_norm', None),
            'epoch': state.epoch,
            'steps_per_second': current_step/(time()-self.start_time),
        }
        
        # 记录显存使用
        if torch.cuda.is_available():
            current_metrics['gpu_mem'] = torch.cuda.memory_allocated() / 1024**3
        
        # 更新指标
        for k, v in current_metrics.items():
            if v is not None:
                self.metrics[k].append(v)
        
        # 每100步打印报告
        if current_step % 100 == 0:
            self._print_report(state)
    
    def _print_report(self, state):
        """安全打印训练状态"""
        if not self.metrics['loss']:
            return
            
        print(f"\n[Step {self.last_log_step}] 训练状态:")
        print(f"  Epoch: {state.epoch:.1f}/{state.num_train_epochs}")
        print(f"  Loss: {self.metrics['loss'][-1]:.4f}" if self.metrics['loss'] else "  Loss: N/A")
        print(f"  LR: {self.metrics['learning_rate'][-1]:.2e}" if self.metrics['learning_rate'] else "  LR: N/A")
        print(f"  Grad Norm: {self.metrics['grad_norm'][-1]:.2f}" if self.metrics['grad_norm'] else "  Grad Norm: N/A")
        print(f"  Speed: {self.metrics['steps_per_second'][-1]:.1f} steps/s" if self.metrics['steps_per_second'] else "  Speed: N/A")
        print(f"  GPU Mem: {self.metrics['gpu_mem'][-1]:.2f}GB" if self.metrics['gpu_mem'] else "  GPU Mem: N/A")

# 2. 完整数据集加载与预处理
def load_and_process_data(data_path="xiaohongshu_lora_dataset.jsonl"):
    """专为小红书JSONL格式设计的数据加载器"""
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件 {data_path} 不存在")

    # 加载并处理数据
    valid_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                
                # 关键字段检查
                required_fields = ["instruction", "input", "output", "metadata"]
                if not all(field in item for field in required_fields):
                    print(f"行 {line_num}: 缺少必要字段，已跳过")
                    continue
                
                # 处理metadata
                metadata = item["metadata"]
                likes = metadata.get("likes", 0)
                root_id = metadata.get("root_id", "")
                reply_id = metadata.get("reply_id", "")
                
                # 构造训练样本
                valid_data.append({
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": item["output"],
                    "likes": int(likes) if str(likes).isdigit() else 0,
                    "root_id": str(root_id),
                    "reply_id": str(reply_id),
                })
                
            except json.JSONDecodeError:
                print(f"行 {line_num}: JSON解析失败，已跳过")
            except Exception as e:
                print(f"行 {line_num}: 处理错误 - {str(e)}，已跳过")
    
    if not valid_data:
        raise ValueError("没有有效数据，请检查文件内容")
    
    # 转换为DataFrame
    df = pd.DataFrame(valid_data)
    
    # 数据统计
    print(f"\n数据集统计：")
    print(f"总样本数: {len(df)}")
    print(f"平均点赞数: {df['likes'].mean():.1f}")
    print(f"唯一会话数: {df['root_id'].nunique()}")
    
    # 划分训练集和验证集（按root_id分组保证会话完整）
    unique_roots = df['root_id'].unique()
    train_roots, val_roots = train_test_split(unique_roots, test_size=0.1, random_state=42)
    
    train_df = df[df['root_id'].isin(train_roots)]
    val_df = df[df['root_id'].isin(val_roots)]
    
    # 转换为对话格式
    def format_conversation(row):
        return {
            "text": f"<|im_start|>system\n{row['instruction']}<|im_end|>\n"
                    f"<|im_start|>user\n{row['input']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{row['output']}<|im_end|>",
            "input": row['input'],
            "output": row['output'],
            "likes": row['likes'],
            "root_id": row['root_id'],
        }
    
    train_data = Dataset.from_pandas(train_df.apply(format_conversation, axis=1, result_type='expand'))
    val_data = Dataset.from_pandas(val_df.apply(format_conversation, axis=1, result_type='expand'))
    
    print(f"\n最终数据集：")
    print(f"训练集: {len(train_data)} 条 (来自 {len(train_roots)} 个会话)")
    print(f"验证集: {len(val_data)} 条 (来自 {len(val_roots)} 个会话)")
    
    return train_data, val_data

# 3. 模型初始化
def initialize_model():
    print("\n初始化模型...")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA设备，将使用CPU（训练速度极慢）")
    
    # 加载模型 (4bit量化)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/xiuyu/models/Qwen3-14B-unsloth-bnb-4bit",  # 使用7B版本更节省显存
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="auto",
        token=os.getenv('HF_TOKEN'),  # 如果需要HuggingFace token
    )
    
    # 添加特殊token
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # 配置LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    print("模型初始化完成")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"总参数: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer

# 4. 主训练流程
def main():
    # 初始化
    train_data, val_data = load_and_process_data()
    model, tokenizer = initialize_model()
    
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir="./xiaohongshu_train",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=400,
        learning_rate=3e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.5,
        optim="adamw_8bit",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=2048,
        callbacks=[XiaohongshuTrainerCallback()],
    )
    
    # 开始训练
    print("\n===== 开始训练 =====")
    trainer.train()
    
    # 保存最终模型
    save_dir = "./xiaohongshu_final"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"\n训练完成！模型已保存到 {save_dir}")
    print("使用 test_xiaohongshu.py 进行模型测试")

if __name__ == "__main__":
    main()