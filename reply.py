#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小红书风格评论回复生成器
基于LoRA微调的Qwen3-14B模型
"""

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
import readline  # 用于改进输入体验

# 初始化模型
def load_model():
    print("正在加载模型...")
    
    try:
        # 先加载基础模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/home/xiuyu/models/Qwen3-14B-unsloth-bnb-4bit",  # 使用与训练时相同的原始模型
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
        )
        
        # 添加特殊token
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        
        # 然后加载LoRA适配器
        model.load_adapter("/home/xiuyu/qwen3train/xiaohongshu_train/checkpoint-6492")
        
        # 显存优化配置
        model.config.use_cache = True
        model.config.pretraining_tp = 1 
        
        print(f"模型加载完成！词汇表大小: {len(tokenizer)}")
        return model, tokenizer
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise

# 生成回复
def generate_reply(model, tokenizer, prompt, max_length=150):
    try:
        # 构建对话格式
        text = (
            f"<|im_start|>system\n你是一个小红书风格的评论回复生成器，"
            f"擅长用刻薄、刁钻的语气回复各种评论<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        # 生成配置
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # 提取生成的回复部分
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        reply = full_text.split("<|im_start|>assistant\n")[-1]
        reply = reply.split("<|im_end|>")[0].strip()
        
        return reply
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "⚠️ 显存不足，请尝试缩短输入或减少max_length"
    except Exception as e:
        return f"生成失败: {str(e)}"

# 主交互循环
def main():
    try:
        model, tokenizer = load_model()
        
        print("\n🎀 小红书风格评论回复生成器 🎀")
        print("输入你的评论(输入'退出'结束):")
        print(f"当前设备: {model.device} | 最大长度: 2048 tokens")
        
        while True:
            try:
                # 获取用户输入（支持多行，用Ctrl+D结束）
                print("\n>>> 你的评论: (Ctrl+D结束输入)")
                user_input = []
                while True:
                    try:
                        line = input()
                        user_input.append(line)
                    except EOFError:
                        break
                user_input = "\n".join(user_input).strip()
                
                if user_input.lower() in ["退出", "exit", "quit"]:
                    print("再见！")
                    break
                    
                if not user_input:
                    print("请输入有效内容")
                    continue
                    
                # 生成回复
                print("\n🔄 生成回复中...")
                reply = generate_reply(model, tokenizer, user_input)
                
                # 美化输出
                print("\n💖 小红书风格回复:")
                print("-" * 50)
                print(reply)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n使用Ctrl+C退出")
                break
                
    finally:
        # 清理显存
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()