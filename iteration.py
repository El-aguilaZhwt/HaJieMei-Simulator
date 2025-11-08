#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小红书风格评论回复生成器（连续对话版）
基于LoRA微调的Qwen3-14B模型
"""

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
import readline
import re

def load_model():
    print("正在加载模型...")
    try:
        # 加载基础模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/home/xiuyu/models/Qwen3-14B-unsloth-bnb-4bit",
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
        )
        
        # 添加特殊token
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        
        # 加载LoRA适配器
        model.load_adapter("/home/xiuyu/qwen3train/xiaohongshu_train/checkpoint-6492")
        
        model.config.use_cache = True
        model.config.pretraining_tp = 1
        
        print(f"模型加载完成！词汇表大小: {len(tokenizer)}")
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise

def generate_reply(model, tokenizer, conversation_history, max_length=150):
    try:
        # 构建完整对话上下文
        system_prompt = "<|im_start|>system\n你是一个小红书风格的评论回复生成器，擅长用刻薄、刁钻的语气回复各种评论<|im_end|>\n"
        full_context = system_prompt + "\n".join(conversation_history)
        
        # 确保上下文不超过模型最大长度
        inputs = tokenizer(full_context, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        reply = full_text.split("<|im_start|>assistant\n")[-1]
        reply = reply.split("<|im_end|>")[0].strip()
        
        # 后处理：确保回复包含文字内容
        reply = ensure_text_content(reply)
        return reply
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "⚠️ 显存不足，请尝试缩短输入"
    except Exception as e:
        return f"生成失败: {str(e)}"

def ensure_text_content(text):
    """确保回复包含实质性文字内容"""
    # 如果只有表情符号
    if re.fullmatch(r'($$[^]]+$$\s*)+', text):
        additions = [
            "这波操作我给满分！",
            "姐妹说得太对了！",
            "我真的会谢！",
            "这简直是我的互联网嘴替！",
            "笑不活了家人们！"
        ]
        return f"{text} {additions[len(text) % len(additions)]}"
    return text

def main():
    try:
        model, tokenizer = load_model()
        
        print("\n🎀 小红书风格评论回复生成器（连续对话版） 🎀")
        print("输入初始评论和迭代次数（例如：'你好 3' 表示生成3层回复）")
        
        while True:
            try:
                # 获取用户输入
                print("\n>>> 初始评论+迭代次数 (用空格分隔):")
                user_input = input().strip()
                
                if user_input.lower() in ["退出", "exit", "quit"]:
                    print("再见！")
                    break
                    
                if not user_input:
                    print("请输入有效内容")
                    continue
                
                # 解析输入
                parts = user_input.split()
                if len(parts) < 2:
                    initial_comment = user_input
                    iterations = 1
                else:
                    try:
                        initial_comment = " ".join(parts[:-1])
                        iterations = min(int(parts[-1]), 35)  # 限制最多5次迭代
                    except:
                        initial_comment = user_input
                        iterations = 1
                
                # 初始化对话历史
                conversation_history = [
                    f"<|im_start|>user\n{initial_comment}<|im_end|>"
                ]
                
                print(f"\n💬 初始评论: {initial_comment}")
                print(f"🔄 将生成 {iterations} 层回复...\n")
                
                # 生成连续回复
                for i in range(iterations):
                    print(f"\n🔄 正在生成第 {i+1} 层回复...")
                    reply = generate_reply(model, tokenizer, conversation_history)
                    
                    # 添加到对话历史
                    conversation_history.append(
                        f"<|im_start|>assistant\n{reply}<|im_end|>\n"
                        f"<|im_start|>user\n继续这个话题<|im_end|>"
                    )
                    
                    # 打印回复
                    print(f"\n📌 第 {i+1} 层回复:")
                    print("-" * 50)
                    print(reply)
                    print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n使用Ctrl+C退出")
                break
                
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()