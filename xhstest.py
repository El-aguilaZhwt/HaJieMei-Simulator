#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小红书风格评论生成模型测试脚本
评估指标：ROUGE、BLEU、情感一致性、生成速度
"""

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu
from textblob import TextBlob
import pandas as pd
import json
import time
from tqdm import tqdm

# 1. 评估指标计算
def calculate_metrics(generated, reference):
    # ROUGE指标
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated, reference)[0]
    
    # BLEU指标
    bleu = sentence_bleu([list(reference)], list(generated))
    
    # 情感分析
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity > 0
    
    sentiment_match = get_sentiment(generated) == get_sentiment(reference)
    
    return {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "bleu": bleu,
        "sentiment_match": sentiment_match,
    }

# 2. 加载测试数据
def load_test_data():
    with open('xiaohongshu_dialogues.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # 取10%作为测试集
    test_data = data[:len(data)//10]
    return [
        {"input": x["input"], "output": x["output"], "likes": x["metadata"]["likes"]} 
        for x in test_data
    ]

# 3. 初始化模型
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./xiaohongshu_final",
        device_map="auto",
    )
    return model, tokenizer

# 4. 生成测试函数
def generate_reply(model, tokenizer, prompt):
    full_prompt = f"<|im_start|>system\n你是一个小红书风格评论生成助手<|im_end|>\n" \
                  f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )
    gen_time = time.time() - start_time
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = generated.split("<|im_start|>assistant\n")[-1]
    
    return generated, gen_time

# 主测试流程
if __name__ == "__main__":
    # 初始化
    test_data = load_test_data()
    model, tokenizer = load_model()
    
    # 测试结果存储
    results = []
    
    # 批量测试
    print("===== 开始测试 =====")
    for item in tqdm(test_data):
        generated, gen_time = generate_reply(model, tokenizer, item["input"])
        metrics = calculate_metrics(generated, item["output"])
        
        results.append({
            "input": item["input"],
            "generated": generated,
            "reference": item["output"],
            "likes": item["likes"],
            "gen_time": gen_time,
            **metrics
        })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("test_results.csv", index=False)
    
    # 打印汇总统计
    print("\n===== 测试结果汇总 =====")
    print(f"平均ROUGE-1: {df['rouge-1'].mean():.4f}")
    print(f"平均ROUGE-2: {df['rouge-2'].mean():.4f}")
    print(f"平均BLEU: {df['bleu'].mean():.4f}")
    print(f"情感匹配率: {df['sentiment_match'].mean():.2%}")
    print(f"平均生成时间: {df['gen_time'].mean():.2f}s/条")
    print(f"高质量评论占比(ROUGE-1>0.5): {(df['rouge-1'] > 0.5).mean():.2%}")
    
    # 示例输出
    print("\n===== 示例生成 =====")
    samples = df.sample(3)
    for _, row in samples.iterrows():
        print(f"\n输入: {row['input']}")
        print(f"生成: {row['generated']}")
        print(f"参考: {row['reference']}")
        print(f"ROUGE-1: {row['rouge-1']:.3f} | 生成时间: {row['gen_time']:.2f}s")