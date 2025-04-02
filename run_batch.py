#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量执行代码生成命令的脚本
用法: python run_batch.py
"""

import subprocess
import itertools
import time
import os
from datetime import datetime

def run_command(cmd):
    """运行命令并打印输出"""
    print(f"\n{'='*80}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 运行命令，实时输出结果
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    
    print(f"\n{'='*80}")
    print(f"命令执行{'成功' if return_code == 0 else '失败'}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return return_code

def main():
    # 添加要测试的参数组合
    models = ["llama3"]  # 可以添加其他模型
    datasets = ["humaneval", "humanevalfix"]
    temperatures = [0.8]
    prompt_levels = [0, 1, 2, 3]  # 根据你的代码，0=原始提示，1=增强提示，2=增强提示+测试用例
    num_samples = [5]  # 每个问题生成的样本数
    
    # 创建日志目录
    log_dir = "batch_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"批量执行开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 计算总命令数
    total_commands = len(models) * len(datasets) * len(temperatures) * len(prompt_levels) * len(num_samples)
    print(f"总共将执行 {total_commands} 个命令\n")
    
    # 创建日志文件
    log_file = os.path.join(log_dir, f"batch_run_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_file, "w") as f:
        f.write(f"批量执行开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总共将执行 {total_commands} 个命令\n\n")
    
    # 计数器
    completed = 0
    failed = 0
    
    # 执行所有参数组合
    for model, dataset, temp, prompt_level, samples in itertools.product(
        models, datasets, temperatures, prompt_levels, num_samples):
        
        # 构建命令
        cmd = [
            "python", "-m", "human_eval.generate_samples",
            "--model", model,
            "--dataset", dataset,
            "--temperature", str(temp),
            "--prompt-level", str(prompt_level),
            "--num-samples", str(samples)
        ]
        
        # 执行命令
        try:
            return_code = run_command(cmd)
            
            # 记录结果
            with open(log_file, "a") as f:
                result = "成功" if return_code == 0 else "失败"
                cmd_str = " ".join(cmd)
                f.write(f"命令: {cmd_str}\n")
                f.write(f"结果: {result}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if return_code == 0:
                completed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"执行出错: {e}")
            with open(log_file, "a") as f:
                cmd_str = " ".join(cmd)
                f.write(f"命令: {cmd_str}\n")
                f.write(f"结果: 异常 - {str(e)}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            failed += 1
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 打印汇总信息
    print(f"\n{'='*80}")
    print(f"批量执行完成!")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"执行时长: {duration}")
    print(f"成功命令: {completed}/{total_commands}")
    print(f"失败命令: {failed}/{total_commands}")
    print(f"日志文件: {log_file}")
    print(f"{'='*80}\n")
    
    # 将汇总信息写入日志
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"批量执行完成!\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"执行时长: {duration}\n")
        f.write(f"成功命令: {completed}/{total_commands}\n")
        f.write(f"失败命令: {failed}/{total_commands}\n")
        f.write(f"{'='*80}\n")

if __name__ == "__main__":
    main() 