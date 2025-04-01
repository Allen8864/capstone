# generate_samples.py

from human_eval.data import write_jsonl, read_problems
# import openai
import requests
import json
from tqdm import tqdm
import time
import datetime
import os
import re
import argparse
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

def parse_args():
    """
    解析命令行参数，优先使用命令行参数，其次使用环境变量，最后使用默认值
    """
    parser = argparse.ArgumentParser(description='生成代码样本的脚本')
    parser.add_argument('--model', type=str, default=os.environ.get('EVAL_MODEL', 'llama3'),
                        help='模型名称（默认: 环境变量EVAL_MODEL或llama3）')
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('EVAL_TEMPERATURE', '0.8')),
                        help='生成温度（默认: 环境变量EVAL_TEMPERATURE或0.8）')
    parser.add_argument('--max-tokens', type=int, default=int(os.environ.get('EVAL_MAX_TOKENS', '1024')),
                        help='最大token数（默认: 环境变量EVAL_MAX_TOKENS或1024）')
    parser.add_argument('--top-p', type=float, default=float(os.environ.get('EVAL_TOP_P', '0.95')),
                        help='Top-P值（默认: 环境变量EVAL_TOP_P或0.95）')
    parser.add_argument('--num-samples', type=int, default=int(os.environ.get('EVAL_NUM_SAMPLES', '1')),
                        help='每个问题生成的样本数（默认: 环境变量EVAL_NUM_SAMPLES或1）')
    parser.add_argument('--prompt-level', type=int, default=int(os.environ.get('EVAL_PROMPT_LEVEL', '0')),
                        help='提示级别: 0=原始提示, 1=增强提示（默认: 环境变量EVAL_PROMPT_LEVEL或1）')
    
    return parser.parse_args()

def call_ollama_model(prompt, model_name=None, temperature=None, max_tokens=None, top_p=None):
    """
    使用ollama API调用本地部署的模型
    
    Args:
        prompt: 提示文本
        model_name: 模型名称，默认为None（从环境变量获取）
        temperature: 生成温度，控制随机性
        max_tokens: 生成的最大token数
        top_p: Top-P值，控制多样性
    Returns:
        tuple: (生成的文本, prompt_tokens, completion_tokens, total_duration)
    """
    # 如果参数为None，则使用环境变量中的值
    model_name = model_name or os.environ.get('EVAL_MODEL', 'llama3')
    temperature = temperature if temperature is not None else float(os.environ.get('EVAL_TEMPERATURE', '0.7'))
    max_tokens = max_tokens or int(os.environ.get('EVAL_MAX_TOKENS', '1024'))
    
    # ollama API默认地址
    url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
    
    # 构建请求数据
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False  # 不使用流式响应
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        response.raise_for_status()  # 检查响应状态
        
        # 解析JSON响应
        result = response.json()
        
        # 获取token统计信息
        prompt_tokens = result.get("prompt_eval_count", 0)  # 提示token数
        completion_tokens = result.get("eval_count", 0)     # 补全token数
        total_duration = result.get("total_duration", 0) / 1e9   # 将纳秒转换为秒
        
        # 返回生成的文本和token统计
        return result.get("response", ""), prompt_tokens, completion_tokens, total_duration
    
    except Exception as e:
        print(f"调用ollama API时出错: {e}")
        return "", 0, 0, 0

def generate_one_completion(prompt: str, args) -> tuple:
    """
    实现代码生成逻辑
    Args:
        prompt: 问题的提示文本
        args: 命令行参数
    Returns:
        tuple: (生成的代码补全, prompt_tokens, completion_tokens)
    """
    # 为了提高代码生成质量，添加一些提示
    code_instruction = """
You are a professional Python programming assistant. 
Please write a complete and correct implementation for the function defined below.

First, analyze the problem step by step:
1. Understand the function signature and expected inputs/outputs
2. Consider edge cases and constraints
3. Plan your approach before coding
4. Implement a clean and efficient solution
5. Verify your code with the examples provided

Remember to import any necessary libraries at the beginning of your solution.
Only provide the implementation part.
Do not include any additional explanations or comments.
The function body should be indented with 4 spaces.

Here is the function definition:
"""
    
    # 根据prompt_level决定使用哪种提示
    if args.prompt_level == 0:
        # 使用原始提示
        enhanced_prompt = prompt
        prompt_type = "原始提示"
    else:
        # 使用增强提示
        enhanced_prompt = code_instruction + prompt
        prompt_type = "增强提示"
    
    print(f"使用{prompt_type}模式")

    # 调用本地部署的模型
    completion, prompt_tokens, completion_tokens, total_duration = call_ollama_model(
        prompt=enhanced_prompt,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )

    print(f"模型输出: \n {completion} \n #########################################################")

    code_block = re.search(r'```([\s\S]+?)```', completion)
      
    if code_block:
        extracted_code = code_block.group(1)
    else:
        extracted_code = completion.replace("```", "")
    if extracted_code.strip().lower().startswith("python"):
        extracted_code = extracted_code[len("Python"):].strip()

    print(f"提取的代码: \n {extracted_code} \n #########################################################")
    print(f"Token统计: 提示tokens: {prompt_tokens}, 补全tokens: {completion_tokens}, 总耗时: {total_duration}")
    
    return extracted_code, prompt_tokens, completion_tokens, total_duration
    

def main():
    # 记录开始时间
    start_time = datetime.datetime.now()
    
    # 解析命令行参数
    args = parse_args()
    
    # 读取所有问题
    problems = read_problems()
    
    # 设置每个问题生成多少个样本
    num_samples_per_task = args.num_samples
    
    print(f"读取了 {len(problems)} 个问题，每个问题生成 {num_samples_per_task} 个样本")
    print(f"使用模型: {args.model}, 温度: {args.temperature}, 最大token数: {args.max_tokens}, Top-P值: {args.top_p}")
    print(f"提示级别: {args.prompt_level} ({'原始提示' if args.prompt_level == 0 else '增强提示'})")
    
    # 使用列表来存储生成的样本
    samples = []
    
    # 创建总进度条
    total_samples = len(problems) * num_samples_per_task
    
    # 使用tqdm显示整体进度
    with tqdm(total=total_samples, desc="生成代码样本") as pbar:
        # 为每个问题生成样本
        for task_id, problem in problems.items():
            # 显示当前处理的问题ID
            pbar.set_description(f"正在处理问题 {task_id}")
            
            for i in range(num_samples_per_task):
                try:
                    # 生成代码补全
                    completion, prompt_tokens, completion_tokens, total_duration = generate_one_completion(problem["prompt"], args)
                    
                    # 添加到样本列表
                    samples.append({
                        "task_id": task_id,
                        "completion": completion,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_duration": total_duration
                    })
                    
                    # 更新进度条
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n处理问题 {task_id} 时出错: {e}")
                    # 即使出错也添加一个空的补全，以保持计数一致
                    samples.append({
                        "task_id": task_id,
                        "completion": "",
                        "error": str(e),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_duration": 0
                    })
                    pbar.update(1)
    
    # 记录结束时间
    end_time = datetime.datetime.now()
    # 计算总运行时间
    run_duration = end_time - start_time
    total_seconds = run_duration.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 计算token总消耗量
    total_prompt_tokens = sum(sample.get("prompt_tokens", 0) for sample in samples)
    total_completion_tokens = sum(sample.get("completion_tokens", 0) for sample in samples)
    total_tokens = total_prompt_tokens + total_completion_tokens
    total_duration = sum(sample.get("total_duration", 0) for sample in samples)
    # 生成时间戳
    timestamp = end_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 设置文件名（不包含时间戳，因为时间戳将用于目录名）
    samples_filename = f"samples.jsonl"
    config_filename = f"config.json"
    
    # 创建experiments目录（如果不存在）
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    
    # 创建模型子目录（如果不存在）
    model_dir = os.path.join(experiments_dir, args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 创建时间戳子目录
    timestamp_dir = os.path.join(model_dir, timestamp)
    if not os.path.exists(timestamp_dir):
        os.makedirs(timestamp_dir)
    
    # 完整的输出路径
    samples_path = os.path.join(timestamp_dir, samples_filename)
    
    # 保存生成的样本
    print(f"\n共生成 {len(samples)} 个样本，正在保存到 {samples_path}...")
    write_jsonl(samples_path, samples)
    
    # 保存配置信息
    config = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_samples": args.num_samples,
        "prompt_level": args.prompt_level,
        "prompt_type": "原始提示" if args.prompt_level == 0 else "增强提示",
        "timestamp": timestamp,
        "total_duration": total_duration,
        "total_problems": len(problems),
        "total_samples": len(samples),
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "average_prompt_tokens_per_sample": round(total_prompt_tokens / len(samples), 2) if samples else 0,
            "average_completion_tokens_per_sample": round(total_completion_tokens / len(samples), 2) if samples else 0,
            "average_total_tokens_per_sample": round(total_tokens / len(samples), 2) if samples else 0
        },
        "timing": {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": total_seconds,
            "duration_formatted": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
            "average_per_sample": round(total_seconds / len(samples), 2) if samples else 0
        },
        "eval_command": f"evaluate_functional_correctness {samples_path}"
    }
    
    config_path = os.path.join(timestamp_dir, config_filename)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("样本和配置保存完成!")
    print(f"实验结果目录: {timestamp_dir}")
    print(f"模型目录: {model_dir}")
    print(f"实验根目录: {experiments_dir}")
    print(f"总运行时间: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"平均每个样本耗时: {round(total_seconds / len(samples), 2) if samples else 0}秒")
    print(f"Token统计:")
    print(f"  - 总提示tokens: {total_prompt_tokens}")
    print(f"  - 总补全tokens: {total_completion_tokens}")
    print(f"  - 总tokens: {total_tokens}")
    print(f"  - 平均每样本提示tokens: {round(total_prompt_tokens / len(samples), 2) if samples else 0}")
    print(f"  - 平均每样本补全tokens: {round(total_completion_tokens / len(samples), 2) if samples else 0}")
    print(f"要评估生成的样本，请运行以下命令:")
    print(f"evaluate_functional_correctness {samples_path}")

if __name__ == "__main__":
    main()