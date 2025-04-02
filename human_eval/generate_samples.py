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
import logging
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
    parser.add_argument('--dataset', type=str, default=os.environ.get('EVAL_DATASET', 'humaneval'),
                        choices=['humaneval', 'humanevalfix'],
                        help='数据集选择: humaneval或humanevalfix（默认: 环境变量EVAL_DATASET或humaneval）')
    
    return parser.parse_args()

def setup_logging(log_dir):
    """
    设置日志记录
    Args:
        log_dir: 日志保存目录
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 设置日志文件路径
    log_file = os.path.join(log_dir, 'generation.log')
    
    # 配置logger
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 创建文件处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器，用于在控制台显示日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

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
    logger = logging.getLogger()
    
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
        logger.error(f"调用ollama API时出错: {e}")
        return "", 0, 0, 0

def generate_one_completion(problem, args) -> tuple:
    """
    实现代码生成逻辑
    Args:
        problem: 问题字典
        args: 命令行参数
    Returns:
        tuple: (生成的代码补全, prompt_tokens, completion_tokens)
    """
    logger = logging.getLogger()
    
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
        enhanced_prompt = problem["prompt"]
        prompt_type = "原始提示"
    elif args.prompt_level == 1:
        # 使用增强提示
        enhanced_prompt = code_instruction + problem["prompt"]
        prompt_type = "增强提示"
    elif args.prompt_level == 2:
        enhanced_prompt = code_instruction + problem["prompt"] + "Here is the test case:\n" + problem["test"]
        prompt_type = "增强提示+测试用例"
    elif args.prompt_level == 3:
        enhanced_prompt = code_instruction + problem["prompt"] + "Here is the canonical solution:\n" + problem["canonical_solution"]
        prompt_type = "增强提示+标准答案"
    
    logger.info(f"使用{prompt_type}模式")
    logger.info(f"模型: {args.model} \t 温度: {args.temperature} \t 最大token数: {args.max_tokens} \t Top-P值: {args.top_p}")
    logger.debug(f"提示词: \n {enhanced_prompt} \n #########################################################")

    # 调用本地部署的模型
    completion, prompt_tokens, completion_tokens, total_duration = call_ollama_model(
        prompt=enhanced_prompt,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )

    logger.info(f"模型输出: \n {completion} \n #########################################################")

    code_block = re.search(r'```([\s\S]+?)```', completion)
      
    if code_block:
        extracted_code = code_block.group(1)
    else:
        extracted_code = completion.replace("```", "")
    if extracted_code.strip().lower().startswith("python"):
        extracted_code = extracted_code[len("Python"):].strip()

    logger.info(f"提取的代码: \n {extracted_code} \n #########################################################")
    logger.info(f"Token统计: 提示tokens: {prompt_tokens}, 补全tokens: {completion_tokens}, 总耗时: {total_duration}")
    
    return extracted_code, prompt_tokens, completion_tokens, total_duration
    

def main():
    # 记录开始时间
    start_time = datetime.datetime.now()
    
    # 解析命令行参数
    args = parse_args()
    
    # 读取问题
    problems = read_problems(dataset=args.dataset)
    
    # 设置每个问题生成多少个样本
    num_samples_per_task = args.num_samples
    
    # 生成时间戳(提前生成，用于设置日志路径)
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 创建实验目录结构
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    
    # 创建数据集子目录
    dataset_dir = os.path.join(experiments_dir, args.dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # 创建模型子目录
    model_dir = os.path.join(dataset_dir, args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 创建prompt level子目录
    prompt_level_dir = os.path.join(model_dir, f"prompt_level_{args.prompt_level}")
    if not os.path.exists(prompt_level_dir):
        os.makedirs(prompt_level_dir)
    
    # 创建温度和top_p组合的子目录 - 将小数点替换为下划线
    temp_str = str(args.temperature).replace('.', '_')
    top_p_str = str(args.top_p).replace('.', '_')
    temp_top_p_dir = os.path.join(prompt_level_dir, f"temp_{temp_str}_top_p_{top_p_str}")
    if not os.path.exists(temp_top_p_dir):
        os.makedirs(temp_top_p_dir)
    
    # 创建时间戳子目录
    timestamp_dir = os.path.join(temp_top_p_dir, timestamp)
    if not os.path.exists(timestamp_dir):
        os.makedirs(timestamp_dir)
    
    # 设置日志记录
    logger = setup_logging(timestamp_dir)
    
    logger.info(f"使用数据集: {args.dataset}")
    logger.info(f"读取了 {len(problems)} 个问题，每个问题生成 {num_samples_per_task} 个样本")
    logger.info(f"使用模型: {args.model}, 温度: {args.temperature}, 最大token数: {args.max_tokens}, Top-P值: {args.top_p}")
    logger.info(f"提示级别: {args.prompt_level} ({'原始提示' if args.prompt_level == 0 else '增强提示'})")
    
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
                    completion, prompt_tokens, completion_tokens, total_duration = generate_one_completion(problem, args)
                    
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
                    logger.error(f"\n处理问题 {task_id} 时出错: {e}")
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
    
    # 设置文件名（不包含时间戳，因为时间戳将用于目录名）
    samples_filename = f"samples.jsonl"
    config_filename = f"config.json"
    
    # 完整的输出路径
    samples_path = os.path.join(timestamp_dir, samples_filename)
    
    # 保存生成的样本
    logger.info(f"\n共生成 {len(samples)} 个样本，正在保存到 {samples_path}...")
    write_jsonl(samples_path, samples)
    
    # 添加代码，自动执行evaluate_functional_correctness命令
    logger.info(f"自动执行功能正确性评估...")
    try:
        from human_eval.evaluate_functional_correctness import entry_point as evaluate_entry_point
        logger.info(f"正在评估样本: {samples_path}")
        # 调用evaluate_functional_correctness的entry_point函数
        evaluate_entry_point(samples_path)
        logger.info(f"评估完成!")
    except Exception as e:
        logger.error(f"执行评估时出错: {e}")
        
    # 保存配置信息
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "num_samples": args.num_samples,
        "prompt_level": args.prompt_level,
        "prompt_type": "原始提示" if args.prompt_level == 0 else ("增强提示" if args.prompt_level == 1 else ("增强提示+测试用例" if args.prompt_level == 2 else "增强提示+标准答案")),
        "timestamp": timestamp,
        "paths": {
            "dataset_dir": dataset_dir,
            "model_dir": model_dir,
            "prompt_level_dir": prompt_level_dir,
            "temp_top_p_dir": temp_top_p_dir,
            "timestamp_dir": timestamp_dir,
            "experiments_dir": experiments_dir,
            "samples_path": samples_path,
            "config_path": os.path.join(timestamp_dir, config_filename)
        },
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
    
    logger.info("样本和配置保存完成!")
    logger.info(f"实验结果目录: {timestamp_dir}")
    logger.info(f"温度和top_p组合目录: {temp_top_p_dir}")
    logger.info(f"提示级别目录: {prompt_level_dir}")
    logger.info(f"模型目录: {model_dir}")
    logger.info(f"数据集目录: {dataset_dir}")
    logger.info(f"实验根目录: {experiments_dir}")
    logger.info(f"总运行时间: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    logger.info(f"平均每个样本耗时: {round(total_seconds / len(samples), 2) if samples else 0}秒")
    logger.info(f"Token统计:")
    logger.info(f"  - 总提示tokens: {total_prompt_tokens}")
    logger.info(f"  - 总补全tokens: {total_completion_tokens}")
    logger.info(f"  - 总tokens: {total_tokens}")
    logger.info(f"  - 平均每样本提示tokens: {round(total_prompt_tokens / len(samples), 2) if samples else 0}")
    logger.info(f"  - 平均每样本补全tokens: {round(total_completion_tokens / len(samples), 2) if samples else 0}")
    logger.info(f"要评估生成的样本，请运行以下命令:")
    logger.info(f"evaluate_functional_correctness {samples_path}")

if __name__ == "__main__":
    main()