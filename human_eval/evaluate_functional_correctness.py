import fire
import sys
import json
import os

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 10,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    sample_dir= sample_file.rsplit('/', 1)[0]
    config_path = os.path.join(sample_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 添加评估指标
            config["evaluation_results"] = results
            
            # 保存更新后的配置
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print(f"配置文件已更新评估指标")
        except Exception as e:
            print(f"更新配置文件时出错: {e}")


def main():
    fire.Fire(entry_point)


sys.exit(main())
