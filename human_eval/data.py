from typing import Iterable, Dict
import gzip
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "HumanEval.jsonl.gz")
HUMAN_EVAL_FIX = os.path.join(ROOT, "..", "data", "humaneval-fix-starcoder.jsonl")


def read_problems(evalset_file: str = HUMAN_EVAL, dataset: str = "humaneval") -> Dict[str, Dict]:
    """
    读取问题数据集
    Args:
        evalset_file: 数据集文件路径
        dataset: 数据集名称，可选值为 "humaneval" 或 "humanevalfix"
    Returns:
        Dict[str, Dict]: 问题字典
    """
    if dataset == "humanevalfix":
        evalset_file = HUMAN_EVAL_FIX
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
