# HumanEval: Hand-Written Evaluation Set 

This is an evaluation harness for the HumanEval problem solving dataset
described in the paper "[Evaluating Large Language Models Trained on
Code](https://arxiv.org/abs/2107.03374)".

## 安装 (Installation)

确保使用Python 3.7或更高版本:
```
$ conda create -n codex python=3.7
$ conda activate codex
```

克隆并安装此仓库:
```
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```

## 环境配置 (Environment Configuration)

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境变量
拷贝.env.example文件为.env文件，并根据需要修改其中的配置：
```bash
cp .env.example .env
```

## 使用说明 (Usage)

**此程序用于运行不受信任的模型生成代码。强烈建议用户不要在没有强大安全沙盒的情况下运行。
`execution.py`中的[执行调用](https://github.com/openai/human-eval/blob/master/human_eval/execution.py#L48-L58)
被故意注释掉，以确保用户在可能不安全的方式运行代码之前阅读此免责声明。
有关更多信息和说明，请参阅`execution.py`中的注释。**

按照上述说明启用执行后，生成样本并将其保存为以下JSON Lines（jsonl）格式，其中每个样本格式化为单行：
```
{"task_id": "对应的HumanEval任务ID", "completion": "仅完成部分，不包含提示"}
```
我们在`data`下提供了`example_problem.jsonl`和`example_solutions.jsonl`来说明格式并帮助调试。

以下是一个几乎可以运行的示例代码（您只需提供`generate_one_completion`使其工作），该代码将生成的完成保存到`samples.jsonl`。
```python
from human_eval.data import write_jsonl, read_problems

problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
```

要评估样本，请运行：
```
$ evaluate_functional_correctness samples.jsonl
Reading samples...
32800it [00:01, 23787.50it/s]
Running test suites...
100%|...| 32800/32800 [16:11<00:00, 33.76it/s]
Writing results to samples.jsonl_results.jsonl...
100%|...| 32800/32800 [00:00<00:00, 42876.84it/s]
{'pass@1': ..., 'pass@10': ..., 'pass@100': ...}
```
此脚本在以`<input_path>_results.jsonl`结尾的新文件中提供更详细的信息。每行现在包含完成是否`passed`以及执行`result`，结果可能是"passed"、"timed out"或"failed"。

作为快速验证，示例样本应产生0.5 pass@1。
```
$ evaluate_functional_correctness data/example_samples.jsonl --problem_file=data/example_problem.jsonl
Reading samples...
6it [00:00, 3397.11it/s]
Running example suites...
100%|...| 6/6 [00:03<00:00,  1.96it/s]
Writing results to data/example_samples.jsonl_results.jsonl...
100%|...| 6/6 [00:00<00:00, 6148.50it/s]
{'pass@1': 0.4999999999999999}
```

由于当样本少于k时，无法无偏地估计pass@k，因此脚本不会评估这些情况的pass@k。要使用其他k值进行评估，请传递`--k=<逗号分隔的值>`。有关其他选项，请参见：
```
$ evaluate_functional_correctness --help
```
但是，我们建议您使用其余参数的默认值。

## 已知问题 (Known Issues)

虽然评估使用很少的内存，但当系统内存不足时，您可能会看到以下错误消息。由于这可能导致一些正确的程序失败，我们建议您释放一些内存并重试。
```
malloc: can't allocate region
```

## 参数配置 (Parameter Configuration)

### 命令行参数

您可以通过命令行参数来覆盖环境变量中的配置：

```bash
python -m human_eval.generate_samples --model llama3 --temperature 0.7 --max-tokens 1500 --num-samples 1 --prompt-level 1
```

### 参数说明

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| --model | EVAL_MODEL | llama3 | 模型名称 |
| --temperature | EVAL_TEMPERATURE | 0.8 | 生成温度（越低越确定性） |
| --max-tokens | EVAL_MAX_TOKENS | 1500 | 最大生成token数 |
| --num-samples | EVAL_NUM_SAMPLES | 1 | 每个问题生成的样本数 |
| --prompt-level | EVAL_PROMPT_LEVEL | 1 | 提示级别: 0=原始提示, 1=增强提示 |

### 提示级别

程序支持两种不同级别的提示模式：

- **级别 0 (原始提示)**: 直接使用原始问题描述作为模型输入，无额外指导
- **级别 1 (增强提示)**: 在原始提示前添加详细的代码指导，帮助模型理解问题并生成高质量代码

增强提示包含以下内容：
```
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
[原始问题描述]
```

通过比较不同提示级别的效果，您可以评估指导性提示对代码生成质量的影响。

### 文件组织结构

生成的样本和配置文件将按以下结构保存：

```
experiments/
├── llama3/                           # 模型名称文件夹
│   ├── 2023-04-01_12-34-56/          # 时间戳文件夹（实验ID）
│   │   ├── samples.jsonl             # 生成的样本文件
│   │   ├── config.json               # 实验配置信息
│   │   ├── results.jsonl             # 评估结果（评估后生成）
│   │   └── metrics.json              # 评估指标（评估后生成）
│   └── 2023-04-02_10-45-30/          # 另一个时间戳文件夹
│       ├── samples.jsonl
│       └── config.json
├── qwen2.5-coder/                    # 另一个模型名称文件夹
    └── 2023-04-03_09-15-22/
        ├── samples.jsonl
        └── config.json
```

每个实验结果存储在唯一的时间戳目录中，包含以下文件：
- `samples.jsonl`: 模型生成的代码样本
- `config.json`: 实验配置信息（模型、温度等参数）
  - 包含详细的时间记录：开始时间、结束时间、总耗时、平均每个样本的耗时
- `results.jsonl`: 评估后的详细结果（可选，评估后生成）
- `metrics.json`: 评估指标摘要，如pass@k值（可选，评估后生成）
  - 包含评估过程的时间记录：开始时间、结束时间、总耗时

### 评估生成的代码

生成样本后，可以使用以下命令评估代码的正确性：

```bash
# 使用原始评估命令
evaluate_functional_correctness experiments/llama3/2023-04-01_12-34-56/samples.jsonl

# 或者使用我们的评估脚本（结果会自动保存到相应的实验目录中）
python eval_script.py experiments/llama3/2023-04-01_12-34-56/samples.jsonl --k=1,10,100
```

## API配置 (API Configuration)

默认使用本地部署的Ollama服务，API地址为`http://localhost:11434/api/generate`。如需修改，请在.env文件中设置`OLLAMA_API_URL`环境变量。

## 实验时间追踪 (Experiment Timing)

系统自动记录样本生成和评估过程的详细时间信息：

### 样本生成时间

在样本生成完成后，您会看到如下输出：
```
样本和配置保存完成!
实验结果目录: experiments/llama3/2023-04-01_12-34-56
总运行时间: 01:23:45
平均每个样本耗时: 30.25秒
```

这些信息也被保存在`config.json`文件中：
```json
{
  "model": "llama3",
  "temperature": 0.8,
  "max_tokens": 1500,
  "num_samples": 1,
  "prompt_level": 1,
  "prompt_type": "增强提示",
  "timestamp": "2023-04-01_12-34-56",
  "total_problems": 164,
  "total_samples": 164,
  "timing": {
    "start_time": "2023-04-01 12:34:56",
    "end_time": "2023-04-01 13:58:41",
    "duration_seconds": 5025.6,
    "duration_formatted": "01:23:45",
    "average_per_sample": 30.25
  }
}
```

### 评估时间

使用`eval_script.py`进行评估后，系统会显示：
```
评估总运行时间: 00:15:32
```

评估的时间信息也被保存在`metrics.json`中：
```json
{
  "metrics": {
    "pass@1": 0.375
  },
  "timing": {
    "start_time": "2023-04-01 14:00:00",
    "end_time": "2023-04-01 14:15:32",
    "duration_seconds": 932.5,
    "duration_formatted": "00:15:32"
  }
}
```

这些详细的时间记录可以帮助您：
1. 评估不同模型的性能差异
2. 优化参数设置，找到效率和质量的平衡点
3. 估算大规模评估所需的时间
4. 追踪长时间运行的实验

## 引用 (Citation)

请使用以下bibtex条目进行引用：

```
@article{chen2021codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```