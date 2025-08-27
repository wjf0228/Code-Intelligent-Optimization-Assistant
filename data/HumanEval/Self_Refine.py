import pdb
from typing import Iterable, Dict
import gzip
import json
import os
from tqdm import tqdm
import requests
import time
import random
import re
from transformers import GPT2Tokenizer
import chardet

HUMAN_EVAL = "HumanEval_subset.jsonl"

API_URL = "https://api.sambanova.ai/v1/chat/completions"
API_KEY = "0771b139-814c-4dd3-8156-f0abb873c4aa "

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def detect_encoding(filename):
    try:
        with open(filename, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        return 'utf-8'  # 默认编码

def stream_jsonl(filename: str) -> Iterable[Dict]:
    encoding = detect_encoding(filename)
    try:
        if filename.endswith(".gz"):
            with open(filename, "rb") as gzfp:
                with gzip.open(gzfp, 'rt', encoding=encoding) as fp:
                    for line in fp:
                        if line.strip():
                            yield json.loads(line)
        else:
            with open(filename, "r", encoding=encoding) as fp:
                for line in fp:
                    if line.strip():
                        yield json.loads(line)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        yield from ()

def extract_first_content_with_regex(s):
    pattern = r"```python([\s\S]*?)```"
    match = re.search(pattern, s)
    if match:
        return match.group(1).strip()
    return "None"

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    mode = 'ab' if append else 'wb'
    filename = os.path.expanduser(filename)
    try:
        if filename.endswith(".gz"):
            with open(filename, mode) as fp:
                with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                    for x in data:
                        gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
        else:
            with open(filename, mode) as fp:
                for x in data:
                    fp.write((json.dumps(x) + "\n").encode('utf-8'))
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")

def read_problems(evalset_file: str = "HumanEval_subset.jsonl") -> Dict[str, Dict]:
    try:
        return {task["task_id"]: task for task in stream_jsonl(evalset_file)}
    except Exception as e:
        print(f"Error reading problems: {e}")
        return {}

def generate_code(prompt: str, max_tokens=400):
    """
    使用 AI 模型生成代码。
    返回生成的代码和其他元数据（如生成时间和 token 数）。
    """
    print(f"Generating code with prompt: {prompt[:100]}...")  # 打印前100个字符
    data = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a code assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_k": 50
    }

    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    try:
        response_data = response.json()
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response: {response.text}")
        return None

    generated_code = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return generated_code

def generate_feedback(prompt: str, code: str, max_tokens=200):
    """
    生成对初始生成代码的反馈。
    """
    feedback_prompt = f"Here is the generated code:\n{code}\nPlease provide feedback for improvement or correctness."
    return generate_code(feedback_prompt, max_tokens)

def generate_samples(problems: Dict[str, Dict], output_file: str = "self_refine.jsonl"):
    samples = []
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    for task_id, task in tqdm(problems.items()):
        input_data = task.get("prompt", "")
        prompt = "Solve the following problem:\n" + input_data

        # 生成初始代码
        initial_code = generate_code(prompt=prompt)
        if not initial_code:
            print(f"Error: No initial code generated for task {task_id}")
            continue

        # SELF-REFINE 过程：进行多轮迭代改进
        refined_code = initial_code
        feedback = ""
        for _ in range(3):  # 假设最多迭代3次
            feedback = generate_feedback(prompt, refined_code)
            if not feedback or "No improvements needed" in feedback:
                break  # 假设模型反馈中含有停止信号

            refined_code = generate_code(f"Improve this code based on the feedback: {feedback}")
            if not refined_code:
                print(f"Error: No refined code generated for task {task_id}")
                break

        # 检查 refined_code 是否有效，然后提取结果
        if refined_code is None:
            print(f"Error: No refined code generated for task {task_id}")
            result = "No valid output"
        else:
            result = extract_first_content_with_regex(refined_code)

        # 将最终结果存储在样本中，包含所有需要的字段
        sample = {
            "task_id": task_id,
            "input": input_data,
            "prompt": prompt,
            "output": result  # 使用 "output" 作为最终生成的代码
        }
        samples.append(sample)

    # 将所有样本写入 JSONL 文件
    write_jsonl(output_file, samples)

def read_results_file(results_file: str) -> Iterable[Dict]:
    """
    读取并解析 JSONL 格式的结果文件
    """
    try:
        with open(results_file, "r", encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        yield from ()

def analyze_results(results_file: str = "baseline.jsonl_results.jsonl"):
    """
    读取并分析结果文件内容
    """
    results = read_results_file(results_file)
    for result in results:
        print(f"Task ID: {result.get('task_id')}")
        print(f"Input: {result.get('input')}")
        print(f"Output: {result.get('output')}")
        print(f"Score: {result.get('score')}")
        print("-" * 20)

# Main execution
if __name__ == "__main__":
    problems = read_problems()
    generate_samples(problems, "self_refine.jsonl")
    print("Finish")
    analyze_results("baseline.jsonl_results.jsonl")
