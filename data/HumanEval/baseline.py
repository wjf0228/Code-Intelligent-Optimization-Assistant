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
from transformers import GPT2Tokenizer  # 用于计算tokens数量
import chardet

HUMAN_EVAL = "HumanEval_subset.jsonl"


API_URL = "https://api.sambanova.ai/v1/chat/completions"
API_KEY = "0771b139-814c-4dd3-8156-f0abb873c4aa"

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
    """
    Parses each jsonl line and yields it as a dictionary
    """
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
    """
    Writes an iterable of dictionaries to jsonl
    """
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
    """
    Reads problems from a JSONL file and returns a dictionary with task_id as keys.
    """
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

    start_time = time.time()  # 记录生成的开始时间
    response = requests.post(API_URL, headers=headers, json=data)
    end_time = time.time()  # 记录生成的结束时间

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    response_data = response.json()
    if "choices" not in response_data or not response_data["choices"]:
        print("Error: No choices found in the API response.")
        return None

    generated_tokens = response_data.get("usage", {}).get("completion_tokens", 0)
    generation_time = end_time - start_time

    print(f"Code generated successfully in {generation_time:.2f} seconds with {generated_tokens} tokens.")
    return response_data["choices"][0]["message"]["content"].strip(), generation_time, generated_tokens

def generate_samples(problems: Dict[str, Dict], output_file: str = "samples_baseline.jsonl"):
    samples = []
    total_tokens = 0  # 用于记录所有生成的 tokens 数量
    num_problems = len(problems)

    # 使用 GPT2 tokenizer 进行 token 计算
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    start_time = time.time()  # 开始计时

    for task_id, task in tqdm(problems.items()):
        input_data = task.get("prompt", "")
        prompt = "Solve the following problem:\n" + input_data
        completion_code = generate_code(prompt=prompt)

        if completion_code is None or completion_code[0] is None:
            print(f"Error: Failed to generate code for task {task_id}")
            continue

        result = extract_first_content_with_regex(completion_code[0])

        # 计算当前生成的 tokens 数量
        tokens_count = len(tokenizer.encode(result)) if result != "None" else 0
        total_tokens += tokens_count

        sample = {
            "task_id": task_id,
            "input": input_data,
            "prompt": prompt,
            "output": result  # 修改字段名称为 output
        }
        print("##" * 10)
        print(result)
        samples.append(sample)
        sleep_time = random.uniform(1, 5)
        time.sleep(sleep_time)

    end_time = time.time()  # 结束计时
    wall_clock_time = end_time - start_time

    write_jsonl(output_file, samples)

    # 计算平均 tokens 数量
    average_tokens = total_tokens / num_problems if num_problems > 0 else 0

    print(f"Total wall-clock time: {wall_clock_time:.2f} seconds")
    print(f"Average number of generated tokens per problem: {average_tokens:.2f}")

# Main execution
if __name__ == "__main__":
    problems = read_problems()
    generate_samples(problems, "baseline.jsonl")
    print("Finish")
