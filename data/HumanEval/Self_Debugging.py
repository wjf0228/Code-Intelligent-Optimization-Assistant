import json
import time
import requests
import os

# 设置 API 端点和密钥
API_URL = "https://api.sambanova.ai/v1/chat/completions"
API_KEY = "0771b139-814c-4dd3-8156-f0abb873c4aa"

# 准备请求数据
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_code(prompt: str, max_tokens=400):
    """
    使用模型生成代码。
    返回生成的代码、生成时间和生成的总 token 数。
    如果遇到速率限制，会自动等待并重试。
    """
    data = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are provided with a piece of Python code that has a bug. Your task is to carefully examine the code, identify the bug, and provide a corrected version of the code. Ensure to explain the cause of the bug and how it was fixed."},
            {"role": "user","content": "Write a function that takes a list of integer transactions where each positive integer represents a deposit and each negative integer represents a withdrawal. The account balance starts at zero.The function should perform the following steps:1.Initialize a variable to keep track of the current balance, starting at zero.2.Iterate through each transaction in the list.3.For each transaction:4.If the transaction is positive, add it to the current balance.5.If the transaction is negative, subtract it from the current balance.6.After updating the balance with each transaction, check if the balance falls below zero.7.If the balance goes below zero at any point, return True.8.If all transactions are processed and the balance never goes below zero, return False."}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_k": 50
    }

    max_retries = 5  # 最大重试次数
    retry_delay = 5  # 默认等待时间（秒）

    for attempt in range(max_retries):
        start_time = time.time()
        response = requests.post(API_URL, headers=headers, json=data)
        end_time = time.time()

        if response.status_code == 200:
            # 请求成功，返回结果
            generated_tokens = response.json().get("usage", {}).get("completion_tokens", 0)
            generation_time = end_time - start_time
            return response.json()["choices"][0]["message"]["content"].strip(), generation_time, generated_tokens

        elif response.status_code == 429:
            # 处理速率限制
            retry_after = int(response.headers.get("Retry-After", retry_delay))
            print(f"Rate limit exceeded. Retrying after {retry_after} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_after)
        else:
            # 非速率限制的其他错误
            raise Exception(f"Error {response.status_code}: {response.text}")

    # 达到最大重试次数仍未成功
    raise Exception("Max retries reached. Unable to generate code.")


def run_tests(code: str, test_cases: str):
    """
    运行生成的代码和预定义的测试用例，返回测试结果。
    """
    full_code = code + "\n" + test_cases

    try:
        exec(full_code, {})
        return "All tests passed"
    except Exception as e:
        return str(e)


def debug_code(prompt: str, code: str, error_message: str):
    """
    利用错误信息，让模型修正代码。
    返回修正后的代码、修正时间和生成的 token 数。
    """
    debug_prompt = f"The following code has an error:\n\n{code}\n\nError message:\n{error_message}\n\nPlease fix the code and provide test cases."
    return generate_code(debug_prompt)


def fix_unterminated_string_literals(code: str) -> str:
    """
    检查并修复生成代码中未结束的字符串字面量。
    """
    if code.count('"') % 2 != 0:
        code += '"'
    if code.count("'") % 2 != 0:
        code += "'"
    if code.count("'''") % 2 != 0:
        code += "'''"
    if code.count('"""') % 2 != 0:
        code += '"""'

    return code


def correct_baseline_samples(baseline_file="baseline.jsonl_results.jsonl", output_file="self_debugging.jsonl",
                             debug_file="debug_self_debugging.jsonl", max_attempts=3):
    """
    纠正 baseline 中生成但未通过测试的样例。
    """
    total_tokens = 0  # 用于记录生成的总 token 数量
    num_problems = 0  # 用于记录问题的总数
    start_time = time.time()  # 记录开始时间

    with open(baseline_file, "r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            task_id = sample.get("task_id")
            prompt = sample.get("prompt")
            test_cases = sample.get("input", "")
            generated_code = sample.get("output")
            result = sample.get("result", "")
            passed = sample.get("passed", False)

            if not task_id or not prompt or not generated_code:
                print(f"Skipping invalid entry in {baseline_file}: {sample}")
                continue

            if not test_cases:
                test_cases = "# TODO: Add test cases here.\n"

            if result == "passed" or passed is True:
                print(f"Task {task_id} already passed in baseline. Skipping...")
                save_samples_to_jsonl(task_id, test_cases, prompt, generated_code, output_file)
                continue

            generated_code = fix_unterminated_string_literals(generated_code)
            final_generated_code = None

            for attempt in range(max_attempts):
                print(f"Attempt {attempt + 1} to correct task_id {task_id}...")
                save_samples_to_jsonl(task_id, test_cases, prompt, generated_code, debug_file)
                result = run_tests(generated_code, test_cases)

                if "All tests passed" in result:
                    print(f"Task {task_id} passed after {attempt + 1} attempt(s).")
                    final_generated_code = generated_code
                    break
                else:
                    print(f"Task {task_id} failed on attempt {attempt + 1}: {result}")
                    debug_result = debug_code(prompt, generated_code, result)
                    generated_code = fix_unterminated_string_literals(debug_result[0])

            # 保存失败的代码，确保即使失败，也会保存到 output_file
            if final_generated_code:
                print(f"Saving final generated code for task {task_id}")
                save_samples_to_jsonl(task_id, test_cases, prompt, final_generated_code, output_file)
            else:
                print(f"Task {task_id} could not be corrected after {max_attempts} attempts.")
                save_samples_to_jsonl(task_id, test_cases, prompt, generated_code, output_file)

            # 计算总的 tokens 数量
            num_problems += 1
            total_tokens += generated_code.count(" ")  # 将生成的代码中的 token 数加入总数

    end_time = time.time()  # 结束计时
    wall_clock_time = end_time - start_time  # 总的壁钟时间

    # 打印总的 wall-clock time 和平均的 token 数量
    average_tokens = total_tokens / num_problems if num_problems > 0 else 0
    print(f"Total wall-clock time: {wall_clock_time:.2f} seconds")
    print(f"Average number of generated tokens per problem: {average_tokens:.2f}")


def save_samples_to_jsonl(task_id, input_data, prompt, final_code, output_file="self_debugging.jsonl"):
    """
    保存生成的代码到 JSONL 文件中，确保每个 task_id 只保存最后一次成功的代码。
    """
    if not final_code.strip():
        print(f"Skipping save for task_id: {task_id} due to empty code.")
        return

    sample = {
        "task_id": task_id,
        "input": input_data,
        "prompt": prompt,
        "output": final_code
    }

    # 调试输出：检查是否已保存相同的 task_id
    print(f"Checking if task_id {task_id} is already saved...")

    # 检查输出文件是否存在
    if os.path.exists(output_file):
        # 如果文件存在，读取已有内容并检查是否有重复的 task_id
        with open(output_file, "r", encoding="utf-8") as file:
            existing_samples = file.readlines()

        # 遍历已有样本，检查是否有相同的 task_id
        for line in existing_samples:
            existing_sample = json.loads(line)
            if existing_sample["task_id"] == task_id:
                print(f"task_id {task_id} already exists in {output_file}. Skipping save.")
                return

    # 如果没有重复的 task_id，才保存新的样本
    try:
        with open(output_file, "a", encoding="utf-8") as file:
            json.dump(sample, file, ensure_ascii=False)
            file.write("\n")
        print(f"Saved task_id {task_id} to {output_file}")
    except Exception as e:
        print(f"Error saving sample for task_id {task_id}: {e}")




# 调用来纠正 baseline 样例
correct_baseline_samples("baseline.jsonl_results.jsonl", "self_debugging.jsonl", "debug_self_debugging.jsonl", max_attempts=3)
