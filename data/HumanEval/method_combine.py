import json
import time
import requests

# 设置 API 端点和密钥
API_URL = "https://api.sambanova.ai/v1/chat/completions"
API_KEY = "1961dcb0-57db-48cc-a0b1-c2841aebc490"

# 准备请求数据
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_code(prompt: str, max_tokens=400):
    """
    使用模型生成代码。
    返回生成的代码、生成时间和生成的总 token 数。
    """
    data = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system",
             "content": 'You are a math assistant. Solve each problem step-by-step, identify all key numbers, perform the necessary calculations, and provide the final answer in the format: #### [answer]. Include test cases to verify the function.'},
            {"role": "user",
             "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_k": 50
    }

    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=data)
    end_time = time.time()

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    generated_tokens = response.json().get("usage", {}).get("completion_tokens", 0)
    generation_time = end_time - start_time

    return response.json()["choices"][0]["message"]["content"].strip(), generation_time, generated_tokens


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
    debug_prompt = f"The following code has an error:\n{error_message}\n\nPlease explain the original code step by step. Then fix the bug and provide full codes. The original code:\n\n{code}\n\n"
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


def correct_baseline_samples(baseline_file="samples_baseline.jsonl_results.jsonl", output_file="combine_corrected_samples.jsonl", debug_file="samples_method_combine.jsonl", max_attempts=3):
    """
    纠正 baseline 中生成但未通过测试的样例。
    从 baseline_file 中读取数据，对未通过测试的样例进行修正，保存修正后的样例。
    """
    with open(baseline_file, "r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)  # 读取每一行并解析为 JSON 对象
            task_id = sample.get("task_id")
            prompt = sample.get("prompt")
            test_cases = sample.get("input", "")  # 从样例中获取测试用例，如果没有则为空字符串
            generated_code = sample.get("output")

            if not task_id or not prompt or not generated_code:
                print(f"Skipping invalid entry in {baseline_file}: {sample}")
                continue

            # 如果测试用例为空，则生成默认的测试用例
            if not test_cases:
                test_cases = "# TODO: Add test cases here.\n"

            # 修复潜在的未结束字符串问题
            generated_code = fix_unterminated_string_literals(generated_code)

            for attempt in range(max_attempts):
                print(f"Attempt {attempt + 1} to correct task_id {task_id}...")

                # 保存修正尝试结果到调试文件
                save_samples_to_jsonl(task_id, test_cases, prompt, generated_code, debug_file)

                # 运行测试用例
                result = run_tests(generated_code, test_cases)
                if "All tests passed" in result:
                    print(f"Code for task_id {task_id} passed all tests after correction.")
                    save_samples_to_jsonl(task_id, test_cases, prompt, generated_code, output_file)
                    break
                else:
                    print(f"Error in task_id {task_id}: {result}")
                    # 调用 debug_code 生成调试后的代码
                    debug_result = debug_code(prompt, generated_code, result)
                    generated_code = fix_unterminated_string_literals(debug_result[0])

            else:
                print(f"Failed to correct code for task_id {task_id} after {max_attempts} attempts.")


def save_samples_to_jsonl(task_id, input_data, prompt, final_code, output_file="samples_method_combine.jsonl"):
    """
    保存生成的代码到 JSONL 文件中。
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

    print(f"Saving task_id: {task_id} with generated code and input data.")

    try:
        with open(output_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Successfully saved sample for task_id: {task_id}.")
    except Exception as e:
        print(f"Failed to save sample for task_id: {task_id}. Error: {str(e)}")


# 调用来纠正 baseline 样例
correct_baseline_samples("samples_baseline.jsonl_results.jsonl", "../GSM8K/combine_corrected_samples.jsonl", "samples_method_combine.jsonl", max_attempts=3)
