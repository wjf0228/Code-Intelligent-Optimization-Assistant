# Code Intelligent Optimization Assistant

## 项目简介

Code Intelligent Optimization Assistant 是一个基于AI的代码生成和优化工具，专门用于解决HumanEval数据集中的编程问题。该项目实现了多种代码生成和优化策略，包括基线方法、自调试、自优化和组合方法等。

## 项目结构

```
data/HumanEval/
├── __pycache__/                    # Python缓存文件
├── baseline.py                     # 基线代码生成方法
├── Self_Debugging.py              # 自调试代码优化方法
├── Self_Refine.py                 # 自优化代码改进方法
├── method_combine.py              # 组合多种优化方法
├── execution.py                   # 代码执行和测试框架
├── evaluation.py                  # 功能正确性评估
├── evaluate_functional_correctness.py  # 功能正确性评估脚本
├── HumanEval.jsonl               # 完整HumanEval数据集
├── HumanEval_subset.jsonl        # HumanEval数据集子集
├── baseline.jsonl                 # 基线方法生成结果
├── baseline.jsonl_results.jsonl  # 基线方法评估结果
├── samples_baseline.jsonl        # 基线方法样本
├── samples_baseline.jsonl_results.jsonl  # 基线方法样本评估结果
├── self_debugging.jsonl          # 自调试方法生成结果
├── debug_self_debugging.jsonl    # 自调试调试过程记录
├── self_debugging.jsonl_results.jsonl  # 自调试方法评估结果
├── samples_self_debugging.jsonl  # 自调试方法样本
├── samples_self_debugging.jsonl_results.jsonl  # 自调试方法样本评估结果
├── self_refine.jsonl             # 自优化方法生成结果
├── samples_self_refine.jsonl     # 自优化方法样本
├── samples_self_refine.jsonl_results.jsonl  # 自优化方法样本评估结果
├── samples_method_combine.jsonl  # 组合方法样本
├── samples_method_combine.jsonl_results.jsonl  # 组合方法样本评估结果
└── Self_Debugging.py             # 自调试实现
```

## 核心功能

### 1. 基线代码生成 (Baseline)
- 使用AI模型直接生成代码
- 支持多种编程语言和问题类型
- 自动处理API速率限制和重试机制

### 2. 自调试优化 (Self-Debugging)
- 自动检测代码中的错误
- 基于错误信息进行代码修正
- 支持多轮调试和优化

### 3. 自优化改进 (Self-Refine)
- 代码质量自动评估
- 迭代式代码改进
- 性能优化建议

### 4. 组合优化方法 (Method Combine)
- 结合多种优化策略
- 智能选择最佳优化方案
- 提高代码生成成功率

## 技术特点

- **AI模型集成**: 使用Meta-Llama-3.1-8B-Instruct模型进行代码生成
- **自动测试**: 内置测试框架，自动验证代码正确性
- **错误处理**: 智能错误检测和修复机制
- **性能评估**: 支持pass@k等评估指标
- **多进程执行**: 支持并发代码执行和测试

## 使用方法

### 环境要求
- Python 3.7+
- 必要的Python包：requests, tqdm, transformers, numpy等

### 基本使用

1. **基线代码生成**:
```python
from baseline import generate_code
code, time, tokens = generate_code(prompt)
```

2. **自调试优化**:
```python
from Self_Debugging import correct_baseline_samples
corrected_samples = correct_baseline_samples()
```

3. **自优化改进**:
```python
from Self_Refine import refine_code
improved_code = refine_code(original_code)
```

4. **组合方法**:
```python
from method_combine import combine_optimization
result = combine_optimization(code, problem)
```

### 评估代码质量

```python
from evaluation import evaluate_functional_correctness
results = evaluate_functional_correctness("samples.jsonl")
```

## 数据集说明

项目使用HumanEval数据集，包含164个编程问题，涵盖：
- 算法实现
- 数据结构操作
- 字符串处理
- 数学计算
- 逻辑推理

每个问题包含：
- 问题描述和示例
- 函数签名
- 测试用例
- 标准解答

## 性能指标

- **pass@1**: 单次生成代码的通过率
- **pass@10**: 10次生成中至少一次通过的比率
- **pass@100**: 100次生成中至少一次通过的比率

## 配置说明

### API配置
项目使用SambaNova AI API，需要在相应文件中配置：
- API端点
- API密钥
- 模型参数

### 执行参数
- 超时设置
- 并发工作进程数
- 最大重试次数
- Token数量限制


