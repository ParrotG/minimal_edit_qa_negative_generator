# 新增SFT训练计划

## 1. 目标

本轮计划只围绕 grounded factuality 的 **SFT 阶段** 展开，目标是：

- 构造 answerable / unanswerable 混合训练数据
- 用 teacher 生成协议化 structured completion
- 训练能够在 infer prompt 下稳定输出 grounded QA 协议的模型
- 用统一评测流程选择最佳 ckpt，并与多个 base baseline 比较

## 2. 数据来源与 split

HotpotQA distractor subset 作为唯一原始来源：

- Hotpot `train`：
  - 仅用于 `train_sft_raw`
- Hotpot `validation`：
  - 仅用于 `validation` / `test`

`data_split` 与 `answerability_split` 保持解耦：

- `data_split`
  - `train_sft_raw`
  - `validation`
  - `test`
- `answerability_split`
  - `answerable`
  - `unanswerable`
  - `both`

默认 `answerability_split` 比例：

- `answerable = 0.8`
- `unanswerable = 0.1`
- `both = 0.1`

## 3. Source 构造

### 3.1 answerable

- 由 Hotpot gold supporting facts 构造 knowledge
- 保留问题、参考答案、supporting facts 信息
- 之后进入 teacher generate / validate

### 3.2 unanswerable

- 基于 answerable scaffold 做 supporting fact 替换
- 保持 knowledge 长度和表面分布尽量接近
- 在 source prefilter 中统一做：
  - infer prompt token 预算过滤
  - reference answer 的 NLI 判负过滤

### 3.3 source prefilter

source 阶段统一完成所有输入级过滤：

- 对所有样本做 infer prompt token 长度过滤
- 对 unanswerable 额外做 NLI prefilter

这一步不再放在 teacher generate 中。

## 4. Teacher completion

teacher 使用统一结构协议：

- `answerability`
- `evidence`
- `rationale`
- `answer`
- `confidence`

其中：

- answerable 要求最小 sufficient evidence 与最小必要推理
- unanswerable 要求：
  - `evidence=[]`
  - 合法 refusal template
  - `rationale` 说明信息缺失、歧义或冲突

teacher validate 统一接受 mixed candidates，并在内部按 `answerability_label` 分流：

- answerable：
  - protocol
  - evidence substring
  - correctness
  - semantic
  - selection / confidence
- unanswerable：
  - protocol
  - answerability match
  - completion token budget
  - 首个 hard-pass 候选

## 5. SFT 数据集

`prepare_sft_dataset` 现按三分区准备：

- `train`
  - 来自 selected SFT records
- `validation`
  - 来自 selected validation SFT records
  - 必须保留 completion 以计算 validation loss
- `test`
  - 可直接来自 source partition 的 `test.jsonl`
  - 不要求 completion

准备阶段支持：

- 多来源输入
- 各 split 最大采样量
- 各 split 内 answerable / unanswerable 最大采样量
- prompt / completion token 再过滤

## 6. 模型评测

### 6.1 validation

对 base 与所有保留 ckpt，分别进行：

- SFT loss
- structured generation
- grounded QA 结构评测

同时额外构造两类 base baseline：

- base protocol
  - teacher few-shot + retry
- base task
  - no-think
  - think

### 6.2 best ckpt 选择

先加硬约束：

- `parse_ok_rate >= 0.95`
- `protocol_ok_rate_given_parse_ok >= 0.98`
- `evidence_substring_ok_rate >= 0.95`

再按以下顺序选最佳：

1. `correctness_reviewed_rate`
2. `answerability_accuracy`
3. `semantic_yes_rate`
4. `mean_loss`
5. `model_step` 越小越优

### 6.3 test

在 test 上比较四条轨道：

- best ckpt structured
- base protocol
- base task no-think
- base task think

其中：

- structured 轨道跑 grounded eval + DeepEval structured
- task 轨道跑 task-content eval + DeepEval flat

## 7. 结果输出

推荐最终使用 `model_eval.run_sft_eval_report` 统一产出：

- validation 曲线
- merged validation curve
- best ckpt selection.json
- test 四轨道评测结果
- final_report.md

当前工程已不再包含 DPO 阶段，也不再维护任何 `meqng` / `ssqpg` 路线。
