目标:

提升模型在“给定 knowledge 回答 question”任务中的：
1. **Protocol compliance**
2. **Evidence fidelity**
3. **Grounded correctness**
4. **Calibration**

* * *

## 总体方法

1. **SFT**：先学习结构化 grounded-QA 协议与拒答行为
2. **DPO**：在相同协议下学习偏好边界

* * *

## 输出协议

```json
{
  "answerability": "answerable | unanswerable",
  "evidence": [{"quote": "..."}],
  "rationale": "...",
  "answer": "...",
  "confidence": "high | medium | low"
}
```

核心约束：

- `answerable`
  - `evidence` 至少 1 条
  - `rationale` 只做最小必要推理
  - `answer` 保持自然分布
- `unanswerable`
  - `evidence = []`
  - `answer` 必须为允许模板之一
  - `rationale` 必须指出缺失、歧义或矛盾

* * *

## 数据组织：双标签

在原始 Hotpot 行上先打两个独立标签：

- `data_split`
  - `validation / test / train_sft_raw / train_dpo_raw`
- `answerability_split`
  - `answerable / unanswerable / both`

默认比例：

- `answerable_ratio = 0.8`
- `unanswerable_ratio = 0.1`
- `both_ratio = 0.1`

其中 `both` 表示同一 raw source 同时产生一条 `answerable` concrete 样本和一条 `unanswerable` concrete 样本。

* * *

## Source 工作流

### Step 1. `source tag`

从 Hotpot 原始行读取数据，并打上：

- `data_split`
- `answerability_split`

两套划分用不同盐值的稳定哈希完成，彼此独立。

### Step 2. `source build`

根据 `answerability_split` 生成混合 concrete 样本：

- `answerability_split = answerable`
  - 生成 1 条 `answerable`
- `answerability_split = unanswerable`
  - 生成 1 条 `unanswerable`
- `answerability_split = both`
  - 生成 1 条 `answerable`
  - 生成 1 条 `unanswerable`

`answerable`：

- 由 HotpotQA gold supporting facts 构造 knowledge
- 默认最多保留 4 条 supporting facts
- 必要时为每条 support 加相邻窗口句

`unanswerable`：

- 先用相同参数生成 answerable scaffold
- 再将关键 supporting fact 替换为邻句 / 同文档非-support 句 / 相邻文档句
- 保持 knowledge block 数量和长度分布尽量接近，而不是简单删除 support

### Step 3. `source prefilter`

对混合 concrete 样本统一过滤：

1. 用 `infer_prompt(schema2)` 做 token 预算过滤
2. 仅对 `unanswerable` 样本做 NLI 判负过滤

`unanswerable` 判负标准：

- premise: `knowledge + question`
- hypothesis: `The answer is {reference_answer}.`
- 保留条件：`full_binary = no`

这一步是唯一的 source 级 prompt token 过滤入口。`teacher generate` 不再做该过滤。

### Step 4. `source partition`

按 `data_split` 输出：

- `validation.jsonl`
- `test.jsonl`
- `train_sft_raw.jsonl`
- `train_dpo_raw.jsonl`

每个文件内部仍保持 `answerable / unanswerable` 混合。

* * *

## Teacher 生成与验证

### `teacher generate`

输入混合样本：

- `answerable` 提供 `reference_answer`
- `unanswerable` 不提供 `reference_answer`

teacher prompt 对 `unanswerable` 的要求：

- `evidence = []`
- 使用允许的 refusal template
- `rationale` 说明缺失、歧义或矛盾

### `teacher validate`

接受混合候选，内部按 `answerability_label` 分流：

`answerable`：

- canonicalization
- parse
- protocol
- evidence
- correctness
- optional semantic
- soft ranking
- `derived_confidence`

`unanswerable`：

- canonicalization
- parse
- protocol
- answerability 一致性
- completion token budget
- 不做 evidence / correctness / semantic
- 选择 `candidate_id` 最小的首个 hard-pass 候选

* * *

## SFT 数据构建

输入为混合 selected 样本：

- `answerable`
  - 必须有 `derived_confidence`
  - 训练前覆盖 completion 中的 `confidence`
- `unanswerable`
  - 保留 teacher 原始 `confidence`

输出字段统一使用 `data_split`，不再使用旧字段 `split`。

* * *

## 当前命令顺序

1. `python -m grounded_qa.cli source tag`
2. `python -m grounded_qa.cli source build`
3. `python -m grounded_qa.cli source prefilter`
4. `python -m grounded_qa.cli source partition`
5. `python -m grounded_qa.cli teacher generate`
6. `python -m grounded_qa.cli teacher validate`
7. `python -m grounded_qa.cli build sft-records`
