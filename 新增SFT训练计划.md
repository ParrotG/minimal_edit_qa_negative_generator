目标:

提升模型在“给定 knowledge 回答 question”任务中的：
1. **Protocol compliance**：结构化输出稳定、可解析、字段/枚举合法
2. **Evidence fidelity**：引用的证据片段真实来自 knowledge（不伪引）
3. **Grounded correctness**：
   - knowledge 足够时：回答正确且不越界
   - knowledge 不足时：正确拒答（NEI）
4. **Calibration**：confidence 与真实正确率一致性提升（分桶校准）

* * *

## 总体方法
1. **SFT**：先让模型学会任务协议（schema + 引用 + NEI 拒答模板），获得稳定、可验证输出
2. **DPO**：在相同协议下学习偏好边界，更偏好“证据支持的正确回答”，而非“看起来像对、伪引用、越界补充”的回答

该方案符合一般 LLM 训练范式：先用 SFT 学任务格式与基本行为，再用 DPO 做偏好对齐。

* * *

## Schema 与协议规则
```json
{
  "answerability": "answerable | unanswerable",
  "evidence": [{"quote": "..."}],
  "rationale": "...",
  "answer": "...",
  "confidence": "high | medium | low"
}
```

注：
- 取消 `reason` 字段，避免在早期流水线中引入额外的越界解释风险。
- 暂不引入 span / title / sent_id 等字段到模型输出 schema；这些信息可在内部数据结构中保留，供构造、检查与后续 DPO 使用。

协议约束：
- **unanswerable 规则**：当 `answerability = unanswerable` 时
  - `evidence` 必须为空数组 `[]`
  - `rationale` 必须简要说明知识中缺失、歧义或矛盾的信息，且不能与 `answer` 完全相同
  - `answer` 必须是拒答模板集合之一
- **answerable 规则**：当 `answerability = answerable` 时
  - `evidence` 至少 1 条，最多 K 条（默认 K=3）
  - 每条 `quote` 必须是 knowledge 原文子串（归一化后精确匹配）
  - `rationale` 必须简短描述从 evidence 到 answer 的推导
  - `answer` 必须直接回答 question，保持短且明确

### 拒答模板

v1 先固定为小集合，便于训练与验证：

- `"I don't know based on the provided knowledge."`
- `"The provided knowledge does not contain enough information to answer the question."`
- `"I cannot answer from the provided knowledge."`

后续如需放宽，再增加语义识别型拒答检查。

### Prompt 构造

需要两套 prompt 构造工具，但不集中塞入单独的 `grounded_qa` 包，而是在 `src/` 下按职责拆分到多个可复用包中：

- `schema1`：完整 schema 示例 + 规则说明 + knowledge + question
- `schema2`：简短任务标签 + 输出约束 + knowledge + question

基本使用方法：
- 数据生成阶段：用 `schema1` 通过 API 调用强模型生成合规回答
- 训练阶段：用 `schema2 + 合规回答` 训练小模型
- 生成阶段：用 `schema2` 启动推理

其中：
- 当问题可回答时，`rationale` 仍要求仅通过 `evidence` 做最小必要推理得到答案。
- 当问题不可回答时，`rationale` 应指出关键信息缺失、歧义或矛盾，而不是只输出模板拒答。

可选地按小比例混入 `schema1` 训练样本，用于 prompt 分布锚定，但不是首要目标。

* * *

## 检查器 / 过滤器

### 1) Protocol 检查器

1.1 **JSON 解析 + JSONSchema / Pydantic 校验**
- 字段齐全、枚举合法、类型合法
- `evidence` 必须为数组

1.2 **长度与预算校验**
- `answer`、`rationale`、`evidence.quote` 的长度上限
- 整体 token 上限

### 2) 引用检查器

2.1 **quote 子串校验**（归一化后）
- `quote in knowledge` 必须为真

2.2 **quote 质量约束**
- 最小长度（避免太短歧义）
- 最大长度（避免整段粘贴）
- 去重（避免 evidence 列表重复）

### 3) Answer 语义检查器

3.1 **NLI / entailment 检查（用于 answerable 样本）**
- 默认输入为：`(question + selected_support_knowledge) -> answer`
- `selected_support_knowledge` 默认由 evidence quotes 拼接而成；必要时可回退到构造时保留的 support-window knowledge
- 复用现有 `qa_judge` 中的 NLI verifier，但需要新增面向结构化输出的适配层

3.2 **Answer type 检查（可复用现有 qa_consistency）**
- 检查答案是否与问题类型基本匹配

### 4) unanswerable 拒答检查器

若 `answerability = unanswerable`：
- `evidence == []`
- `answer ∈ refusal_templates`

* * *

## 数据准备（HotpotQA）

数据源使用 HotpotQA(https://huggingface.co/datasets/hotpotqa/hotpot_qa); 注意：其数据中知识的形式是：
context: 给定的候选证据（一个字典）
context["title"]: 段落标题列表（通常是 Wikipedia 条目标题）
context["sentences"]: 与标题一一对应的句子列表（每个段落被切成若干句子
supporting_facts: 标注的“支持事实”（一个字典）
supporting_facts["title"]: 支持句所在段落的标题
supporting_facts["sent_id"]: 支持句在该段落中的句子序号（从 0 开始）

由于其原始 `context` 较长，而训练平台显存有限，基本原则是应仅截取支持段落（或支持句），unanswerable数据构造通过不取支持句方法构造（查阅hotpot qa相关文档，讨论具体构造方法以平衡长度，支持性和难度）

### 输入数据准备

对每条 HotpotQA，构造数据：
- `question`
- `knowledge`
- 可选 `reference_answer`
- `answerability_label`
- `difficulty`（保留 HotpotQA 的 `easy / medium / hard`）
- 内部保留 supporting metadata（如 title、sentence index），但默认不写入模型输出 schema

### knowledge 构造原则

#### answerable 样本

尽量仅使用 `supporting_facts` 对应句子构造 knowledge：
- 默认取全部 supporting sentences
- 如上下文过于破碎，可为每个 supporting sentence 额外拼接相邻 1 句，必要时最多扩到相邻 2 句
- 不默认保留整段 context
- 构造后执行 `max_token` 检查，只保留在限额内的样本

#### unanswerable 样本

`unanswerable` 走独立工作流，不再只依赖简单 support-drop：
- 一半输入来自已选 `answerable` 样本的同源派生
- 另一半输入来自同 split 的剩余 Hotpot raw 池
- 两类来源都先按 `answerable` 相同参数构造 supporting scaffold
- 再将 1 条或多条关键 supporting fact 替换为：
  - 同文档邻句
  - 同文档非-support 句
  - 临近文档句
- 基本思想是：排除支持句，但让 knowledge 长度和表面分布保持接近

在 teacher 生成前，还需增加一层 NLI 先验判负：
- 将 `knowledge + question + "The answer is {reference_answer}."` 输入 NLI 判定器
- 仅保留 `full_binary = no` 的样本
- 这样可以剔除“虽然替换了 support，但原答案仍然可推出”的伪负例

### 数据切分

由于后续生成、过滤、人工复核会显著改变分布，采用先切 Raw 池、再分别加工的做法：

1. 先从 HotpotQA 构造合法原始样本
2. 对 `schema2` 包装后的输入做 token 预算过滤
3. 将通过过滤的数据切成：
   - `Validation`
   - `Test`
   - `Train-SFT-Raw`
   - `Train-DPO-Raw`
4. 后续 SFT 与 DPO 仅从各自 Raw 池继续生成与处理

该策略可减少 teacher generation 和人工筛选带来的跨 split 污染。

* * *

## SFT 数据生成与过滤流水线

### Step A：教师生成（强模型 API）

对每条样本，用 `schema1` 包装，调用 API 生成候选：
- 每条样本生成若干候选（参数控制）
- 用检查器自动选优：先满足所有硬约束，再比较语义支持度
- 该阶段必须支持断点续跑、续生成、替换指定样本
- 实践中先做小规模试验，选择合适的 API 模型

补充：
- `answerable` 样本可提供 `reference_answer`
- `unanswerable` 样本不提供 `reference_answer`
- `unanswerable` 也走 teacher 生成，以补齐符合协议的 `rationale`

### Step B：自动过滤（强制 + 分流）

- 共用 protocol / quote / 长度检查
- 分流 `answerable` 与 `unanswerable`
- `answerable` 样本不应拒答，需通过引用与语义检查
- `unanswerable` 样本需以符合规范的形式拒答
- `unanswerable` validate 使用独立管线：
  - 做 canonicalization、parse、protocol、answerability 一致性、completion token budget
  - 不做 evidence / correctness / semantic 检查
  - v1 暂不做软排序，选择首个 hard-pass 候选

### Step C：人工复核

- 对自动检查失败边界样本、语义高风险样本进行抽样复核
- 记录失败原因，便于后续规则收敛

### Step D：生成训练集

将合规输出与 `schema2` 包装后的输入组合成适合 SFT 的 `prompt-completion` 数据，并与预留的 validation / test 集打包保存。

* * *

## SFT 训练（LoRA）

用上一步生成的数据执行 SFT：
- 按固定 step 保存 checkpoint，便于后续测评曲线
- 可选地进行难度递增采样，但首版不是必须项

confidence 处理规则：
- `answerable` completion 的 `confidence` 在入训前由 `derived_confidence` 覆盖
- `unanswerable` completion 的 `confidence` v1 暂保留 teacher 原值

* * *

## 评估方案

用训练后的模型对 validation / test 集生成结构化输出，并评估。
评估可复用既有 `model_eval` 的加载、生成与曲线组织方式，但需要新增结构化输出解析和指标实现。

### 1) Protocol
- JSON parse rate
- Schema valid rate
- Enum valid rate
- Length compliance rate
- 拒答合规率

### 2) Evidence fidelity
- Quote substring rate（归一化后精确匹配）
- Evidence count distribution
- Evidence redundancy rate

### 3) answerability 识别正确率
- `answerable / unanswerable` 的 accuracy、precision、recall、F1

### 4) Groundedness
- 在 `answerability = answerable` 样本中：
  - answer 的 NLI 支持率
  - answer type 一致率
  - DeepEval hallucination metric（LLM-as-judge）

### 5) Calibration
- `confidence` 分桶后的准确率
- 简单 ECE 风格统计或分桶偏差分析


## DPO 预留接口

DPO 总体思路保持，但必须调整为与 SFT 后协议一致：
- DPO prompt 使用 `schema2`
- chosen / rejected 统一为结构化输出
- pair 过滤需要增加 schema-valid / quote-valid / answerability 一致性等条件

该部分暂不执行，但需要在新增模块中注意预留相应的可扩展空间
