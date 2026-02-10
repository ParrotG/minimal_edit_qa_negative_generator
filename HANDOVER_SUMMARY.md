# 项目交接备忘（非正式，已同步到最新代码/结果）

## 1. 背景与目标
- 目标：用 DPO 提升类 RAG QA 忠实度（已知可回答前提下，答案应被给定知识支持）。
- 当前不讨论拒答机制。
- 早期直接用 HaluEval 原始正反例训练，提升有限且后期震荡；怀疑存在可被模型利用的表观偏差。
- 因此切换到最小编辑负例路线：从 `knowledge + question + right_answer` 生成“最小但不被支持”的 rejected，再过滤与分桶。

## 2. 当前范围与约束
- 数据范围：当前阶段仅 `HaluEval/qa`。
- 资源约束：本地 12GB VRAM / 32GB RAM，优先可复现和性价比。
- 工程方式：分段运行，保留中间产物（sample/candidate/dpo pair）。

## 3. 已落地能力（meqng 侧）
- 统一 NLI premise 拼接：`knowledge + "\\nQuestion: " + question`（避免各模块不一致）。
- 采样阶段新增 entailment 阈值过滤（正例可按门槛预筛）。
- 扰动层：
  - 规则扰动（实体、数字、否定）保留。
  - TextAttack/NLI-flip 扰动已接入（可配置约束与目标标签比例）。
  - 轻量 span 级扰动（`span_drop`）已实现。
  - 支持同一候选的多次扰动尝试（attempt 参数）。
- 过滤与排序：
  - 长度比、编辑距离、答案类型、NLI 支持度过滤。
  - QA 一致性/语法过滤可选。
  - 候选排序（最小编辑 + 难度 + 其他启发）。
- 难度分桶：
  - 按 `delta = logP(chosen) - logP(rejected)` 打分并分桶。
  - 默认难度模型接口为 `Qwen/Qwen3-0.6B`（可替换）。

## 4. 已落地能力（dpo_trainer 侧）
- 新增 meqng 数据接入：
  - `src/dpo_trainer/preprocess_halueval_pairs_to_dpo.py` 支持 `--source meqng --meqng_jsonl ...`。
  - 会优先读取 `eval_question/eval_contexts`；缺失时从 `prompt` 反推兼容字段。
- 训练/评估管线维持原逻辑为主，仅增加输入兼容接口。
- 运行方式提醒：
  - 直接执行脚本会触发相对导入错误，建议 `python -m src.dpo_trainer.xxx` 或安装 editable 包后再调用。

## 5. RAGAS 评估问题与修复（关键）
- 已定位并修复 `answer` 污染问题（混入 `user/assistant/<think>/Question:`）：
  - 根因是左填充下按 `attention_mask.sum()` 截断生成结果，导致切片错位。
  - 修复为按 padded 宽度截断，并补 `pad_token_id`。
  - 在 `src/dpo_trainer/ragas_faithfulness_halueval.py` 新增清洗选项：
    - `--strip_role_markers`
    - `--strip_think_tags`
- 修复后抽检：
  - `assistant` 泄漏 0%
  - `<think>` 泄漏 0%
  - 仍有少量 `Question:` 回显（约 1.33%，主要集中早期 checkpoint）。

## 6. 最新实验信号（修复后曲线）
- `outputs/results/meqng_ragas_curve.csv`：
  - base: `0.772865`
  - best checkpoint: step 60, `0.774749`（仅微幅高于 base）
  - 多数 checkpoint 低于 base，step 20 明显下滑（`0.662514`）
- `outputs/results/meqng_pairwise_curve.csv`：
  - `pairwise_acc` 持续接近 `0.989`
  - 与 RAGAS 忠实度趋势不一致，说明“偏好学习成功”未稳定转化为“事实忠实提升”。
- 当前结论：包装噪声已大幅缓解，但训练信号与目标指标仍存在错配风险。

## 7. 已知风险与未完成项
- 偏差审计（artifact probe）尚未实现（按计划暂缓）。
- NLI 多模型集成投票/交叉验证仍未落地。
- 难度分桶阈值与真实训练收益尚未系统标定。
- TextAttack 依赖链重：`.[attack]` 可能触发 `sentencepiece` 源码编译失败（`cmake/pkg-config` 环境问题）。
- RAGAS 样本仍见少量 `faithfulness=None`，统计时需明确处理方式（剔除或单独计数）。

## 8. 协作偏好与规范（持续有效）
- Python 3.11；默认不主动升级依赖。
- 代码注释/日志/报错英文；沟通总结中文。
- 偏好分段运行、参数可调、可观察中间产物。
- 你希望每次代码编辑后给可执行命令，但不代跑命令。

## 9. 建议的下一步（最小可执行）
1. 以 step 60 为候选早停点，做多随机种子复现实验，确认是否稳健优于 base。
2. 同时报 `pairwise + ragas + 人工抽检` 三指标，避免单指标误判。
3. 对早期崩塌（step 20）做保守超参回调（更小 LR / 更长 warmup / 更保守 beta）。
4. 在候选构造侧提高 hard 比例并做分桶混合对照，验证是否能把偏好收益转成忠实度收益。
