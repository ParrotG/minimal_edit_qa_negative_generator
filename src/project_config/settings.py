from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelSettings:
    """Project-wide default model identifiers."""

    target_training_llm: str = "Qwen/Qwen3-4B"
    default_tokenizer_name: str = "Qwen/Qwen3-4B"


@dataclass(frozen=True)
class TeacherApiSettings:
    """Default API runtime for teacher supervision generation."""

    model_name: str = "qwen3.5-plus"
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 90.0
    max_concurrency: int = 64
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 42


@dataclass(frozen=True)
class AnswerExtractionApiSettings:
    """Default API runtime for flat-answer extraction."""

    model_name: str = "qwen-plus"
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 90.0
    max_concurrency: int = 64
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_attempts: int = 2


@dataclass(frozen=True)
class NliSettings:
    """Default NLI verifier runtime."""

    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    device: str = "cuda"
    batch_size: int = 64
    max_length: int = 512
    fp16: bool = True


@dataclass(frozen=True)
class JudgeSettings:
    """Calibrated NLI and QA consistency defaults."""

    temperature: float = 2.17
    full_margin_threshold: float = 0.4
    reject_margin_threshold: float = 0.6
    reject_band_half_width: float = 0.05
    qa_fail_as_negative: bool = True
    qa_check_answer_type: bool = True
    qa_spacy_model: str = "en_core_web_sm"


@dataclass(frozen=True)
class ProtocolSettings:
    """Grounded-QA structured protocol defaults."""

    version: str = "grounded-qa-v1"
    max_evidence_count: int = 4
    max_prompt_tokens: int = 512
    max_completion_tokens: int = 512
    canonical_refusal: str = "I don't know based on the provided knowledge."
    refusal_templates: Tuple[str, ...] = (
        "I don't know based on the provided knowledge.",
        "The provided knowledge does not contain enough information to answer the question.",
        "I cannot answer from the provided knowledge.",
    )
    field_order: Tuple[str, ...] = ("answerability", "evidence", "rationale", "answer", "confidence")
    teacher_prompt_style: str = "teacher_v1"
    infer_prompt_style: str = "infer_v1"


@dataclass(frozen=True)
class TokenBudgetSettings:
    """Tokenizer-based budget and accounting defaults."""

    prompt_limit: int = 512
    completion_limit: int = 512
    count_batch_size: int = 512
    record_token_usage: bool = False


@dataclass(frozen=True)
class GenerationSettings:
    """Local generation runtime defaults."""

    device: str = "cuda"
    device_map: Optional[str] = None
    dtype: str = "bf16"
    trust_remote_code: bool = True
    padding_side: str = "left"
    use_fast_tokenizer: bool = True
    seed: int = 42
    batch_size: int = 32
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    use_chat_template: bool = True
    enable_thinking: bool = False
    strip_think_tags: bool = True
    strip_role_markers: bool = True


@dataclass(frozen=True)
class SourceSettings:
    """Source tagging, construction, and filtering defaults."""

    dataset_name: str = "hotpotqa/hotpot_qa"
    train_split: str = "train"
    validation_split: str = "validation"
    max_train_samples: int = 20000
    max_validation_samples: int = 2000
    seed: int = 42
    validation_ratio: float = 0.5
    test_ratio: float = 0.5
    data_split_hash_salt: str = "data_split"
    answerable_ratio: float = 0.7
    unanswerable_ratio: float = 0.1
    both_ratio: float = 0.2
    answerability_hash_salt: str = "answerability_split"
    window_size: int = 1
    max_supporting_facts: int = 4
    drop_over_max_supporting_facts: bool = True
    include_title_prefix: bool = True
    replace_supporting_facts_min: int = 1
    replace_supporting_facts_max: int = 1
    same_doc_candidate_radius: int = 1
    allow_same_doc_non_adjacent: bool = True
    adjacent_doc_sentence_limit: int = 1
    enable_unanswerable_nli: bool = True


@dataclass(frozen=True)
class TeacherSettings:
    """Teacher generation, validation, and SFT-record defaults."""

    prompt_style: str = "teacher_v1"
    infer_prompt_style: str = "infer_v1"
    answerable_num_candidates_per_example: int = 3
    unanswerable_num_candidates_per_example: int = 1
    validate_enable_semantics: bool = True
    semantic_drop_by_nli: bool = True
    semantic_decision_source: str = "full_binary"
    keep_only_overall_ok: bool = True


@dataclass(frozen=True)
class TrainingSettings:
    """SFT dataset preparation and training defaults."""

    seed: int = 42
    max_train_samples: int = 10000
    max_validation_samples: int = 500
    max_test_samples: int = 500
    max_train_answerable_samples: int = 8000
    max_train_unanswerable_samples: int = 2000
    max_validation_answerable_samples: int = 400
    max_validation_unanswerable_samples: int = 100
    max_test_answerable_samples: int = 400
    max_test_unanswerable_samples: int = 100
    max_prompt_tokens: int = 512
    max_completion_tokens: int = 512
    train_epochs: float = 2.0
    learning_rate: float = 2e-4
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    save_steps: int = 50
    save_total_limit: int = 20
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class EvalStructuredGenerationSettings:
    """Structured generation defaults for checkpoint evaluation under infer prompts."""

    split: str = "validation"
    max_samples: int = -1
    include_base: bool = False
    batch_size: int = 32
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    prompt_mode: str = "infer"
    fewshot_k: int = 0
    retry_on_protocol_fail: bool = False
    max_attempts: int = 1
    eval_track: str = "sft_structured"
    eval_variant: str = "checkpoint"


@dataclass(frozen=True)
class EvalBaseProtocolSettings:
    """Base-model protocol baseline defaults on the test split."""

    split: str = "test"
    max_samples: int = -1
    include_base: bool = True
    batch_size: int = 32
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    prompt_mode: str = "teacher_fewshot"
    fewshot_k: int = 2
    retry_on_protocol_fail: bool = False
    max_attempts: int = 1
    eval_track: str = "base_protocol"
    eval_variant: str = "fewshot"


@dataclass(frozen=True)
class EvalFlatGenerationSettings:
    """Flat base-task generation defaults on the test split."""

    split: str = "test"
    max_samples: int = -1
    include_base: bool = True
    batch_size: int = 32
    max_new_tokens: int = 512
    think_max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    encourage_refusal: bool = True
    eval_track: str = "base_task"


@dataclass(frozen=True)
class EvalGroundedSettings:
    """Structured grounded-evaluation defaults for parsed protocol outputs."""

    split: str = "train"
    max_samples: int = -1
    enable_semantics: bool = True
    semantic_decision_source: str = "full_binary"
    # This threshold is consumed by the rule-based correctness checker.
    # It is a token-F1 cutoff against the reference answer, not an NLI semantic threshold.
    semantic_match_f1_threshold: float = 0.85


@dataclass(frozen=True)
class EvalLossCurveSettings:
    """Validation loss-curve defaults."""

    split: str = "validation"
    max_samples: int = -1
    include_base: bool = True
    batch_size: int = 16
    eval_track: str = "sft_structured"
    eval_variant: str = "checkpoint"


@dataclass(frozen=True)
class EvalTaskBaselineSettings:
    """Flat task-baseline evaluation defaults after answer extraction."""

    split: str = "train"
    max_samples: int = -1
    semantic_decision_source: str = "full_binary"
    # This threshold is consumed by the rule-based correctness checker.
    # It is a token-F1 cutoff against the reference answer, not an NLI semantic threshold.
    semantic_match_f1_threshold: float = 0.85


@dataclass(frozen=True)
class EvalDeepEvalSettings:
    """DeepEval runtime defaults shared by flat and structured modes."""

    split: str = "train"
    max_samples: int = -1
    input_mode: str = "flat"
    answer_source: str = "answer"
    judge_model: str = "gpt-5.2"
    threshold: float = 0.5
    max_concurrent: int = 4
    throttle_value: float = 1


@dataclass(frozen=True)
class EvalSelectionSettings:
    """Validation checkpoint selection defaults."""

    parse_ok_threshold: float = 0.95
    protocol_ok_threshold: float = 0.98
    evidence_ok_threshold: float = 0.95


@dataclass(frozen=True)
class EvalSettings:
    """Top-level evaluation defaults."""

    structured_generation: EvalStructuredGenerationSettings = field(default_factory=EvalStructuredGenerationSettings)
    base_protocol: EvalBaseProtocolSettings = field(default_factory=EvalBaseProtocolSettings)
    loss_curve: EvalLossCurveSettings = field(default_factory=EvalLossCurveSettings)
    flat_generation: EvalFlatGenerationSettings = field(default_factory=EvalFlatGenerationSettings)
    grounded: EvalGroundedSettings = field(default_factory=EvalGroundedSettings)
    task_baseline: EvalTaskBaselineSettings = field(default_factory=EvalTaskBaselineSettings)
    deepeval: EvalDeepEvalSettings = field(default_factory=EvalDeepEvalSettings)
    selection: EvalSelectionSettings = field(default_factory=EvalSelectionSettings)


@dataclass(frozen=True)
class MatcherSettings:
    """Answer-equivalence matcher defaults."""

    model_name: str = "zli12321/answer_equivalence_roberta-large"
    runtime_threshold: float = 0.27


@dataclass(frozen=True)
class CorrectnessSettings:
    """Rule-based correctness defaults."""

    # This threshold is consumed by the rule-based correctness checker.
    # It is a token-F1 cutoff against the reference answer, not an NLI semantic threshold.
    semantic_match_f1_threshold: float = 0.85


@dataclass(frozen=True)
class VerifierSettings:
    """Ad-hoc NLI verifier CLI defaults."""

    support_entail_threshold: float = 0.60
    support_contradict_threshold: float = 0.90
    knowledge_field: str = "knowledge"
    question_field: str = "question"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    skip_empty_rejected: bool = True


@dataclass(frozen=True)
class CalibrationSettings:
    """Calibration workflow defaults."""

    annotation_task_types: Tuple[str, ...] = ("nli_structured", "nli_flat", "matcher")
    split: str = "train"
    max_samples: int = -1
    group_by: str = "task"
    search_objective: str = "f1"
    enable_nli_reject_search: bool = False
    reject_alpha: float = 0.5
    temperature_min: float = 0.05
    temperature_max: float = 5.0
    temperature_step: float = 0.01
    margin_threshold_min: float = 0.0
    margin_threshold_max: float = 1.0
    margin_threshold_step: float = 0.05
    argmax_conf_threshold_min: float = 0.34
    argmax_conf_threshold_max: float = 0.95
    argmax_conf_threshold_step: float = 0.05
    band_half_width_min: float = 0.0
    band_half_width_max: float = 0.3
    band_half_width_step: float = 0.05
    matcher_model_name: str = "zli12321/answer_equivalence_roberta-large"
    matcher_runtime_threshold: float = 0.5
    matcher_threshold_min: float = 0.1
    matcher_threshold_max: float = 0.95
    matcher_threshold_step: float = 0.05
    max_samples_per_task: int = 200


@dataclass(frozen=True)
class PathSettings:
    """Project-wide path defaults."""

    error_log_dir: str = "log"


@dataclass(frozen=True)
class ProjectSettings:
    """Single source of truth for project runtime defaults."""

    model: ModelSettings = field(default_factory=ModelSettings)
    teacher_api: TeacherApiSettings = field(default_factory=TeacherApiSettings)
    answer_extraction_api: AnswerExtractionApiSettings = field(default_factory=AnswerExtractionApiSettings)
    nli: NliSettings = field(default_factory=NliSettings)
    judge: JudgeSettings = field(default_factory=JudgeSettings)
    protocol: ProtocolSettings = field(default_factory=ProtocolSettings)
    token_budget: TokenBudgetSettings = field(default_factory=TokenBudgetSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    source: SourceSettings = field(default_factory=SourceSettings)
    teacher: TeacherSettings = field(default_factory=TeacherSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
    matcher: MatcherSettings = field(default_factory=MatcherSettings)
    correctness: CorrectnessSettings = field(default_factory=CorrectnessSettings)
    verifier: VerifierSettings = field(default_factory=VerifierSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    paths: PathSettings = field(default_factory=PathSettings)


PROJECT_SETTINGS = ProjectSettings()
