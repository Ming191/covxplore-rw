"""PromptConfig — a dataclass of boolean flags, one per ablation section.

Each flag controls whether the corresponding section from ``prompts.yaml`` is
included when building the agent's system prompt and task description.

Section mapping (from the literature):
  S1 role_persona      → AgentCoder persona + safety-standard framing (2312.13010)
  S2 cot_reasoning     → LIBRO/TELPA two-stage CoT: reason about pairs first (2307.07055)
  S3 coverage_guidance → CoverAgent local gap injection (2402.09171)
  S4 self_reflection   → Self-Edit execution-error repair loop (2305.04087)
  S5 few_shot_examples → FuzzLLM / ChatUniTest focal-method few-shot (2305.04764)
  S6 output_format     → Schafer et al. structured output format (2302.06527)
"""
from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    name: str

    # Ablation flags
    role_persona: bool = True       # S1
    cot_reasoning: bool = True      # S2
    coverage_guidance: bool = True  # S3
    self_reflection: bool = True    # S4
    few_shot_examples: bool = False # S5 — off by default (token-expensive)
    output_format: bool = True      # S6
    search_tools: bool = True
    preload_context: bool = True
    unlimited_batch: bool = False

    def enabled_sections(self) -> list[str]:
        """Return ordered list of section names that are enabled."""
        all_sections = [
            ("role_persona",      self.role_persona),
            ("cot_reasoning",     self.cot_reasoning),
            ("coverage_guidance", self.coverage_guidance),
            ("self_reflection",   self.self_reflection),
            ("few_shot_examples", self.few_shot_examples),
            ("output_format",     self.output_format),
        ]
        return [name for name, enabled in all_sections if enabled]

    def __str__(self) -> str:
        flags = ", ".join(self.enabled_sections()) or "none"
        return f"PromptConfig({self.name!r}, sections=[{flags}])"
