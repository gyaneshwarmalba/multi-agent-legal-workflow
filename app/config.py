"""Environment-driven settings. Defaults run the whole workflow offline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "local"))
    aws_region: str = field(default_factory=lambda: _env("AWS_DEFAULT_REGION", "us-west-2"))
    bedrock_model_id: str = field(default_factory=lambda: _env("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"))
    policy_path: Path = field(
        default_factory=lambda: Path(_env("COMPLIANCE_POLICY_PATH", str(PROJECT_ROOT / "policies" / "compliance_policy.json")))
    )
    checkpoint_db: Path = field(
        default_factory=lambda: Path(_env("CHECKPOINT_DB", str(PROJECT_ROOT / "data" / "state" / "checkpoints.sqlite")))
    )

    @property
    def sample_contracts_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "sample_contracts"
