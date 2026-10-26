"""LLM access for agents.

Agents depend on the small `LLM` protocol, not on a vendor SDK. `BedrockLLM`
raises `TransientLLMError` for throttling and timeouts so the LangGraph node
retry policy can back off and retry; any other error fails fast.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)

TRANSIENT_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException", "InternalServerException"}


class TransientLLMError(RuntimeError):
    """Retryable failure (rate limit, timeout, temporary outage)."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str: ...


class BedrockLLM:
    def __init__(self, model_id: str, region: str) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=region)
        self.name = model_id

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        from botocore.exceptions import ClientError, ReadTimeoutError

        try:
            resp = self._client.converse(
                modelId=self.name,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            logger.warning("bedrock request failed", extra={"model_id": self.name, "error_code": code})
            if code in TRANSIENT_CODES:
                raise TransientLLMError(code) from exc
            raise
        except ReadTimeoutError as exc:
            logger.warning("bedrock request timed out", extra={"model_id": self.name})
            raise TransientLLMError("timeout") from exc
        return "".join(block.get("text", "") for block in resp["output"]["message"]["content"]).strip()


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model response (models sometimes wrap JSON in prose or fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model response")
    return json.loads(match.group(0))
