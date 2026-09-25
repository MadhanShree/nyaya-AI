"""Groq (fast open-model hosting) through its OpenAI-compatible API, plus tolerant JSON parsing."""

import json
import os
import re
from typing import Protocol

from openai import OpenAI, OpenAIError


class LLMError(RuntimeError):
    """The message is safe to show to users."""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class GroqClient:
    def __init__(self) -> None:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise LLMError("The server has no GROQ_API_KEY configured.")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._client = OpenAI(
            api_key=key,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=90,
            max_retries=2,
        )

    def complete(self, system: str, user: str) -> str:
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except OpenAIError as e:
            raise LLMError("The AI service could not answer. Please try again.") from e
        return r.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise LLMError("The AI reply was not in the expected format. Please try again.")
    return data
