from __future__ import annotations

import json
import os
from typing import Any

import requests


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class JSONSchemaClient:
    def create_json_response(
        self,
        *,
        instructions: str,
        user_input: str,
        schema: dict[str, Any],
        max_output_tokens: int = 1200,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _extract_json_block(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        return stripped


class AnthropicMessagesClient(JSONSchemaClient):
    def __init__(self, api_key: str | None = None, model: str = "claude-3-7-sonnet-20250219") -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self.model = model

    def create_json_response(
        self,
        *,
        instructions: str,
        user_input: str,
        schema: dict[str, Any],
        max_output_tokens: int = 1200,
    ) -> dict[str, Any]:
        schema_text = json.dumps(schema, indent=2)
        prompt = "\n\n".join(
            [
                "Return only valid JSON.",
                "The JSON must match this schema exactly:",
                schema_text,
                "User request:",
                user_input,
            ]
        )
        payload = {
            "model": self.model,
            "system": instructions,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            ANTHROPIC_MESSAGES_URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"Anthropic Messages API error {response.status_code}: {response.text}")
        response_json = response.json()
        text = self._extract_text(response_json)
        return json.loads(self._extract_json_block(text))

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        content = response_json.get("content", [])
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        if not text_parts:
            raise ValueError("No text content found in Anthropic response.")
        return "".join(text_parts).strip()


class OpenAIChatCompletionsClient(JSONSchemaClient):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1-mini") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self.model = model

    def create_json_response(
        self,
        *,
        instructions: str,
        user_input: str,
        schema: dict[str, Any],
        max_output_tokens: int = 1200,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_response",
                    "schema": schema,
                    "strict": True,
                },
            },
            "max_tokens": max_output_tokens,
        }

        response = requests.post(
            OPENAI_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"OpenAI Chat Completions API error {response.status_code}: {response.text}")
        response_json = response.json()
        text = self._extract_text(response_json)
        return json.loads(self._extract_json_block(text))

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError("No choices found in OpenAI response.")
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not text:
            raise ValueError("No message content found in OpenAI response.")
        return str(text).strip()


def create_json_client(provider: str, model: str) -> JSONSchemaClient:
    normalized = provider.strip().lower()
    if normalized == "anthropic":
        return AnthropicMessagesClient(model=model)
    if normalized == "openai":
        return OpenAIChatCompletionsClient(model=model)
    raise ValueError(f"Unsupported agent provider: {provider}")
