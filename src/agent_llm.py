from __future__ import annotations

import json
import os
from typing import Any

import requests


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


class AnthropicMessagesClient:
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
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
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
