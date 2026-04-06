from __future__ import annotations

from openai import OpenAI


class SummarizationProvider:
    """Wraps an OpenAI-compatible API (Ollama or LM Studio) for summarization."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key="not-needed")

    def list_models(self) -> list[str]:
        """Query the provider for available models."""
        try:
            response = self._client.models.list()
            return [m.id for m in response.data]
        except Exception:
            return []

    def summarize(self, system_prompt: str, user_prompt: str) -> str:
        """Send a summarization request and return the response text."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
