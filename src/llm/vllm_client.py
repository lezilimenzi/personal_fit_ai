"""vLLM OpenAI API 호환 클라이언트. Colab ngrok 터널 지연 대비 timeout 설정 포함."""

import httpx
from openai import OpenAI


class VLLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "dummy"):
        headers = {}
        if "ngrok" in base_url:
            headers["ngrok-skip-browser-warning"] = "true"
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=10.0),
            default_headers=headers if headers else None,
        )
        self.model = model

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def generate(self, messages: list[dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.4),
            top_p=kwargs.get("top_p", 0.8),
            max_tokens=kwargs.get("max_tokens", 512),
            extra_body={
                "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
            },
        )
        return response.choices[0].message.content
