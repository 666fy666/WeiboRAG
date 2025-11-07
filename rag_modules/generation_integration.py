"""生成集成模块。

通过 DeepSeek API 将检索结果转化为回答。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import requests

from .retrieval_optimization import RetrievalResult

logger = logging.getLogger(__name__)


class DeepSeekGenerator:
    """DeepSeek LLM 调用封装。"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"{__name__}.DeepSeekGenerator")

    def generate(self, query: str, contexts: List[RetrievalResult]) -> Dict[str, object]:
        """组合检索上下文并生成回答。"""

        if not self.api_key:
            raise ValueError("DeepSeek API Key 未配置")

        messages = self._build_messages(query, contexts)
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._post_with_retry(payload, headers)
        data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "answer": answer.strip(),
            "contexts": [self._format_context(item) for item in contexts],
            "raw": data,
        }

    def _build_messages(
        self, query: str, contexts: List[RetrievalResult]
    ) -> List[Dict[str, str]]:
        context_text = "\n\n".join(
            [
                f"[片段 {idx + 1}]\n来源账号: {item.metadata.get('account_name', '未知')}\n"
                f"发布时间: {item.metadata.get('publish_time', '未知')}\n"
                f"内容: {item.text}"
                for idx, item in enumerate(contexts)
            ]
        )

        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"当前时间：{now_str}\n"
                    "用户问题如下：\n"
                    f"{query}\n\n"
                    "以下是可用的内容片段：\n"
                    f"{context_text if context_text else '（未检索到相关内容）'}"
                ),
            },
        ]
        return messages

    def _post_with_retry(self, payload: Dict[str, object], headers: Dict[str, str]) -> requests.Response:
        retries = 3
        backoff = 2.0
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:  # pylint: disable=broad-except
                self.logger.error("DeepSeek 调用失败（第 %d 次）: %s", attempt, exc)
                if attempt == retries:
                    raise
                time.sleep(backoff * attempt)
        raise RuntimeError("DeepSeek 调用失败")

    @staticmethod
    def _format_context(item: RetrievalResult) -> Dict[str, object]:
        return {
            "chunk_id": item.chunk_id,
            "score": item.score,
            "metadata": item.metadata,
            "text": item.text,
        }


