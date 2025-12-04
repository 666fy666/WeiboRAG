"""生成集成模块。

通过 DeepSeek API 将检索结果转化为回答。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

from .retrieval_optimization import RetrievalResult

logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0
DEFAULT_TIMEOUT = 60
DEFAULT_MODEL = "deepseek-chat"
MAX_BACKOFF_MULTIPLIER = 3


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

    def generate(
        self, 
        query: str, 
        contexts: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, object]:
        """组合检索上下文并生成回答。
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文片段
            conversation_history: 可选的对话历史，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """

        if not self.api_key or not self.api_key.strip():
            raise ValueError("DeepSeek API Key 未配置")

        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        messages = self._build_messages(query, contexts, conversation_history)
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "temperature": max(0.0, min(2.0, self.temperature)),  # 确保温度在有效范围内
            "max_tokens": max(1, self.max_new_tokens),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._post_with_retry(payload, headers)
            data = response.json()
        except Exception as exc:
            self.logger.error("API 调用失败: %s", exc)
            raise RuntimeError("生成回答失败") from exc

        # 安全地提取回答
        choices = data.get("choices", [])
        if not choices:
            self.logger.warning("API 响应中没有 choices 字段")
            return {
                "answer": "",
                "contexts": [self._format_context(item) for item in contexts],
                "raw": data,
            }

        message = choices[0].get("message", {})
        answer = message.get("content", "")

        return {
            "answer": answer.strip() if answer else "",
            "contexts": [self._format_context(item) for item in contexts],
            "raw": data,
        }

    def _build_messages(
        self, 
        query: str, 
        contexts: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """构建消息列表，包含系统提示、历史对话和当前查询。
        
        Args:
            query: 当前用户查询
            contexts: 检索到的上下文片段
            conversation_history: 可选的对话历史
        """
        context_text = "\n\n".join(
            [
                f"[片段 {idx + 1}]\n来源账号: {item.metadata.get('account_name', '未知')}\n"
                f"发布时间: {item.metadata.get('publish_time', '未知')}\n"
                f"内容: {item.text}"
                for idx, item in enumerate(contexts)
            ]
        )

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 如果有对话历史，插入历史消息（在system之后，当前查询之前）
        if conversation_history:
            messages.extend(conversation_history)
        
        # 添加当前查询和上下文
        messages.append({
            "role": "user",
            "content": (
                f"当前时间：{now_str}\n"
                "用户问题如下：\n"
                f"{query}\n\n"
                "以下是可用的内容片段：\n"
                f"{context_text if context_text else '（未检索到相关内容）'}"
            ),
        })
        
        return messages

    def _post_with_retry(
        self, payload: Dict[str, object], headers: Dict[str, str]
    ) -> requests.Response:
        """带重试的 POST 请求。

        Args:
            payload: 请求载荷
            headers: 请求头

        Returns:
            HTTP 响应对象

        Raises:
            requests.RequestException: 如果所有重试都失败
        """
        for attempt in range(1, DEFAULT_RETRIES + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                return response
            except requests.Timeout as exc:
                self.logger.warning(
                    "DeepSeek 调用超时（第 %d/%d 次）", attempt, DEFAULT_RETRIES
                )
                if attempt == DEFAULT_RETRIES:
                    raise RuntimeError("API 调用超时，已重试所有次数") from exc
            except requests.HTTPError as exc:
                # HTTP 错误（4xx, 5xx）不重试
                self.logger.error("DeepSeek API 返回 HTTP 错误: %s", exc)
                raise
            except requests.RequestException as exc:  # pylint: disable=broad-except
                self.logger.warning(
                    "DeepSeek 调用失败（第 %d/%d 次）: %s", attempt, DEFAULT_RETRIES, exc
                )
                if attempt == DEFAULT_RETRIES:
                    raise RuntimeError(
                        f"API 调用失败，已重试 {DEFAULT_RETRIES} 次"
                    ) from exc

            # 指数退避
            backoff_time = DEFAULT_BACKOFF * min(attempt, MAX_BACKOFF_MULTIPLIER)
            time.sleep(backoff_time)

        raise RuntimeError("DeepSeek 调用失败")

    @staticmethod
    def _format_context(item: RetrievalResult) -> Dict[str, object]:
        return {
            "chunk_id": item.chunk_id,
            "score": item.score,
            "metadata": item.metadata,
            "text": item.text,
        }


