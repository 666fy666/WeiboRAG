"""LLM 提供商抽象和实现。

支持多种 LLM 提供商，包括 DeepSeek、OpenAI、通义千问等。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import requests

from .retrieval_optimization import RetrievalResult

logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0
DEFAULT_TIMEOUT = 60
MAX_BACKOFF_MULTIPLIER = 3


class LLMGenerator(ABC):
    """LLM 生成器抽象基类。"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> None:
        """初始化 LLM 生成器。

        Args:
            api_key: API 密钥
            api_url: API 端点 URL
            model_name: 模型名称
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            system_prompt: 系统提示词
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        """生成回答。

        Args:
            query: 用户查询
            contexts: 检索到的上下文片段
            conversation_history: 可选的对话历史

        Returns:
            包含 answer、contexts 和 raw 的字典
        """
        pass

    def _build_messages(
        self,
        query: str,
        contexts: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建消息列表，包含系统提示、历史对话和当前查询。

        Args:
            query: 当前用户查询
            contexts: 检索到的上下文片段
            conversation_history: 可选的对话历史

        Returns:
            消息列表
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

        messages = [{"role": "system", "content": self.system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append(
            {
                "role": "user",
                "content": (
                    f"当前时间：{now_str}\n"
                    "用户问题如下：\n"
                    f"{query}\n\n"
                    "以下是可用的内容片段：\n"
                    f"{context_text if context_text else '（未检索到相关内容）'}"
                ),
            }
        )

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
                    "API 调用超时（第 %d/%d 次）", attempt, DEFAULT_RETRIES
                )
                if attempt == DEFAULT_RETRIES:
                    raise RuntimeError("API 调用超时，已重试所有次数") from exc
            except requests.HTTPError as exc:
                self.logger.error("API 返回 HTTP 错误: %s", exc)
                raise
            except requests.RequestException as exc:
                self.logger.warning(
                    "API 调用失败（第 %d/%d 次）: %s", attempt, DEFAULT_RETRIES, exc
                )
                if attempt == DEFAULT_RETRIES:
                    raise RuntimeError(
                        f"API 调用失败，已重试 {DEFAULT_RETRIES} 次"
                    ) from exc

            backoff_time = DEFAULT_BACKOFF * min(attempt, MAX_BACKOFF_MULTIPLIER)
            time.sleep(backoff_time)

        raise RuntimeError("API 调用失败")

    @staticmethod
    def _format_context(item: RetrievalResult) -> Dict[str, object]:
        """格式化检索结果上下文。

        Args:
            item: 检索结果

        Returns:
            格式化后的上下文字典
        """
        return {
            "chunk_id": item.chunk_id,
            "score": item.score,
            "metadata": item.metadata,
            "text": item.text,
        }


class DeepSeekGenerator(LLMGenerator):
    """DeepSeek LLM 调用封装。"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> None:
        """初始化 DeepSeek 生成器。

        Args:
            api_key: DeepSeek API 密钥
            api_url: API 端点 URL
            model_name: 模型名称（默认：deepseek-chat）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            system_prompt: 系统提示词
        """
        super().__init__(
            api_key=api_key,
            api_url=api_url or "https://api.deepseek.com/v1/chat/completions",
            model_name=model_name or "deepseek-chat",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    def generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        """生成回答。

        Args:
            query: 用户查询
            contexts: 检索到的上下文片段
            conversation_history: 可选的对话历史

        Returns:
            包含 answer、contexts 和 raw 的字典
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError("DeepSeek API Key 未配置")

        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        messages = self._build_messages(query, contexts, conversation_history)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": max(0.0, min(2.0, self.temperature)),
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


class OpenAIGenerator(LLMGenerator):
    """OpenAI LLM 调用封装。"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> None:
        """初始化 OpenAI 生成器。

        Args:
            api_key: OpenAI API 密钥
            api_url: API 端点 URL（默认：OpenAI 官方端点）
            model_name: 模型名称（默认：gpt-3.5-turbo）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            system_prompt: 系统提示词
        """
        super().__init__(
            api_key=api_key,
            api_url=api_url or "https://api.openai.com/v1/chat/completions",
            model_name=model_name or "gpt-3.5-turbo",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    def generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        """生成回答。

        Args:
            query: 用户查询
            contexts: 检索到的上下文片段
            conversation_history: 可选的对话历史

        Returns:
            包含 answer、contexts 和 raw 的字典
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError("OpenAI API Key 未配置")

        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        messages = self._build_messages(query, contexts, conversation_history)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": max(0.0, min(2.0, self.temperature)),
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


class QwenGenerator(LLMGenerator):
    """通义千问（Qwen）LLM 调用封装。

    支持 DashScope API（兼容 OpenAI 格式）。
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> None:
        """初始化通义千问生成器。

        Args:
            api_key: 通义千问 API 密钥（DashScope API Key）
            api_url: API 端点 URL（默认：DashScope 兼容端点）
            model_name: 模型名称（默认：qwen-turbo）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            system_prompt: 系统提示词
        """
        super().__init__(
            api_key=api_key,
            api_url=api_url
            or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            model_name=model_name or "qwen-turbo",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    def generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        """生成回答。

        Args:
            query: 用户查询
            contexts: 检索到的上下文片段
            conversation_history: 可选的对话历史

        Returns:
            包含 answer、contexts 和 raw 的字典
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError("通义千问 API Key 未配置")

        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        messages = self._build_messages(query, contexts, conversation_history)
        # 通义千问兼容 OpenAI 格式
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": max(0.0, min(2.0, self.temperature)),
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

        # 通义千问兼容 OpenAI 响应格式
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


def create_llm_generator(
    provider: str,
    api_key: str,
    api_url: str | None,
    model_name: str | None,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str,
) -> LLMGenerator:
    """创建 LLM 生成器工厂函数。

    Args:
        provider: 提供商名称（deepseek、openai、qwen）
        api_key: API 密钥
        api_url: API 端点 URL（可选）
        model_name: 模型名称（可选）
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        system_prompt: 系统提示词

    Returns:
        LLM 生成器实例

    Raises:
        ValueError: 如果提供商不支持
    """
    provider_lower = provider.lower()
    if provider_lower == "deepseek":
        return DeepSeekGenerator(
            api_key=api_key,
            api_url=api_url or "",
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    elif provider_lower == "openai":
        return OpenAIGenerator(
            api_key=api_key,
            api_url=api_url or "",
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    elif provider_lower == "qwen":
        return QwenGenerator(
            api_key=api_key,
            api_url=api_url or "",
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(
            f"不支持的 LLM 提供商: {provider}。支持的提供商: deepseek, openai, qwen"
        )

