"""数据准备模块。

负责读取微博 JSON 数据，清洗文本并生成检索用文本块。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)

# 常量定义
CONTROL_CHAR_PATTERN = re.compile(r"\s+")
LINK_PLACEHOLDER_PATTERN = re.compile(r"网页链接|原图|\[组图共\d+张]\s*")
URL_PATTERN = re.compile(r"https?://\S+")
DEFAULT_ACCOUNT_NAME = "未知账号"
ZERO_WIDTH_SPACE = "\u200b"


@dataclass
class WeiboDocument:
    """表示经过清洗的微博文档。"""

    doc_id: str
    account_id: str
    account_name: str
    publish_time: str
    content: str
    source_path: Path
    stats: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, str] = field(default_factory=dict)

    def as_metadata(self) -> Dict[str, str]:
        """生成用于索引的元数据。"""

        metadata = {
            "doc_id": self.doc_id,
            "account_id": self.account_id,
            "account_name": self.account_name,
            "publish_time": self.publish_time,
            "source_path": str(self.source_path),
        }
        metadata.update({f"stat_{k}": str(v) for k, v in self.stats.items()})
        metadata.update(self.extra)
        return metadata

    def render_with_context(self, body: str) -> str:
        """将正文与关键上下文拼接为统一文本。"""

        header = (
            f"账号: {self.account_name}（ID: {self.account_id}）\n"
            f"发布时间: {self.publish_time}\n"
        )
        return f"{header}正文: {body.strip()}".strip()


class DataPreparationPipeline:
    """微博 JSON 数据处理管线。"""

    def __init__(self, data_root: Path, chunk_size: int, chunk_overlap: int, min_content_chars: int) -> None:
        self.data_root = data_root
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_content_chars = min_content_chars
        self.logger = logging.getLogger(f"{__name__}.DataPreparationPipeline")

    def prepare_corpus(self) -> List[Dict[str, str]]:
        """读取数据并生成文本块列表。

        Returns:
            文本块列表，每个字典包含 id、text 和 metadata 字段

        Raises:
            ValueError: 如果数据目录不存在或无效
        """
        if not self.data_root.exists():
            raise ValueError(f"数据目录不存在: {self.data_root}")

        if not self.data_root.is_dir():
            raise ValueError(f"数据路径不是目录: {self.data_root}")

        documents = list(self._load_documents())
        if not documents:
            self.logger.warning("未加载到任何文档，请检查数据目录和文件格式")
            return []

        chunks: List[Dict[str, str]] = []
        for doc in documents:
            for idx, chunk in enumerate(self._chunk_text(doc.content)):
                chunk_id = f"{doc.doc_id}_{idx}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": doc.render_with_context(chunk),
                        "metadata": doc.as_metadata(),
                    }
                )
        self.logger.info("已生成 %d 篇微博，%d 个文本块", len(documents), len(chunks))
        return chunks

    def _load_documents(self) -> Iterable[WeiboDocument]:
        """遍历数据目录并产出清洗后的文档。"""

        json_files = sorted(self.data_root.rglob("*.json"))
        if not json_files:
            self.logger.warning("未在 %s 下找到任何 JSON 文件", self.data_root)
            return []

        for path in json_files:
            try:
                with path.open("r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error("读取 JSON 失败: %s，错误: %s", path, exc)
                continue

            user_info = payload.get("user", {})
            if not isinstance(user_info, dict):
                self.logger.warning("JSON 文件 %s 中的 user 字段格式异常，跳过", path)
                continue

            account_id = str(user_info.get("id", ""))
            account_name = str(user_info.get("nickname", DEFAULT_ACCOUNT_NAME))
            weibo_items = payload.get("weibo", [])

            if not isinstance(weibo_items, list):
                self.logger.warning("JSON 文件 %s 中的 weibo 字段不是列表，跳过", path)
                continue

            for item in weibo_items:
                if not isinstance(item, dict):
                    self.logger.debug("跳过非字典类型的微博条目: %s", path)
                    continue

                content = self._clean_text(str(item.get("content", "")))
                if len(content) < self.min_content_chars:
                    self.logger.debug("跳过过短的微博内容: %s", path)
                    continue

                try:
                    stats = {
                        "up_num": int(item.get("up_num", 0)),
                        "retweet_num": int(item.get("retweet_num", 0)),
                        "comment_num": int(item.get("comment_num", 0)),
                    }
                    extra = {
                        "publish_tool": str(item.get("publish_tool", "")),
                        "publish_place": str(item.get("publish_place", "")),
                    }
                    doc_id = str(item.get("id", ""))
                    if not doc_id:
                        doc_id = f"{account_id}_{item.get('publish_time', '')}"

                    yield WeiboDocument(
                        doc_id=doc_id,
                        account_id=account_id,
                        account_name=account_name,
                        publish_time=str(item.get("publish_time", "")),
                        content=content,
                        source_path=path,
                        stats=stats,
                        extra=extra,
                    )
                except (ValueError, KeyError) as exc:
                    self.logger.warning(
                        "处理微博条目时出错，文件: %s，错误: %s", path, exc
                    )
                    continue

    def _clean_text(self, text: str) -> str:
        """对微博正文进行基础清洗。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        text = text.replace(ZERO_WIDTH_SPACE, " ")
        text = LINK_PLACEHOLDER_PATTERN.sub("", text)
        text = URL_PATTERN.sub("", text)
        text = CONTROL_CHAR_PATTERN.sub(" ", text)
        return text.strip()

    def _chunk_text(self, text: str) -> Iterable[str]:
        """根据配置将文本切分为多个块。

        Args:
            text: 待切分的文本

        Yields:
            文本块
        """
        if not text:
            return

        if len(text) <= self.chunk_size:
            yield text
            return

        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():  # 只返回非空块
                yield chunk
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)


