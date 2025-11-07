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


_CONTROL_CHAR_PATTERN = re.compile(r"\s+")
_LINK_PLACEHOLDER_PATTERN = re.compile(r"网页链接|原图|\[组图共\d+张]\s*")


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
        """读取数据并生成文本块列表。"""

        documents = list(self._load_documents())
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
            account_id = str(user_info.get("id", ""))
            account_name = str(user_info.get("nickname", "未知账号"))
            weibo_items = payload.get("weibo", [])

            for item in weibo_items:
                content = self._clean_text(str(item.get("content", "")))
                if len(content) < self.min_content_chars:
                    continue

                stats = {
                    "up_num": int(item.get("up_num", 0)),
                    "retweet_num": int(item.get("retweet_num", 0)),
                    "comment_num": int(item.get("comment_num", 0)),
                }
                extra = {
                    "publish_tool": str(item.get("publish_tool", "")),
                    "publish_place": str(item.get("publish_place", "")),
                }
                yield WeiboDocument(
                    doc_id=str(item.get("id", "")) or f"{account_id}_{item.get('publish_time', '')}",
                    account_id=account_id,
                    account_name=account_name,
                    publish_time=str(item.get("publish_time", "")),
                    content=content,
                    source_path=path,
                    stats=stats,
                    extra=extra,
                )

    def _clean_text(self, text: str) -> str:
        """对微博正文进行基础清洗。"""

        text = text.replace("\u200b", " ")
        text = _LINK_PLACEHOLDER_PATTERN.sub("", text)
        text = re.sub(r"https?://\S+", "", text)
        text = _CONTROL_CHAR_PATTERN.sub(" ", text)
        return text.strip()

    def _chunk_text(self, text: str) -> Iterable[str]:
        """根据配置将文本切分为多个块。"""

        if len(text) <= self.chunk_size:
            yield text
            return

        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            yield chunk
            if end == len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)


