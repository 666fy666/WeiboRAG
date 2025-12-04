"""索引构建模块。

负责加载嵌入模型、生成向量并构建 FAISS 索引。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class BGESentenceEncoder:
    """BGE-small-zh-v1.5 句向量编码器。"""

    def __init__(
        self,
        model_repo: str,
        model_name: str,
        cache_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_repo = model_repo
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self.logger = logging.getLogger(f"{__name__}.BGESentenceEncoder")

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            model_path = self._download_model()
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return self._tokenizer

    @property
    def model(self) -> AutoModel:
        if self._model is None:
            model_path = self._download_model()
            self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def embed(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        """对文本批量编码并返回归一化后的向量。

        Args:
            texts: 文本迭代器
            batch_size: 批处理大小

        Returns:
            归一化后的向量矩阵，形状为 (n_texts, embedding_dim)

        Raises:
            ValueError: 如果 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前值: {batch_size}")

        vectors: List[np.ndarray] = []
        batch: List[str] = []
        for text in texts:
            if not text or not text.strip():
                self.logger.debug("跳过空文本")
                continue
            batch.append(text)
            if len(batch) >= batch_size:
                try:
                    vectors.append(self._encode_batch(batch))
                except Exception as exc:
                    self.logger.error("批量编码失败，错误: %s", exc)
                    raise
                batch = []

        if batch:
            vectors.append(self._encode_batch(batch))

        if not vectors:
            return np.empty((0, 0), dtype="float32")

        return np.vstack(vectors)

    def _download_model(self) -> str:
        """下载模型到缓存目录。

        Returns:
            模型路径

        Raises:
            RuntimeError: 如果模型下载失败
        """
        try:
            cache_path = snapshot_download(
                self.model_repo,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            if not cache_path:
                self.logger.warning(
                    "ModelScope 下载返回空路径，使用模型名称: %s", self.model_name
                )
                return self.model_name
            return cache_path
        except Exception as exc:
            self.logger.error("下载模型失败: %s，错误: %s", self.model_repo, exc)
            raise RuntimeError(f"无法下载模型 {self.model_repo}") from exc

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        tokenizer = self.tokenizer
        model = self.model
        with torch.no_grad():
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            model_output = model(**encoded)
            last_hidden_state = model_output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            sum_embeddings = (last_hidden_state * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().astype("float32")


class IndexBuilder:
    """构建与加载 FAISS 索引的管理器。"""

    def __init__(
        self,
        vector_store_path: Path,
        metadata_store_path: Path,
        encoder: BGESentenceEncoder,
        use_gpu: bool = True,
        gpu_device: int = 0,
    ) -> None:
        self.vector_store_path = vector_store_path
        self.metadata_store_path = metadata_store_path
        self.encoder = encoder
        self.logger = logging.getLogger(f"{__name__}.IndexBuilder")
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self._gpu_resources: Any | None = None

    def build_or_load(
        self,
        corpus: List[Dict[str, str]],
        rebuild: bool = False,
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """若缓存存在则加载，否则重新构建索引。"""

        if not rebuild and self._cache_available():
            index, metadata = self._load_cache()
            if len(metadata) == len(corpus) and index.ntotal == len(corpus):
                self.logger.info("复用现有索引，共 %d 条向量", index.ntotal)
                index = self._maybe_to_gpu(index)
                return index, metadata
            self.logger.info("检测到文档数量变化，将重建索引")

        return self._rebuild_index(corpus)

    def _cache_available(self) -> bool:
        return self.vector_store_path.exists() and self.metadata_store_path.exists()

    def _load_cache(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """加载缓存的索引和元数据。

        Returns:
            索引对象和元数据列表

        Raises:
            RuntimeError: 如果加载失败
        """
        try:
            index = faiss.read_index(str(self.vector_store_path))
        except Exception as exc:
            raise RuntimeError(
                f"无法读取向量索引文件: {self.vector_store_path}"
            ) from exc

        try:
            with self.metadata_store_path.open("r", encoding="utf-8") as fp:
                metadata: List[Dict[str, Any]] = json.load(fp)
        except (IOError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"无法读取元数据文件: {self.metadata_store_path}"
            ) from exc

        if not isinstance(metadata, list):
            raise RuntimeError(f"元数据格式错误，应为列表: {self.metadata_store_path}")

        return index, metadata

    def _rebuild_index(
        self, corpus: List[Dict[str, Any]]
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """重新构建索引。

        Args:
            corpus: 语料列表

        Returns:
            索引对象和元数据列表

        Raises:
            ValueError: 如果语料为空或格式错误
            RuntimeError: 如果构建过程失败
        """
        if not corpus:
            raise ValueError("传入的语料为空，无法构建索引")

        texts = []
        payloads = []
        for item in corpus:
            if not isinstance(item, dict):
                self.logger.warning("跳过非字典类型的语料项")
                continue
            text = item.get("text", "")
            if not text or not text.strip():
                self.logger.debug("跳过空文本块")
                continue
            texts.append(text)
            payloads.append(item)

        if not texts:
            raise ValueError("没有有效的文本块可用于构建索引")

        self.logger.info("开始编码 %d 个文本块", len(texts))
        try:
            embeddings = self.encoder.embed(texts)
        except Exception as exc:
            raise RuntimeError("文本编码失败") from exc

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"嵌入结果形状错误，期望 2 维，实际 {embeddings.ndim} 维"
            )

        if embeddings.shape[0] == 0:
            raise RuntimeError("编码后没有有效向量")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        try:
            self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self.vector_store_path))
        except Exception as exc:
            raise RuntimeError(
                f"无法写入向量索引文件: {self.vector_store_path}"
            ) from exc

        try:
            with self.metadata_store_path.open("w", encoding="utf-8") as fp:
                json.dump(payloads, fp, ensure_ascii=False, indent=2)
        except IOError as exc:
            raise RuntimeError(
                f"无法写入元数据文件: {self.metadata_store_path}"
            ) from exc

        self.logger.info("索引构建完成，共 %d 条向量", index.ntotal)
        index = self._maybe_to_gpu(index)
        return index, payloads

    def _maybe_to_gpu(self, index: faiss.Index) -> faiss.Index:
        if not self.use_gpu:
            return index
        if not torch.cuda.is_available():
            self.logger.info("CUDA 不可用，继续使用 CPU 索引")
            return index
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
            self.logger.warning("当前 FAISS 未编译 GPU 支持，保持 CPU 索引")
            return index
        try:
            if self._gpu_resources is None:
                self._gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, self.gpu_device, index)
            self.logger.info("索引已迁移至 GPU device=%d", self.gpu_device)
            return gpu_index
        except Exception as exc:
            self.logger.warning("迁移索引到 GPU 失败，将回退至 CPU 索引: %s", exc)
            return index


