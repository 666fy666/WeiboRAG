"""配置管理模块。

该模块定义应用级配置，支持环境变量覆盖默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AppConfig:
    """应用运行所需的核心配置。"""

    #: 微博 JSON 数据的根目录路径，默认指向 `weibo/`。
    data_root: Path = Path(os.getenv("WEIBO_DATA_ROOT", "weibo"))

    #: 用于缓存向量索引及元数据的目录。
    vector_dir: Path = Path(os.getenv("VECTOR_CACHE_DIR", "vector_index"))

    #: 向量索引文件的完整路径（FAISS）。
    vector_store_path: Path = Path(
        os.getenv("VECTOR_STORE_PATH", "vector_index/persona.index")
    )

    #: 文档元数据存储文件的路径（JSON）。
    metadata_store_path: Path = Path(
        os.getenv("METADATA_STORE_PATH", "vector_index/persona_metadata.json")
    )

    #: ModelScope 上的嵌入模型仓库名称。
    embedding_model_repo: str = os.getenv(
        "EMBEDDING_MODEL_REPO", "BAAI/bge-small-zh-v1.5"
    )

    #: Transformers 加载的嵌入模型名称。
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"
    )

    #: ModelScope 模型缓存目录，若未设置则使用默认缓存位置。
    modelscope_cache: Path | None = (
        Path(cache_dir) if (cache_dir := os.getenv("MODEL_SCOPE_CACHE")) else None
    )

    #: 文本切分时的最大 token 数（字符近似）。
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "320"))

    #: 邻接文本块的重叠字符数。
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))

    #: 过滤短内容的阈值，低于该字符数的文本将被丢弃。
    min_content_chars: int = int(os.getenv("MIN_CONTENT_CHARS", "30"))

    #: 检索阶段返回的上下文数量。
    top_k: int = int(os.getenv("TOP_K", "10"))

    #: Rerank 候选集的规模。
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "18"))

    #: BM25 检索候选集规模。
    bm25_top_k: int = int(
        os.getenv("BM25_TOP_K", os.getenv("RERANK_TOP_K", "18"))
    )

    #: RRF 融合时的平滑参数。
    rrf_k: float = float(os.getenv("RRF_K", "60"))

    #: DeepSeek API 的访问密钥。
    deepseek_api_key: str = os.getenv(
        "DEEPSEEK_API_KEY", "sk-754ddb7544314d708b11b1337271f9af"
    )

    #: DeepSeek API 调用地址。
    deepseek_api_url: str = os.getenv(
        "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
    )

    #: 模型生成阶段允许的最大新增 token 数。
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "1024"))

    #: 生成阶段的温度参数，数值越高输出越随机。
    temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))

    #: 是否优先使用 GPU 版本的 FAISS。
    faiss_use_gpu: bool = _bool_env("FAISS_USE_GPU", True)

    #: 指定用于 FAISS 的 GPU 设备编号。
    faiss_gpu_device: int = int(os.getenv("FAISS_GPU_DEVICE", "0"))

    #: LLM 对话的系统提示词，可通过环境变量 `SYSTEM_PROMPT` 覆盖。
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        (
            "你是黄霄雲（1998年12月22日出生于贵州省黔南布依族苗族自治州，中国内地流行乐女歌手、唱作人，"
            "毕业于中央音乐学院）基于你发布的微博历史内容回答粉丝的提问。在对话中："
            "- 发挥想象力，生成有趣、丰富、有深度的回复内容。"
            "- 粉丝可能会叫你面包,你可能会称呼粉丝为魔星宝宝们。"
            "- 如果对话中出现角色无法回答的问题，学会表达困惑并寻求澄清，不要试图编造答案。"
            "请严格依据提供的片段作答，如无法找到答案请明确说明。"
        ),
    )

    def ensure_directories(self) -> None:
        """确保运行所需的目录存在。"""

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    """载入配置并准备环境。"""

    config = AppConfig()
    config.ensure_directories()
    return config


