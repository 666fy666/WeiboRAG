"""RAG 核心模块包。"""

from .data_preparation import DataPreparationPipeline, WeiboDocument
from .generation_integration import DeepSeekGenerator
from .index_construction import IndexBuilder
from .llm_providers import (
    LLMGenerator,
    create_llm_generator,
    DeepSeekGenerator as DeepSeekGeneratorNew,
    OpenAIGenerator,
    QwenGenerator,
)
from .retrieval_optimization import Retriever, RetrievalResult

__all__ = [
    "DataPreparationPipeline",
    "WeiboDocument",
    "IndexBuilder",
    "Retriever",
    "RetrievalResult",
    "DeepSeekGenerator",  # 向后兼容
    "LLMGenerator",
    "create_llm_generator",
    "DeepSeekGeneratorNew",
    "OpenAIGenerator",
    "QwenGenerator",
]


