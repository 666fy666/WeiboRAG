"""主程序入口。

执行数据准备、索引构建与问答流程。
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional

from config import load_config
from rag_modules import (
    DataPreparationPipeline,
    DeepSeekGenerator,
    IndexBuilder,
    Retriever,
)
from rag_modules.index_construction import BGESentenceEncoder


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于微博内容的 RAG 问答系统")
    parser.add_argument("--query", type=str, help="需要回答的问题", required=False)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建向量索引",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="返回的上下文数量（默认读取配置）",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="是否在控制台输出检索到的上下文",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()
    logger = logging.getLogger("WeiboRAG")

    config = load_config()
    logger.info("数据目录: %s", config.data_root)

    data_pipeline = DataPreparationPipeline(
        data_root=config.data_root,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        min_content_chars=config.min_content_chars,
    )
    corpus = data_pipeline.prepare_corpus()
    if not corpus:
        logger.error("未生成任何文本块，请检查数据集")
        return 1

    encoder = BGESentenceEncoder(
        model_repo=config.embedding_model_repo,
        model_name=config.embedding_model_name,
        cache_dir=config.modelscope_cache,
    )
    index_builder = IndexBuilder(
        vector_store_path=config.vector_store_path,
        metadata_store_path=config.metadata_store_path,
        encoder=encoder,
        use_gpu=config.faiss_use_gpu,
        gpu_device=config.faiss_gpu_device,
    )
    index, payloads = index_builder.build_or_load(corpus, rebuild=args.rebuild_index)

    retriever = Retriever(
        index=index,
        payloads=payloads,
        encoder=encoder,
        top_k=args.top_k or config.top_k,
        rerank_top_k=config.rerank_top_k,
        bm25_top_k=config.bm25_top_k,
        rrf_k=config.rrf_k,
    )

    generator = DeepSeekGenerator(
        api_key=config.deepseek_api_key,
        api_url=config.deepseek_api_url,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        system_prompt=config.system_prompt,
    )

    if not args.query:
        logger.info("未提供 query，进入交互式模式（输入 exit 退出，输入 clear 清空对话历史）")
        conversation_history: List[Dict[str, str]] = []
        while True:
            user_query = input("请输入问题: ").strip()
            if user_query.lower() in {"exit", "quit"}:
                break
            if user_query.lower() == "clear":
                conversation_history.clear()
                logger.info("对话历史已清空")
                continue
            answer = _run_query(
                user_query, 
                retriever, 
                generator, 
                show_context=args.show_context,
                conversation_history=conversation_history
            )
            # 将当前对话添加到历史中
            if answer:
                conversation_history.append({"role": "user", "content": user_query})
                conversation_history.append({"role": "assistant", "content": answer})
        return 0

    _run_query(args.query, retriever, generator, show_context=args.show_context)
    return 0


def _run_query(
    query: str, 
    retriever: Retriever, 
    generator: DeepSeekGenerator, 
    show_context: bool,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Optional[str]:
    """执行查询并返回回答。
    
    Args:
        query: 用户查询
        retriever: 检索器
        generator: 生成器
        show_context: 是否显示检索到的上下文
        conversation_history: 可选的对话历史
        
    Returns:
        生成的回答，如果未检索到相关内容则返回 None
    """
    logger = logging.getLogger("WeiboRAG.Query")
    results = retriever.search(query)
    if not results:
        logger.info("未检索到相关内容")
        return None

    if show_context:
        for idx, item in enumerate(results, start=1):
            logger.info(
                "片段 %d | 得分 %.4f | 账号 %s | 时间 %s",
                idx,
                item.score,
                item.metadata.get("account_name", "未知"),
                item.metadata.get("publish_time", "未知"),
            )
            logger.info("内容: %s", item.text)

    answer_payload = generator.generate(query, results, conversation_history=conversation_history)
    answer = answer_payload.get("answer", "")
    print("\n=== 答复 ===")
    print(answer)
    return answer
"""     print("\n=== 参考片段 ===")
    for idx, ctx in enumerate(answer_payload.get("contexts", []), start=1):
        print(
            f"[{idx}] {ctx['metadata'].get('account_name', '未知')} | {ctx['metadata'].get('publish_time', '未知')}"
        ) """


if __name__ == "__main__":
    sys.exit(main())


