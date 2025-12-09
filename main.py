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
    IndexBuilder,
    LLMGenerator,
    Retriever,
    create_llm_generator,
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
    parser.add_argument(
        "--web",
        action="store_true",
        help="启动 Web 服务器模式",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web 服务器端口（默认：8000）",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main(argv: List[str] | None = None) -> int:
    """主程序入口。

    Args:
        argv: 命令行参数列表，默认为 None 时会从 sys.argv 读取

    Returns:
        退出代码，0 表示成功，非 0 表示错误
    """
    args = build_arg_parser().parse_args(argv)
    configure_logging()
    logger = logging.getLogger("WeiboRAG")

    # Web 模式
    if args.web:
        logger.info("启动 Web 服务器模式...")
        try:
            import uvicorn
            from web_app import app

            uvicorn.run(app, host="0.0.0.0", port=args.port)
        except ImportError as exc:
            logger.error("Web 模式需要安装 uvicorn: %s", exc)
            logger.error("请运行: uv pip install uvicorn[standard]")
            return 1
        return 0

    try:
        config = load_config()
    except Exception as exc:
        logger.error("加载配置失败: %s", exc, exc_info=True)
        return 1

    logger.info("数据目录: %s", config.data_root)

    try:
        data_pipeline = DataPreparationPipeline(
            data_root=config.data_root,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_content_chars=config.min_content_chars,
        )
        corpus = data_pipeline.prepare_corpus()
    except Exception as exc:
        logger.error("数据准备失败: %s", exc, exc_info=True)
        return 1

    if not corpus:
        logger.error("未生成任何文本块，请检查数据集")
        return 1

    try:
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
        index, payloads = index_builder.build_or_load(
            corpus, rebuild=args.rebuild_index
        )
    except Exception as exc:
        logger.error("索引构建或加载失败: %s", exc, exc_info=True)
        return 1

    try:
        retriever = Retriever(
            index=index,
            payloads=payloads,
            encoder=encoder,
            top_k=args.top_k or config.top_k,
            rerank_top_k=config.rerank_top_k,
            bm25_top_k=config.bm25_top_k,
            rrf_k=config.rrf_k,
        )
    except Exception as exc:
        logger.error("检索器初始化失败: %s", exc, exc_info=True)
        return 1

    try:
        # 向后兼容：如果配置了deepseek_api_key但没有配置llm_api_key，使用deepseek_api_key
        api_key = config.llm_api_key
        if not api_key and config.deepseek_api_key:
            api_key = config.deepseek_api_key
            logger.info("使用向后兼容的 DEEPSEEK_API_KEY")

        api_url = config.llm_api_url
        if not api_url and config.deepseek_api_url:
            api_url = config.deepseek_api_url

        generator = create_llm_generator(
            provider=config.llm_provider,
            api_key=api_key,
            api_url=api_url,
            model_name=config.llm_model_name,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            system_prompt=config.system_prompt,
        )
        logger.info("使用 LLM 提供商: %s", config.llm_provider)
    except Exception as exc:
        logger.error("生成器初始化失败: %s", exc, exc_info=True)
        return 1

    if not args.query:
        logger.info(
            "未提供 query，进入交互式模式（输入 exit 退出，输入 clear 清空对话历史）"
        )
        conversation_history: List[Dict[str, str]] = []
        try:
            while True:
                try:
                    user_query = input("请输入问题: ").strip()
                except (EOFError, KeyboardInterrupt):
                    logger.info("用户中断输入")
                    break

                if not user_query:
                    continue

                if user_query.lower() in {"exit", "quit", "q"}:
                    logger.info("退出交互式模式")
                    break

                if user_query.lower() == "clear":
                    conversation_history.clear()
                    logger.info("对话历史已清空")
                    continue

                try:
                    answer = _run_query(
                        user_query,
                        retriever,
                        generator,
                        show_context=args.show_context,
                        conversation_history=conversation_history,
                    )
                    # 将当前对话添加到历史中
                    if answer:
                        conversation_history.append(
                            {"role": "user", "content": user_query}
                        )
                        conversation_history.append(
                            {"role": "assistant", "content": answer}
                        )
                except Exception as exc:
                    logger.error("处理查询时出错: %s", exc, exc_info=True)
                    print(f"\n错误: {exc}")

        except Exception as exc:
            logger.error("交互式模式运行出错: %s", exc, exc_info=True)
            return 1

        return 0

    try:
        _run_query(args.query, retriever, generator, show_context=args.show_context)
    except Exception as exc:
        logger.error("执行查询失败: %s", exc, exc_info=True)
        return 1

    return 0


def _run_query(
    query: str,
    retriever: Retriever,
    generator: LLMGenerator,
    show_context: bool,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """执行查询并返回回答。

    Args:
        query: 用户查询
        retriever: 检索器
        generator: 生成器
        show_context: 是否显示检索到的上下文
        conversation_history: 可选的对话历史

    Returns:
        生成的回答，如果未检索到相关内容或生成失败则返回 None

    Raises:
        ValueError: 如果查询为空
        RuntimeError: 如果检索或生成失败
    """
    logger = logging.getLogger("WeiboRAG.Query")

    if not query or not query.strip():
        raise ValueError("查询不能为空")

    try:
        results = retriever.search(query)
    except Exception as exc:
        logger.error("检索失败: %s", exc, exc_info=True)
        raise RuntimeError("检索失败") from exc

    if not results:
        logger.info("未检索到相关内容")
        return None

    if show_context:
        logger.info("检索到 %d 个相关片段:", len(results))
        for idx, item in enumerate(results, start=1):
            logger.info(
                "片段 %d | 得分 %.4f | 账号 %s | 时间 %s",
                idx,
                item.score,
                item.metadata.get("account_name", "未知"),
                item.metadata.get("publish_time", "未知"),
            )
            logger.info("内容: %s", item.text[:200] + "..." if len(item.text) > 200 else item.text)

    try:
        answer_payload = generator.generate(
            query, results, conversation_history=conversation_history
        )
    except Exception as exc:
        logger.error("生成回答失败: %s", exc, exc_info=True)
        raise RuntimeError("生成回答失败") from exc

    answer = answer_payload.get("answer", "")
    print("\n=== 答复 ===")
    print(answer)
    return answer


if __name__ == "__main__":
    sys.exit(main())
