"""FastAPI Web 应用。

提供 REST API 端点和 Web 界面用于 RAG 问答。
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import load_config
from rag_modules import (
    DataPreparationPipeline,
    IndexBuilder,
    LLMGenerator,
    Retriever,
    create_llm_generator,
)
from rag_modules.index_construction import BGESentenceEncoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("WeiboRAG.WebApp")

# 全局变量存储应用状态
app_state: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("正在初始化应用...")
    try:
        config = load_config()
        logger.info("配置加载成功")

        # 数据准备
        logger.info("准备数据...")
        data_pipeline = DataPreparationPipeline(
            data_root=config.data_root,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_content_chars=config.min_content_chars,
        )
        corpus = data_pipeline.prepare_corpus()
        if not corpus:
            raise RuntimeError("未生成任何文本块，请检查数据集")

        # 索引构建
        logger.info("构建/加载索引...")
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
        index, payloads = index_builder.build_or_load(corpus, rebuild=False)

        # 检索器初始化
        logger.info("初始化检索器...")
        retriever = Retriever(
            index=index,
            payloads=payloads,
            encoder=encoder,
            top_k=config.top_k,
            rerank_top_k=config.rerank_top_k,
            bm25_top_k=config.bm25_top_k,
            rrf_k=config.rrf_k,
        )

        # LLM 生成器初始化
        logger.info("初始化 LLM 生成器...")
        api_key = config.llm_api_key
        if not api_key and config.deepseek_api_key:
            api_key = config.deepseek_api_key

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

        # 存储到应用状态
        app_state["config"] = config
        app_state["retriever"] = retriever
        app_state["generator"] = generator

        logger.info("应用初始化完成，使用 LLM 提供商: %s", config.llm_provider)
        yield

    except Exception as exc:
        logger.error("应用初始化失败: %s", exc, exc_info=True)
        raise
    finally:
        logger.info("应用正在关闭...")


app = FastAPI(
    title="WeiboRAG API",
    description="微博人物多账号RAG问答系统 API",
    version="0.1.0",
    lifespan=lifespan,
)


# Pydantic 模型
class QueryRequest(BaseModel):
    """查询请求模型。"""

    query: str = Field(..., description="用户查询问题", min_length=1)
    top_k: Optional[int] = Field(None, description="返回的上下文数量（覆盖配置）")
    show_context: bool = Field(False, description="是否返回检索到的上下文")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="对话历史"
    )


class QueryResponse(BaseModel):
    """查询响应模型。"""

    answer: str = Field(..., description="生成的回答")
    contexts: Optional[List[Dict[str, object]]] = Field(
        None, description="检索到的上下文片段"
    )
    success: bool = Field(True, description="请求是否成功")
    message: Optional[str] = Field(None, description="错误消息（如果有）")


class HealthResponse(BaseModel):
    """健康检查响应模型。"""

    status: str = Field(..., description="服务状态")
    llm_provider: str = Field(..., description="当前使用的 LLM 提供商")


# API 端点
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回 Web 界面。"""
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeiboRAG - 微博人物多账号RAG问答系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            text-align: right;
        }
        .message.assistant {
            text-align: left;
        }
        .message-content {
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .message.user .message-content {
            background: #667eea;
            color: white;
        }
        .message.assistant .message-content {
            background: white;
            color: #333;
            border: 1px solid #e9ecef;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            border-color: #667eea;
        }
        button {
            padding: 15px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
            font-weight: 600;
        }
        button:hover {
            background: #5568d3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            color: #666;
            margin: 10px 0;
        }
        .loading.show {
            display: block;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .clear-btn {
            background: #6c757d;
            margin-left: 10px;
        }
        .clear-btn:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 WeiboRAG</h1>
        <p class="subtitle">微博人物多账号RAG问答系统</p>
        
        <div id="error-container"></div>
        
        <div class="chat-container" id="chat-container">
            <div class="message assistant">
                <div class="message-content">
                    你好！我是基于微博历史内容的智能问答助手。你可以问我任何关于微博内容的问题。
                </div>
            </div>
        </div>
        
        <div class="loading" id="loading">正在思考...</div>
        
        <div class="input-container">
            <input type="text" id="query-input" placeholder="输入你的问题..." autocomplete="off">
            <button id="send-btn" onclick="sendQuery()">发送</button>
            <button class="clear-btn" id="clear-btn" onclick="clearHistory()">清空</button>
        </div>
    </div>

    <script>
        let conversationHistory = [];

        const queryInput = document.getElementById('query-input');
        const sendBtn = document.getElementById('send-btn');
        const clearBtn = document.getElementById('clear-btn');
        const chatContainer = document.getElementById('chat-container');
        const loading = document.getElementById('loading');
        const errorContainer = document.getElementById('error-container');

        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendQuery();
            }
        });

        function showError(message) {
            errorContainer.innerHTML = `<div class="error">${message}</div>`;
            setTimeout(() => {
                errorContainer.innerHTML = '';
            }, 5000);
        }

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendQuery() {
            const query = queryInput.value.trim();
            if (!query) {
                return;
            }

            // 添加用户消息
            addMessage('user', query);
            queryInput.value = '';
            sendBtn.disabled = true;
            loading.classList.add('show');

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        conversation_history: conversationHistory,
                    }),
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.message || '请求失败');
                }

                if (data.success) {
                    addMessage('assistant', data.answer);
                    // 更新对话历史
                    conversationHistory.push({ role: 'user', content: query });
                    conversationHistory.push({ role: 'assistant', content: data.answer });
                } else {
                    throw new Error(data.message || '生成回答失败');
                }
            } catch (error) {
                showError(`错误: ${error.message}`);
                console.error('Error:', error);
            } finally {
                sendBtn.disabled = false;
                loading.classList.remove('show');
            }
        }

        function clearHistory() {
            conversationHistory = [];
            chatContainer.innerHTML = `
                <div class="message assistant">
                    <div class="message-content">
                        对话历史已清空。你可以继续提问。
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点。"""
    config = app_state.get("config")
    if not config:
        raise HTTPException(status_code=503, detail="应用未初始化")
    return HealthResponse(
        status="healthy",
        llm_provider=config.llm_provider,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询端点。

    Args:
        request: 查询请求

    Returns:
        查询响应
    """
    retriever = app_state.get("retriever")
    generator = app_state.get("generator")

    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        # 检索
        top_k = request.top_k
        if top_k:
            # 临时修改检索器的 top_k
            original_top_k = retriever.top_k
            retriever.top_k = top_k
            results = retriever.search(request.query)
            retriever.top_k = original_top_k
        else:
            results = retriever.search(request.query)

        if not results:
            return QueryResponse(
                answer="抱歉，未检索到相关内容。",
                success=True,
            )

        # 生成回答
        answer_payload = generator.generate(
            request.query,
            results,
            conversation_history=request.conversation_history,
        )

        answer = answer_payload.get("answer", "")
        contexts = None
        if request.show_context:
            contexts = answer_payload.get("contexts", [])

        return QueryResponse(
            answer=answer,
            contexts=contexts,
            success=True,
        )

    except Exception as exc:
        logger.error("查询处理失败: %s", exc, exc_info=True)
        return QueryResponse(
            answer="",
            success=False,
            message=str(exc),
        )


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

