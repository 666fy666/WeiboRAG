WeiboRAG —— 微博人物多账号RAG问答系统
=======================================

项目简介
--------

WeiboRAG 用于汇总同一人物多个微博账号的历史内容，构建基于检索增强生成（RAG）的问答能力。系统会遍历 `weibo/` 目录下所有子文件夹中的 JSON 数据，自动抽取微博文本、元数据和多媒体描述，构建向量索引，并通过大语言模型提供问答服务。

本项目遵循 Python 最佳实践，采用模块化设计、完善的错误处理、类型提示和详细的日志记录，确保代码的可维护性和健壮性。

目录结构
--------

```text
├── config.py                   # 配置管理
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖列表
├── rag_modules/                # 核心模块
│   ├── __init__.py
│   ├── data_preparation.py     # 数据准备模块
│   ├── index_construction.py   # 索引构建模块
│   ├── retrieval_optimization.py # 检索优化模块
│   └── generation_integration.py # 生成集成模块
├── vector_index/               # 向量索引缓存（运行时自动生成）
└── weibo/                      # 原始微博 JSON 数据
```

快速开始
--------

1. **安装 uv（如果尚未安装）**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # 或者使用 pip
   pip install uv
   ```

2. **安装依赖**

   ```bash
   # 使用 uv 创建虚拟环境并安装依赖
   uv venv
   source .venv/bin/activate  # Linux/macOS
   # 或 .venv\Scripts\activate  # Windows
   uv pip install -e .
   ```

   如果需要使用 GPU 版本的 FAISS（仅限 Linux x86_64）：

   ```bash
   uv pip install -e ".[gpu]"
   ```

   或者直接使用 uv 运行（无需激活虚拟环境）：

   ```bash
   uv run python main.py --query "你的宇宙无敌号开到了哪些城市？"
   ```

2. **配置环境变量**

   创建 `.env` 文件（或设置系统环境变量）：

   ```bash
   # LLM 提供商配置（选择一种）
   LLM_PROVIDER=deepseek  # 可选值: deepseek, openai, qwen
   LLM_API_KEY=your_api_key_here
   LLM_API_URL=  # 可选，使用提供商默认值
   LLM_MODEL_NAME=  # 可选，使用提供商默认值

   # 或者使用向后兼容的配置（DeepSeek）
   DEEPSEEK_API_KEY=your_api_key_here
   DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions

   # 其他可选配置
   MODEL_SCOPE_CACHE=/path/to/cache
   ```

   必需的环境变量：
   - `LLM_API_KEY` 或 `DEEPSEEK_API_KEY`：LLM API 密钥（根据选择的提供商）

   更多配置选项请参考下方的"环境变量配置"章节。

3. **运行主程序**

   ```bash
   uv run python main.py --query "你的宇宙无敌号开到了哪些城市？" --top-k 10 --show-context
   ```

   常用参数说明（均可组合使用）：

   - `--query`：提出需要回答的问题。例如：

     ```bash
     uv run python main.py --query "你的宇宙无敌号开到了哪些城市？"
     ```

   - `--rebuild-index`：强制重新构建向量索引，适用于新增/修改微博数据后。例如：

     ```bash
     uv run python main.py --rebuild-index
     ```

   - `--top-k`：设置返回的上下文数量（覆盖配置文件中的默认值）。例如：

     ```bash
     uv run python main.py --query "粉丝互动情况怎样？" --top-k 8
     ```

   - `--show-context`：在控制台打印检索到的原始片段，便于调试。例如：

     ```bash
     uv run python main.py --query "最近发布的公益活动有哪些？" --show-context
     ```

   - **交互式模式**：若不提供 `--query`，程序会进入命令行问答循环，支持多轮对话：

     ```bash
     uv run python main.py
     ```

     - 输入 `exit`、`quit` 或 `q` 退出程序
     - 输入 `clear` 清空对话历史
     - 支持多轮对话，系统会记住上下文
     - 按 `Ctrl+C` 可中断输入

   - **Web 模式**：启动 Web 服务器，提供 Web 界面和 REST API：

     ```bash
     uv run python main.py --web
     # 或指定端口
     uv run python main.py --web --port 8081
     ```

     启动后访问 `http://localhost:8000` 即可使用 Web 界面。

     也可以直接运行：

     ```bash
     uv run python web_app.py
     # 或指定端口
     uv run python web_app.py 8080
     ```

配置说明
--------

`config.py` 暴露 `AppConfig` 数据类，主要字段包括：

- `data_root`：微博 JSON 数据根目录，默认 `weibo/`。
- `vector_store_path`：FAISS 索引文件存放位置，默认 `vector_index/persona.index`。
- `embedding_model_repo`：ModelScope 仓库名，默认 `BAAI/bge-small-zh-v1.5`。
- `embedding_model_name`：HF 模型名称，用于 Transformers 加载同一模型。
- `chunk_size` / `chunk_overlap`：文本切分长度与重叠率。
- `top_k`：检索返回文档数量。
- `llm_provider`：LLM 提供商（deepseek、openai、qwen），默认 `deepseek`。
- `llm_api_key`：LLM API 密钥（根据提供商选择）。
- `llm_api_url`：LLM API 端点 URL（可选，使用提供商默认值）。
- `llm_model_name`：LLM 模型名称（可选，使用提供商默认值）。
- `deepseek_api_key`：DeepSeek 调用密钥（向后兼容，已废弃）。
- `rerank_top_k`：向量检索候选集规模，用于后续融合。
- `bm25_top_k`：BM25 候选集规模，缺省与 `rerank_top_k` 一致。
- `rrf_k`：RRF 融合的平滑系数，越大代表对长尾排名更“保守”。
- `faiss_use_gpu`：是否优先启用 GPU 版 FAISS（默认自动开启，若无 GPU 会自动回退 CPU）。
- `faiss_gpu_device`：GPU 编号，默认为 `0`。

如需修改，直接在 `config.py` 中调整或通过环境变量覆盖。

模块概览
--------

- **`rag_modules.data_preparation`** - 数据准备模块
  - 递归扫描 `data_root`，解析 JSON 中的 `user` 与 `weibo` 字段
  - 构建标准化 `WeiboDocument` 数据结构，包含文本、时间、互动信息及来源账号
  - 对内容进行清洗（去重、去除空文本、替换链接占位符等），并使用可配置的文本切分策略生成段落
  - 包含完善的错误处理和日志记录，支持数据验证和异常恢复

- **`rag_modules.index_construction`** - 索引构建模块
  - 使用 ModelScope 下载并缓存 BGE-small-zh-v1.5 模型
  - 通过 HuggingFace Transformers 加载同一模型权重，生成文本嵌入
  - 构建或增量更新 FAISS 向量索引，持久化至 `vector_store_path`
  - 支持 GPU/CPU 自动切换，包含模型下载失败重试机制

- **`rag_modules.retrieval_optimization`** - 检索优化模块
  - 结合向量检索与 BM25 关键词检索，利用 RRF（Reciprocal Rank Fusion）融合候选
  - `Retriever` 同时返回各通道得分与排名，便于后续调试和监控
  - 包含词汇匹配提升机制，优化检索结果相关性
  - 支持空查询检测和异常处理

- **`rag_modules.generation_integration`** - 生成集成模块（已废弃，使用 `llm_providers`）
  - 封装 DeepSeek API 的调用逻辑，支持上下文提示模板和智能重试
  - 接收检索结果，生成最终回答与支持证据
  - 实现指数退避重试机制，区分不同异常类型（超时、HTTP 错误等）
  - 支持多轮对话历史管理

- **`rag_modules.llm_providers`** - LLM 提供商模块
  - 提供抽象的 `LLMGenerator` 基类，支持多种 LLM 提供商
  - 实现 DeepSeek、OpenAI、通义千问（Qwen）等提供商
  - 统一的接口和错误处理机制
  - 支持工厂模式创建不同提供商的生成器

- **`web_app`** - Web 应用模块
  - 基于 FastAPI 的 REST API 服务
  - 提供 Web 界面用于交互式问答
  - 支持对话历史管理和上下文显示
  - 健康检查端点

运行流程
--------

1. **数据准备**：`DataPreparationPipeline` 从 JSON 文件生成标准文本块。
2. **索引构建**：`IndexBuilder` 根据文本块创建/更新 FAISS 向量索引，若检测到 GPU 且安装了 `faiss-gpu`，会自动将索引迁移到 GPU 提速检索。
3. **检索优化**：`Retriever` 同时运行向量检索与 BM25 检索，并通过 RRF 融合选出最相关片段。
4. **生成集成**：`LLMGenerator`（根据配置选择提供商）组合上下文，通过 LLM API 生成回答。

代码特性
--------

- **类型安全**：全面使用类型提示，提高代码可读性和 IDE 支持
- **错误处理**：完善的异常处理机制，包含详细的错误日志和友好的错误消息
- **日志记录**：统一的日志系统，支持不同级别的日志输出，便于调试和监控
- **模块化设计**：清晰的模块划分，便于维护和扩展
- **输入验证**：所有输入都经过验证，防止无效数据导致的错误
- **符合 PEP 8**：遵循 Python 代码风格指南，保持代码一致性

常见问题
--------

- **如何新增微博数据？** 将新的 JSON 文件放入 `weibo/某账号/` 目录，重新运行 `uv run python main.py` 即可自动合并。
- **如何清理索引？** 删除 `vector_index/` 目录后再次运行主程序，系统会自动重建。
- **为何使用 ModelScope 下载模型？** 避免网络限制，统一从 ModelScope 缓存模型文件，再由 Transformers 加载，保证与 HuggingFace 接口兼容。
- **遇到错误怎么办？** 系统会输出详细的错误日志，包括异常堆栈信息。检查日志输出可以帮助定位问题。
- **如何查看调试信息？** 使用 `--show-context` 参数可以查看检索到的原始片段，便于调试检索效果。

技术栈
------

- **Python 3.9+**：核心编程语言
- **PyTorch**：深度学习框架（用于文本嵌入）
- **Transformers**：HuggingFace 模型库
- **FAISS**：向量相似度搜索库
- **rank-bm25**：BM25 关键词检索
- **requests**：HTTP 客户端（用于 API 调用）
- **ModelScope**：模型下载和缓存
- **FastAPI**：现代 Web 框架（用于 REST API 和 Web 界面）
- **Uvicorn**：ASGI 服务器（用于运行 FastAPI 应用）
- **Pydantic**：数据验证和序列化

环境变量配置
------------

以下环境变量可以通过 `.env` 文件或系统环境变量进行配置：

**LLM 配置：**
- `LLM_PROVIDER`：LLM 提供商（`deepseek`、`openai`、`qwen`），默认：`deepseek`
- `LLM_API_KEY`：LLM API 密钥（必需，根据选择的提供商）
  - DeepSeek: 使用 DeepSeek API Key
  - OpenAI: 使用 OpenAI API Key
  - Qwen: 使用 DashScope API Key（阿里云通义千问）
- `LLM_API_URL`：LLM API 端点 URL（可选，使用提供商默认值）
  - DeepSeek 默认: `https://api.deepseek.com/v1/chat/completions`
  - OpenAI 默认: `https://api.openai.com/v1/chat/completions`
  - Qwen 默认: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- `LLM_MODEL_NAME`：LLM 模型名称（可选，使用提供商默认值）
  - DeepSeek 默认: `deepseek-chat`
  - OpenAI 默认: `gpt-3.5-turbo`
  - Qwen 默认: `qwen-turbo`
- `DEEPSEEK_API_KEY`：DeepSeek API 密钥（向后兼容，已废弃，使用 `LLM_API_KEY`）
- `DEEPSEEK_API_URL`：DeepSeek API 端点（向后兼容，已废弃，使用 `LLM_API_URL`）

**数据配置：**
- `WEIBO_DATA_ROOT`：微博数据根目录（默认：`weibo/`）
- `VECTOR_STORE_PATH`：向量索引文件路径（默认：`vector_index/persona.index`）
- `EMBEDDING_MODEL_REPO`：嵌入模型仓库（默认：`BAAI/bge-small-zh-v1.5`）
- `MODEL_SCOPE_CACHE`：模型缓存目录（可选）

**检索配置：**
- `CHUNK_SIZE`：文本块大小（默认：320）
- `CHUNK_OVERLAP`：文本块重叠大小（默认：40）
- `TOP_K`：检索返回数量（默认：10）
- `RERANK_TOP_K`：向量检索候选集规模（默认：18）
- `BM25_TOP_K`：BM25 候选集规模（默认：18）
- `RRF_K`：RRF 融合平滑系数（默认：60）

**其他配置：**
- `FAISS_USE_GPU`：是否使用 GPU（默认：`true`）
- `FAISS_GPU_DEVICE`：GPU 设备编号（默认：`0`）
- `MAX_NEW_TOKENS`：最大生成 token 数（默认：1024）
- `GENERATION_TEMPERATURE`：生成温度参数（默认：0.2）
- `SYSTEM_PROMPT`：系统提示词（可选，用于自定义 LLM 行为）

详细配置说明请参考 `config.py` 中的 `AppConfig` 类。

后续规划
--------

- 增加多轮对话记忆管理功能（✅ 已实现）
- 支持多模态检索（图片、视频描述）与情绪标签
- 引入质量评估模块，监控检索与回答质量
- 支持更多 LLM 提供商（OpenAI、通义千问等）（✅ 已实现）
- 添加 Web 界面（基于FastAPI）（✅ 已实现）
- 实现增量索引更新功能
