WeiboRAG —— 微博人物多账号RAG问答系统
=======================================

项目简介
--------

WeiboRAG 用于汇总同一人物多个微博账号的历史内容，构建基于检索增强生成（RAG）的问答能力。系统会遍历 `weibo/` 目录下所有子文件夹中的 JSON 数据，自动抽取微博文本、元数据和多媒体描述，构建向量索引，并通过大语言模型提供问答服务。

目录结构
--------

```
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

1. **安装依赖**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   - `DEEPSEEK_API_KEY`：DeepSeek API 密钥（若未设置，将默认使用 `config.py` 中的占位密钥）。
   - `MODEL_SCOPE_CACHE`（可选）：ModelScope 模型缓存目录。

3. **运行主程序**
   ```bash
   python main.py --query "你的下一场演唱会在什么时候？" --top-k 6 --show-context
   ```

   常用参数说明（均可组合使用）：

   - `--query`：提出需要回答的问题。例如：
     ```bash
     python main.py --query "请为黄霄雲的巡演总结亮点"
     ```
   - `--rebuild-index`：强制重新构建向量索引，适用于新增/修改微博数据后。例如：
     ```bash
     python main.py --rebuild-index
     ```
   - `--top-k`：设置返回的上下文数量（覆盖配置文件中的默认值）。例如：
     ```bash
     python main.py --query "粉丝互动情况怎样？" --top-k 8
     ```
   - `--show-context`：在控制台打印检索到的原始片段，便于调试。例如：
     ```bash
     python main.py --query "最近发布的公益活动有哪些？" --show-context
     ```
   - **交互式模式**：若不提供 `--query`，程序会进入命令行问答循环，可多轮提问，输入 `exit` 退出：
     ```bash
     python main.py
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
- `deepseek_api_key`：DeepSeek 调用密钥。
- `rerank_top_k`：向量检索候选集规模，用于后续融合。
- `bm25_top_k`：BM25 候选集规模，缺省与 `rerank_top_k` 一致。
- `rrf_k`：RRF 融合的平滑系数，越大代表对长尾排名更“保守”。
- `faiss_use_gpu`：是否优先启用 GPU 版 FAISS（默认自动开启，若无 GPU 会自动回退 CPU）。
- `faiss_gpu_device`：GPU 编号，默认为 `0`。

如需修改，直接在 `config.py` 中调整或通过环境变量覆盖。

模块概览
--------

- `rag_modules.data_preparation`
  - 递归扫描 `data_root`，解析 JSON 中的 `user` 与 `weibo` 字段。
  - 构建标准化 `WeiboDocument` 数据结构，包含文本、时间、互动信息及来源账号。
  - 对内容进行清洗（去重、去除空文本、替换链接占位符等），并使用可配置的文本切分策略生成段落。

- `rag_modules.index_construction`
  - 使用 ModelScope 下载并缓存 BGE-small-zh-v1.5 模型。
  - 通过 HuggingFace Transformers 加载同一模型权重，生成文本嵌入。
  - 构建或增量更新 FAISS 向量索引，持久化至 `vector_store_path`。

- `rag_modules.retrieval_optimization`
  - 结合向量检索与 BM25 关键词检索，利用 RRF（Reciprocal Rank Fusion）融合候选。
  - `Retriever` 同时返回各通道得分与排名，便于后续调试和监控。

- `rag_modules.generation_integration`
  - 封装 DeepSeek API 的调用逻辑，支持上下文提示模板和安全重试。
  - 接收检索结果，生成最终回答与支持证据。

运行流程
--------

1. **数据准备**：`DataPreparationPipeline` 从 JSON 文件生成标准文本块。
2. **索引构建**：`IndexBuilder` 根据文本块创建/更新 FAISS 向量索引，若检测到 GPU 且安装了 `faiss-gpu`，会自动将索引迁移到 GPU 提速检索。
3. **检索优化**：`Retriever` 同时运行向量检索与 BM25 检索，并通过 RRF 融合选出最相关片段。
4. **生成集成**：`DeepSeekGenerator` 组合上下文，通过 DeepSeek API 生成回答。

常见问题
--------

- **如何新增微博数据？** 将新的 JSON 文件放入 `weibo/某账号/` 目录，重新运行 `main.py` 即可自动合并。
- **如何清理索引？** 删除 `vector_index/` 目录后再次运行主程序，系统会自动重建。
- **为何使用 ModelScope 下载模型？** 避免网络限制，统一从 ModelScope 缓存模型文件，再由 Transformers 加载，保证与 HuggingFace 接口兼容。

后续规划
--------

- 增加多轮对话记忆管理功能。
- 支持多模态检索（图片、视频描述）与情绪标签。
- 引入质量评估模块，监控检索与回答质量。


