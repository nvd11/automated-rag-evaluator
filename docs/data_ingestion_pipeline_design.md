# 1.2 Data Ingestion Pipeline Design (`01_data_ingestion.py`)

## 1. 概述 (Overview)
本项目要求使用真实的语料库进行 RAG 评估。本模块负责读取本地的《汇丰 2025 年年度财报》(PDF格式)，对其进行文本提取、分块 (Chunking)、调用 Google Vertex AI 接口生成向量 (Embeddings)，并将所有元数据及向量持久化至 GCP Cloud SQL (pgvector) 中。

为了满足面试要求中对“企业级架构”与“Metadata Pre-filtering（混合检索）”的考察，整个流水线将严格保证数据的幂等性、事务完整性，并正确填充所有的 5 个合规审计字段。

---

## 2. 核心架构选型 (Tech Stack)
*   **PDF 解析**: `PyMuPDF` (即 `fitz`)。速度极快，且能精准提取页码等元数据。
*   **文本分块 (Chunking)**: `langchain-text-splitters` (`RecursiveCharacterTextSplitter`)。支持重叠块（Overlap），保证上下文语义不断层。
*   **向量化 (Embedding)**: `google-cloud-aiplatform` (Vertex AI `TextEmbeddingModel`)，采用企业级模型 `text-embedding-004` (维度: 768)。
*   **数据库交互**: `psycopg2-binary` 结合 `pgvector` 扩展。使用 `execute_values` 进行批量高效插入 (Bulk Insert)。

---

## 3. 程序执行流程 (Pipeline Flow)

### Step 1: 环境与配置初始化 (Initialization)
1. 从 `.env` 文件读取环境变量（`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME`, `GCP_PROJECT`, `GCP_REGION`）。
2. 初始化 Vertex AI Embeddings 客户端（使用机器绑定的 Service Account 或环境变量）。
3. 建立 PostgreSQL 数据库连接池，并开启事务控制 (`conn.autocommit = False`)。

### Step 2: 语料解析与元数据抽取 (Parsing & Metadata Extraction)
1. 读取 `data/HSBC_Annual_Report_2025.pdf`。
2. 提取全局元数据（文件名、总页数、MD5 哈希值，用于幂等性校验）。
3. **按页抽取文本**，保留每一页的文本以及对应的页码 (Page Number)，组装成结构化数据 `List[Dict]`，供下一步切片使用。
4. **人工规则提炼 Topic**：作为 Annual Report，我们静态定义或利用 LLM 提取 4 个核心 Topics（例如："Financial Performance", "Risk Management", "Sustainability", "Corporate Governance"），以便后续演示 Metadata Pre-filtering 机制。

### Step 3: 语义切块 (Chunking)
1. 采用 `RecursiveCharacterTextSplitter`。
    *   **Chunk Size**: 1000 字符 (Characters)。
    *   **Chunk Overlap**: 200 字符。
2. 遍历每一页的文本，进行切块。每个产生的 Chunk 必须继承它所在的页码 (Page Number) 作为元数据。

### Step 4: 批量向量化 (Batch Embedding)
1. 将上一步生成的所有 Chunks 文本收集成一个大列表。
2. 分批次（Batch Size = 100，避免触发 API Rate Limit）调用 Vertex AI `get_embeddings`。
3. 将返回的 `List[List[float]]` 与原 Chunk 结构拼装在一起。

### Step 5: 数据库事务化入库 (Transactional Upsert)
此步骤在单一事务 (Single Transaction) 中完成，保证全有或全无 (Atomicity)：
1. **Upsert `documents` 表**：
    *   根据 MD5 或文件路径检查是否存在。若存在则更新，若不存在则插入，并获取 `document_id`。
    *   注入审计字段：`created_by = 'ingestion_pipeline'`, `is_deleted = false`。
2. **Upsert `topics` 表**：
    *   插入事先定义好的 Topics，并获取 `topic_id` 列表。
3. **Upsert `document_topics_map` 表**：
    *   建立当前 `document_id` 与所有 `topic_id` 的关联记录。
4. **清理旧数据 (Idempotency)**：
    *   执行 `DELETE FROM document_chunks WHERE document_id = [current_id]`，确保重复运行该脚本不会导致向量数据翻倍（体现幂等性）。
5. **批量插入 `document_chunks` 表**：
    *   使用 `psycopg2.extras.execute_values` 高效插入所有的 Chunk Text, Embedding Vector (强转为 PostgreSQL 数组形式 `[0.1, 0.2, ...]`), Page Number, Token Count。
    *   注入审计字段。
6. **提交事务 (Commit)**。若中途出错，执行 `Rollback`。

---

## 4. 关键亮点设计 (Design Highlights to Impress the Interviewer)
1.  **真正的幂等性 (True Idempotency)**：通过 MD5 哈希和预清理机制，同一份 PDF 跑 100 次数据库里的数据也只有 1 份，不会产生脏数据。
2.  **批处理与速率限制 (Batching & Rate Limiting)**：考虑了调用 GCP大模型 API 的流控，体现了工程师处理生产级大批量数据的真实经验。
3.  **细粒度元数据保留 (Granular Metadata)**：向量不仅关联到文档，还深入到**页码**和**多对多标签 (Topics)**。这在后续 `RAG Evaluator` 检索时，可以用来做高度复杂的条件组合过滤。
4.  **严格的审计留存 (Audit Trails)**：所有的入库操作均强制写入 `created_at`, `created_by`, `updated_at`, `updated_by`。

