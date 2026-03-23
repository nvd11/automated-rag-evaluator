# Phase 3: Golden Dataset Generation Architecture
*Automated Synthetic Ground Truth for RAG Evaluation*

## 1. Overview
A robust enterprise RAG evaluation framework requires a high-quality benchmark dataset (Golden Records) containing pairs of realistic user questions and verifiable "perfect" answers (Ground Truth). Instead of relying on manual SME annotation, this phase introduces an **Automated Synthetic Dataset Generator**. 

By randomly sampling factual text chunks directly from the ingested corpus (e.g., the HSBC Annual Report) and prompting an advanced LLM to "reverse-engineer" professional questions and answers, we guarantee that the evaluation benchmark is perfectly aligned with the actual domain data and completely free from data contamination.

## 2. Core Architecture & Component Responsibilities
The generation pipeline adheres to strict SOLID principles, separating database access, LLM orchestration, and workflow execution.

### 2.1 Domain Models (`src/domain/models.py`)
Introduces structured data carriers for the generation pipeline:
- **`GoldenRecord`**: A Pydantic model mapped directly to the `golden_records` database table. Contains `id`, `dataset_name`, `question`, `ground_truth`, `expected_topics`, and `complexity`.
- **`QA_Pair`**: A lightweight Pydantic model specifically designed for LLM **Structured Output**. It guarantees the LLM returns exactly a `question`, `answer`, and `complexity` rating without conversational filler.

### 2.2 Data Access Object (DAO) (`src/dao/golden_record_dao.py`)
Handles all PostgreSQL interactions for the benchmark dataset.
- **`get_random_seed_chunks(limit: int) -> List[Chunk]`**: Queries the `document_chunks` table to extract $N$ random, high-quality text fragments to serve as the factual seed for the LLM.
- **`bulk_insert_golden_records(records: List[GoldenRecord]) -> None`**: Executes a batch `INSERT` to persist the newly synthesized QA pairs into the `golden_records` table, making them available for the Evaluation Runner.

### 2.3 The Core Generator (`src/evaluator/dataset_generator.py`)
The "Brain" of the operation, powered by LangChain and Gemini.
- **`LangchainDatasetGenerator`**:
  - Uses `ChatGoogleGenerativeAI` configured with a specific persona (e.g., "Senior Financial Auditor").
  - Employs the `with_structured_output(QA_Pair)` feature to force the LLM to return reliable JSON objects.
  - **`agenerate_qa_from_chunk(chunk: Chunk) -> GoldenRecord`**: Takes a single seed chunk, injects it into a meticulously crafted Prompt Template, and awaits the structured QA pair from the LLM.

### 2.4 The Orchestrator Runner (`src/runners/golden_dataset_runner.py`)
The executable script that wires the components together.
- **Workflow**:
  1. Initializes the DB Pool, DAO, and Generator.
  2. Defines the benchmark configuration (e.g., `dataset_name="hsbc_2025_eval_v1"`, `sample_size=20`).
  3. Fetches 20 random chunks via the DAO.
  4. Dispatches the chunks to the Generator using `asyncio.gather()` for high-throughput concurrent generation.
  5. Bulk inserts the resulting `GoldenRecord` objects via the DAO.
  6. Outputs a human-readable summary log of the generated dataset.

## 3. Architectural Highlights for the Interview
- **Prevention of Data Contamination**: The Generator acts strictly as the "Teacher". It only sees the raw text chunks and does *not* interact with the RAG Retrieval pipeline, ensuring the Ground Truth is completely objective.
- **Structured LLM Output**: By using Pydantic schema enforcement during LLM generation, the pipeline is highly resilient against parsing errors and unpredictable LLM behaviors.
- **High-Concurrency Execution**: Leveraging Python's `asyncio.gather`, the generation of hundreds of Golden Records can be parallelized, significantly reducing pipeline execution time while remaining within GCP rate limits.
