# Phase 3: Golden Dataset Generation Architecture
*Automated Synthetic Ground Truth for RAG Evaluation*

## 1. Overview
A stable enterprise RAG evaluation framework requires a high-quality benchmark dataset (Golden Records) containing pairs of realistic user questions and verifiable "exact" answers (Ground Truth). Instead of relying on manual SME annotation, this phase introduces an **Automated Synthetic Dataset Generator**. 

By randomly sampling factual text chunks directly from the ingested corpus (e.g., the HSBC Annual Report) and prompting an advanced LLM to "reverse-engineer" professional questions and answers, we guarantee that the evaluation benchmark is accurately aligned with the actual domain data and completely free from data contamination.

## 2. Architecture Diagram (Mermaid)

```mermaid
classDiagram
  %% Domain Layer
  class Chunk {
    +String text
    +int page_number
    +int chunk_index
    +int token_count
    +List[float] embedding
  }
  
  class GoldenRecord {
    +UUID id
    +String batch_name
    +String question
    +String ground_truth
    +List[String] expected_topics
    +String complexity
  }

  class QA_Pair {
    <<Pydantic Schema>>
    +String question
    +String answer
    +String complexity
  }

  %% Interfaces
  class IGoldenRecordDAO {
    <<Interface>>
    +get_random_seed_chunks(limit: int, topics: List) List[Chunk]
    +bulk_insert_golden_records(records: List[GoldenRecord])
  }
  
  class IDatasetGenerator {
    <<Interface>>
    +agenerate_qa_from_chunk(chunk: Chunk, batch_name: String) GoldenRecord
  }

  %% Implementations
  class PgVectorGoldenRecordDAO {
    +get_random_seed_chunks(limit: int, topics: List) List[Chunk]
    +bulk_insert_golden_records(records: List[GoldenRecord])
  }

  class LangchainDatasetGenerator {
    -ChatGoogleGenerativeAI llm
    -ChatPromptTemplate prompt
    +agenerate_qa_from_chunk(chunk: Chunk, batch_name: String) GoldenRecord
  }

  %% Orchestrator
  class GoldenDatasetRunner {
    -IGoldenRecordDAO dao
    -IDatasetGenerator generator
    +run(batch_name: String, sample_size: int, topics: List)
  }

  GoldenDatasetRunner o-- IGoldenRecordDAO : Dependency Injection
  GoldenDatasetRunner o-- IDatasetGenerator : Dependency Injection
  
  IGoldenRecordDAO <|.. PgVectorGoldenRecordDAO : Implements
  IDatasetGenerator <|.. LangchainDatasetGenerator : Implements
  
  IDatasetGenerator ..> QA_Pair : Uses for Structured Output
  IDatasetGenerator ..> GoldenRecord : Produces
  IGoldenRecordDAO ..> Chunk : Produces
  IGoldenRecordDAO ..> GoldenRecord : Consumes
```

---

## 2.5 Logic Flow Diagram (Sequence)

This sequence diagram illustrates the parallelized execution flow used to rapidly synthesize the Golden Dataset without hitting API timeouts.

```mermaid
sequenceDiagram
  actor Developer
  participant Runner as GoldenDatasetRunner
  participant DAO as PgVectorGoldenRecordDAO
  participant DB as Cloud SQL (PostgreSQL)
  participant Generator as LangchainDatasetGenerator
  participant LLM as LLM_TEACHER_MODEL (Gemini 3.1 Pro Preview)

  Developer->>Runner: Execute Script (batch_name, sample_size=20)
  activate Runner

  %% Step 1: Fetch Seeds
  Note over Runner, DB: Step 1: Fetch Random Seed Contexts
  Runner->>DAO: get_random_seed_chunks(limit=20)
  activate DAO
  DAO->>DB: SELECT text, metadata FROM document_chunks ORDER BY RANDOM() LIMIT 20
  activate DB
  DB-->>DAO: Return 20 Rows
  deactivate DB
  DAO-->>Runner: List[Chunk] (Length: 20)
  deactivate DAO

  %% Step 2: Concurrent Generation
  Note over Runner, LLM: Step 2: High-Concurrency QA Generation
  Runner->>Generator: asyncio.gather(*[agenerate_qa_from_chunk(c) for c in Chunks])
  activate Generator
  
  loop For Each Chunk (Concurrent)
    Generator->>LLM: ainvoke(Prompt + Chunk Text) with structured_output(QA_Pair)
    activate LLM
    Note over LLM: Evaluates text, formulates question, extracts exact answer
    LLM-->>Generator: Return QA_Pair (JSON)
    deactivate LLM
    Note over Generator: Maps QA_Pair to GoldenRecord DTO
  end
  
  Generator-->>Runner: List[GoldenRecord] (Length: 20)
  deactivate Generator

  %% Step 3: Persistence
  Note over Runner, DB: Step 3: Transactional Persistence
  Runner->>DAO: bulk_insert_golden_records(List[GoldenRecord])
  activate DAO
  DAO->>DB: INSERT INTO golden_records (id, question, ground_truth, ...)
  activate DB
  DB-->>DAO: Success
  deactivate DB
  DAO-->>Runner: Success
  deactivate DAO

  Runner-->>Developer: Print Summary Log
  deactivate Runner
```

---

## 3. Core Components Deep Dive
The generation pipeline adheres to strict SOLID principles, separating database access, LLM orchestration, and workflow execution.

### 3.1 Domain Models (`src/domain/models.py`)
Introduces structured data carriers for the generation pipeline:
- **`GoldenRecord`**: A Pydantic model mapped directly to the `golden_records` database table. Contains `id`, `batch_name`, `question`, `ground_truth`, `expected_topics`, and `complexity`.
- **`QA_Pair`**: A lightweight Pydantic model specifically designed for LLM **Structured Output**. It guarantees the LLM returns exactly a `question`, `answer`, and `complexity` rating without conversational filler.

### 3.2 Data Access Object (DAO) (`src/dao/golden_record_dao.py`)
Handles all PostgreSQL interactions for the benchmark dataset.
- **`get_random_seed_chunks(limit: int) -> List[Chunk]`**: Queries the `document_chunks` table to extract $N$ random, high-quality text fragments to serve as the factual seed for the LLM.
- **`bulk_insert_golden_records(records: List[GoldenRecord]) -> None`**: Executes a batch `INSERT` to persist the newly synthesized QA pairs into the `golden_records` table, making them available for the Evaluation Runner.

### 3.3 The Core Generator (`src/evaluator/dataset_generator.py`)
The "Brain" of the operation, powered by LangChain and Gemini.
- **`LangchainDatasetGenerator`**:
 - Uses `ChatGoogleGenerativeAI` configured with a specific persona (e.g., "Senior Financial Auditor").
 - Evaluates via the `LLM_TEACHER_MODEL` (e.g., `gemini-3.1-pro-preview`), implementing the Asymmetric Compute pattern to ensure the "Teacher" out-reasons the "Student" (RAG Agent).
 - Employs the `with_structured_output(QA_Pair)` feature to force the LLM to return reliable JSON objects.
 - **`agenerate_qa_from_chunk(chunk: Chunk) -> GoldenRecord`**: Takes a single seed chunk, injects it into a meticulously crafted Prompt Template, and awaits the structured QA pair from the LLM.

### 3.4 The Orchestrator Runner (`src/runners/golden_dataset_runner.py`)
The executable script that wires the components together.
- **Workflow**:
 1. Initializes the DB Pool, DAO, and Generator.
 2. Defines the benchmark configuration (e.g., `batch_name="hsbc_2025_eval_v1"`, `sample_size=20`).
 3. Fetches 20 random chunks via the DAO.
 4. Dispatches the chunks to the Generator using `asyncio.gather()` for high-throughput concurrent generation.
 5. Bulk inserts the resulting `GoldenRecord` objects via the DAO.
 6. Outputs a human-readable summary log of the generated dataset.

## 4. Architectural Highlights for the Interview
- **Asymmetric Compute (Teacher vs. Student)**: By isolating `LLM_JUDGE_MODEL` (e.g., `gemini-2.5-pro` for low-latency inference) and `LLM_TEACHER_MODEL` (e.g., `gemini-3.1-pro-preview` for high-reasoning ground truth generation), the system proves its enterprise readiness to balance operational cost against evaluation quality.
- **Prevention of Data Contamination**: The Generator acts strictly as the "Teacher". It only sees the raw text chunks and does *not* interact with the RAG Retrieval pipeline, ensuring the Ground Truth is completely objective.
- **Structured LLM Output**: By using Pydantic schema enforcement during LLM generation, the pipeline is very resilient against parsing errors and unpredictable LLM behaviors.
- **High-Concurrency Execution**: Leveraging Python's `asyncio.gather`, the generation of hundreds of Golden Records can be parallelized, significantly reducing pipeline execution time while remaining within GCP rate limits.
