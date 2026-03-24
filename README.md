# Automated RAG Evaluator and Diagnoser

A framework designed to evaluate, score, and diagnose Retrieval-Augmented Generation (RAG) systems across multiple pipeline settings. This project implements an automated evaluation pipeline using LLM-as-a-Judge patterns and a stateless heuristic rule engine for diagnostics.

---

## 1. Project Directory Structure

```text
automated-rag-evaluator/
├── data/           # Reference corpus (HSBC 2025 Annual Report PDF) and auto-generated benchmark datasets.
├── docs/           # Architecture and schema design documentation.
├── infra/          # Infrastructure schema files (SQL DDL) for the pgvector database.
├── output/         # Assignment deliverables (JSON Diagnosis Reports, CSV Evaluation scores, Optimizer Config).
├── src/            # Core Python source code.
│   ├── agents/     # RAG Agent executing inference logic.
│   ├── configs/    # Environment configurations, Database pooling, and Logging settings.
│   ├── dao/        # Data Access Objects (DAOs) interfacing with PostgreSQL.
│   ├── diagnosis/  # Stateless Heuristic Rule Engine for diagnostics.
│   ├── domain/     # Core entities and Data Transfer Objects (DTOs).
│   ├── evaluator/  # LLM-as-a-Judge logic.
│   ├── ingestion/  # Loaders, Chunkers, and Embedders for populating the vector database.
│   ├── interfaces/ # Abstract Base Classes (ABCs).
│   ├── llm/        # Factory pattern implementing Google Gemini generation.
│   ├── pipelines/  # Orchestrators chaining domain logic together.
│   ├── retrieval/  # Base semantic retriever logic executing vector similarity searches.
│   └── runners/    # CLI entry points to execute specific Pipelines.
├── test/           # Pytest suite covering DAO integration and Pipeline mocks.
├── .env.example    # Environment variable template with pre-configured Cloud SQL settings.
├── requirements.txt# Project dependencies.
└── REQUIREMENTS.md # Breakdown of the assignment objectives and expected system scope.
```

---

## 2. Setup & Installation

### Prerequisites
- Python 3.11+

### Environment Setup
1. **Unzip the package and navigate to the project root:**
   ```bash
   cd automated-rag-evaluator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependencies include: `langchain`, `google-genai`, `psycopg[binary]`, `pydantic`, `loguru`, `pytest`)*

4. **Configure Environment Variables:**
   Copy the example environment file and customize it:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and configure the required settings:
   
   | Variable | Description |
   |----------|-------------|
   | `GEMINI_API_KEY` | Your Google Gemini API Key for generation and evaluation. |
   | `DB_HOST` | Hostname of the PostgreSQL database. |
   | `DB_PORT` | Port of the PostgreSQL database (default `5432`). |
   | `DB_NAME` | Database name (e.g., `rag_evaluator`). |
   | `DB_USER` | Database username. |
   | `DB_PASSWORD` | Database password. |
   | `HTTP_PROXY` / `HTTPS_PROXY` | Proxy URL if your network requires one. |
   | `ENABLE_PROXY` | Set this to `true` to force a REST fallback and bypass `gRPC` issues caused by proxy environments. |

   **Pre-configured Cloud SQL Instance**: 
   The provided `.env.example` file contains connection details to a Google Cloud SQL (PostgreSQL + pgvector) instance (`db.jpgcp.cloud`) hosted for this assignment. You do not need to install a local database; provide your `GEMINI_API_KEY` to run the system.

   **Note on Proxy Environments**: 
   LangChain's async structured outputs default to the `gRPC` protocol, which can cause connection timeouts over HTTP/1.1 proxies. If you are operating behind a proxy, set `ENABLE_PROXY=true` in your `.env`. This configures the LLM Factory to dynamically wrap synchronous REST requests in `asyncio.to_thread()`, bypassing the `gRPC` channel.

---

## 3. Technology Stack & Model Roles

### LLM Assignments
The framework injects specific LLM models based on the stage of the pipeline:
- **Embedding & Indexing**: `text-embedding-004` (Google Gemini Embedder) is used for dense vector indexing.
- **RAG Generation**: `gemini-2.5-pro` (Temperature = 0) is tasked with synthesizing answers from retrieved contexts.
- **LLM-as-a-Judge (Evaluator)**: `gemini-2.5-pro` (with `Structured Outputs` enforced) acts as the grader for both the Golden Baseline matches (Case 1) and the RAG Triad heuristics (Case 2).
- **Golden Dataset Synthesis**: `gemini-3.1-pro-preview` powers the inverse-generation pipeline to extract facts from the financial reports and formulate Question/Answer benchmark pairs.

### Key Libraries
- **`langchain` & `langchain-google-genai`**: Utilized for pipeline orchestration, prompt templating, and text splitting.
- **`psycopg` (v3)**: Asynchronous PostgreSQL driver used for `pgvector` transactions and connection pooling.
- **`pydantic` (v2)**: Enforces schema validation for internal Data Transfer Objects (DTOs) and structures JSON outputs from the LLM Evaluators.
- **`loguru`**: Asynchronous logging.
- **`pytest` & `pytest-asyncio`**: Test suite and integration testing.

---

## 4. Dataset & Reference Corpus

As per the assignment requirements, this framework is tested against financial data. 

- **Reference Corpus**: The system ingests the **HSBC Annual Report 2025**.
- **Location**: The raw PDF is located at **`data/HSBC_Annual_Report_2025.pdf`**.
- **Evaluation Set**: The benchmark Ground Truth QA pairs and the blind test queries were auto-generated based on this document and can be found under the `data/` directory.

---

## 5. Design Documentation

The architectural design is documented across multiple files, detailing the evolution of each phase:

- **`REQUIREMENTS.md`**: A breakdown of the assignment objectives, evaluation modes (Case 1 vs Case 2), search spaces, and constraints.
- **`docs/database_schema_design.md`**: Details the Upgraded EAV (Entity-Attribute-Value) schema that avoids hard FKs.
- **`docs/phase1_ingestion_architecture.md`** & **`docs/data_ingestion_pipeline_design.md`**: Explains the extraction, chunking, and idempotent transactional persistence logic into `pgvector`.
- **`docs/phase2_retriever_architecture.md`**: Details the design of the RAG semantic search components.
- **`docs/phase3_golden_dataset_generation.md`**: Outlines the LLM pipeline used to synthesize the benchmark QA dataset based on financial reports.
- **`docs/phase4_inference_runner_architecture.md`**: Explains the decoupled inference layer executing queries across configurations without immediate evaluation.
- **`docs/phase5_evaluation_runner_architecture.md`**: Details the polymorphic LLM-as-a-Judge evaluators (`GoldenBaselineJudge` vs `RagTriadJudge`).
- **`docs/phase6_diagnoser_architecture.md`**: Contains the UML class diagram and explains the stateless Diagnoser Rule Engine for generating reports.

---

## 6. Pipeline Runners (Execution Flow)

The system decouples the entry points (`Runners`) from the business logic (`Pipelines`). To execute the full lifecycle, run the following scripts in order:

### Phase 1: Data Ingestion
```bash
python src/runners/ingestion_runner.py
```
> Extracts data from PDFs, applies configured chunking strategies, generates dense vector embeddings, and persists them into the database.

### Phase 2: RAG Inference
```bash
python src/runners/inference_runner.py
```
> Executes queries against the vector store using the RAG Agent. Saves the retrieved contexts and generated answers into `query_history`.

### Phase 3: Automated Evaluation (LLM-as-a-Judge)
```bash
python src/runners/evaluation_runner.py
```
> The Evaluator. Routes queries to judges (`GoldenBaselineJudge` for benchmark datasets, `RagTriadJudge` for blind tests). Computes metrics like Correctness, Faithfulness, and Context Relevance, then saves them into the `evaluation_metrics` table.

### Phase 4: Expert Diagnosis
```bash
python src/runners/diagnoser_runner.py
```
> The Diagnoser. A stateless analysis layer. Aggregates the Phase 3 metrics and passes them through a Heuristic Rule Engine. Outputs optimization recommendations based on configured thresholds.

---

## 7. Output Files & Artifacts

All final deliverables requested by the assignment are located in the **`output/`** directory:

- **`case1_benchmark_evaluation.csv`**: Detailed scoring of the 48-query Benchmark Test (Case 1). Includes the original Query, Retrieved Contexts, Generated Answer, Golden Answer (Ground Truth), and the evaluated Correctness score.
- **`case2_blind_test_evaluation.csv`**: Detailed scoring of the 30-query Blind Test (Case 2). Includes the Query, Retrieved Contexts, Generated Answer, and the RAG Triad scores (Context Relevance, Faithfulness, Answer Relevance).
- **`case1_diagnosis_report.json` & `case2_diagnosis_report.json`**: The final JSON diagnostic reports generated by Phase 4. Structurally adheres to the requested `diagnosis_report_template.json` format, featuring `overall_summary`, `stage_metrics`, and a rule-triggered `diagnosis` array.
- **`optimizer_config.json`**: The hyperparameter search space schema. Defines the configuration format for an experiment orchestrator to run automated sweeps across varying chunking, indexing, reranking, and prompting strategies.
