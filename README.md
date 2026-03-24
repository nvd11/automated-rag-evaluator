# Automated RAG Evaluator and Diagnoser

An enterprise-grade framework designed to evaluate, score, and diagnose Retrieval-Augmented Generation (RAG) systems across multiple pipeline settings. This project implements a fully automated evaluation pipeline using LLM-as-a-Judge patterns and a stateless heuristic rule engine for deep diagnostics.

---

## 1. Project Directory Structure

```text
automated-rag-evaluator/
├── data/           # Reference corpus (HSBC 2025 Annual Report PDF) and auto-generated benchmark datasets.
├── docs/           # Comprehensive architecture and schema design documentation (8 detailed documents).
├── infra/          # Infrastructure schema files (SQL DDL) for the pgvector database.
├── output/         # Required assignment deliverables (JSON Diagnosis Reports, CSV Evaluation scores, Optimizer Config).
├── src/            # Core Python source code (structured by Domain Driven Design).
│   ├── agents/     # High-level RAG Agent executing inference logic.
│   ├── configs/    # Environment configurations, Database pooling, and thread-safe Logging settings.
│   ├── dao/        # Data Access Objects (DAOs) interfacing with PostgreSQL for persistence and isolation.
│   ├── diagnosis/  # The Stateless Heuristic Rule Engine identifying root causes based on metrics.
│   ├── domain/     # Core entities and Data Transfer Objects (DTOs) heavily utilizing Pydantic.
│   ├── evaluator/  # LLM-as-a-Judge logic (GoldenBaselineJudge and RagTriadJudge).
│   ├── ingestion/  # Loaders, Chunkers, and Embedders for populating the vector database.
│   ├── interfaces/ # Abstract Base Classes (ABCs) enforcing Polymorphism and the Open/Closed Principle.
│   ├── llm/        # Factory pattern implementing Google Gemini generation with REST fallback proxies.
│   ├── pipelines/  # Orchestrators chaining domain logic together (EvaluationPipeline, DiagnoserPipeline).
│   ├── retrieval/  # Base semantic retriever logic executing vector similarity searches.
│   └── runners/    # CLI entry points (Ignition scripts) to execute specific Pipelines.
├── test/           # Comprehensive Pytest suite covering DAO integration and Pipeline mocks.
├── .env.example    # Environment variable template with pre-configured live Cloud SQL settings.
├── requirements.txt# Project dependencies (Langchain, Google-Genai, Psycopg, Pydantic, Pytest).
└── REQUIREMENTS.md # A structured breakdown of the assignment objectives and expected system scope.
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
   *(Core dependencies include: `langchain`, `google-genai`, `psycopg[binary]`, `pydantic`, `loguru`, `pytest`)*

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
   | `ENABLE_PROXY` | **Critical for Mainland China users**: Set this to `true` to force a REST fallback and bypass `gRPC` deadlocks caused by proxy environments. |

   💡 **Pre-configured Cloud SQL Instance**: 
   To save you the hassle of spinning up your own infrastructure, the provided `.env.example` file already contains connection details to a **live Google Cloud SQL (PostgreSQL + pgvector)** instance (`db.jpgcp.cloud`) hosted explicitly for this assignment. You do not need to install a local database; simply provide your `GEMINI_API_KEY` and the system will run out-of-the-box.

   ⚠️ **Important Note for Mainland China / Proxy Environments**: 
   LangChain's async structured outputs default to the `gRPC` protocol, which often deadlocks over HTTP/1.1 proxies. If you are operating behind a proxy, you **must** set `ENABLE_PROXY=true` in your `.env`. This tells our LLM Factory to dynamically wrap synchronous REST requests in `asyncio.to_thread()`, bypassing the blocked `gRPC` channel.

---

## 3. Technology Stack & Model Roles

### LLM Assignments
The framework dynamically injects specific LLM models based on the stage of the pipeline to optimize for both cost and reasoning capabilities:
- **Embedding & Indexing**: `text-embedding-004` (Google Gemini Embedder via Vertex/AI Studio) is used for dense vector indexing.
- **RAG Generation**: `gemini-2.5-pro` (Temperature = 0) is tasked with synthesizing answers from retrieved contexts.
- **LLM-as-a-Judge (Evaluator)**: `gemini-2.5-pro` (with `Structured Outputs` enforced) acts as the strict, deterministic grader for both the Golden Baseline matches (Case 1) and the RAG Triad heuristics (Case 2).
- **Golden Dataset Synthesis**: `gemini-3.1-pro-preview` powers the inverse-generation pipeline to extract facts from the financial reports and formulate Question/Answer benchmark pairs, leveraging Google's most advanced preview reasoning model.

### Key Libraries
- **`langchain` & `langchain-google-genai`**: Utilized for pipeline orchestration, prompt templating, and text splitting (e.g., `RecursiveCharacterTextSplitter`).
- **`psycopg` (v3)**: The modern, fully async PostgreSQL driver used for high-performance `pgvector` transactions and connection pooling.
- **`pydantic` (v2)**: Enforces strict schema validation for internal Data Transfer Objects (DTOs) and guarantees deterministic JSON outputs from the LLM Evaluators.
- **`loguru`**: Powers the thread-safe, asynchronous logging throughout the pipeline.
- **`pytest` & `pytest-asyncio`**: Drives the test suite, enforcing robust integration testing with dynamic database teardowns.

---

## 4. Dataset & Reference Corpus

As per the assignment requirements ("No actual data is provided... please download some annual reports / financial statements"), this framework is built and tested against real-world financial data. 

- **Reference Corpus**: The system ingests the official **HSBC Annual Report 2025** (Mock/Early Release).
- **Location**: The raw PDF is located at **[`data/HSBC_Annual_Report_2025.pdf`](./data/HSBC_Annual_Report_2025.pdf)**.
- **Evaluation Set**: The benchmark Ground Truth QA pairs and the blind test queries were auto-generated based on this specific document and can be found under the `data/` directory.

---

## 5. Design Documentation

This project enforces strict software engineering principles (SOLID, Dependency Injection, Polymorphism). The architectural rationale is comprehensively documented across multiple design documents, detailing the evolution of each phase:

- 📄 **[`REQUIREMENTS.md`](./REQUIREMENTS.md)**: A structured breakdown of the assignment objectives, evaluation modes (Case 1 vs Case 2), search spaces, and strict constraints.
- 📄 **[`docs/database_schema_design.md`](./docs/database_schema_design.md)**: Details the highly extensible Upgraded EAV (Entity-Attribute-Value) schema that avoids hard FKs and `ON DELETE CASCADE` anti-patterns.
- 📄 **[`docs/phase1_ingestion_architecture.md`](./docs/phase1_ingestion_architecture.md)** & **[`docs/data_ingestion_pipeline_design.md`](./docs/data_ingestion_pipeline_design.md)**: Explains the extraction, chunking, and idempotent transactional persistence logic into `pgvector`.
- 📄 **[`docs/phase2_retriever_architecture.md`](./docs/phase2_retriever_architecture.md)**: Details the design of the RAG semantic search components.
- 📄 **[`docs/phase3_golden_dataset_generation.md`](./docs/phase3_golden_dataset_generation.md)**: Outlines the automated LLM pipeline used to synthesize the benchmark QA dataset based on financial reports.
- 📄 **[`docs/phase4_inference_runner_architecture.md`](./docs/phase4_inference_runner_architecture.md)**: Explains the decoupled inference layer executing queries across configurations without immediate evaluation.
- 📄 **[`docs/phase5_evaluation_runner_architecture.md`](./docs/phase5_evaluation_runner_architecture.md)**: Details the polymorphic LLM-as-a-Judge evaluators (`GoldenBaselineJudge` vs `RagTriadJudge`) routing dynamically by dataset type.
- 📄 **[`docs/phase6_diagnoser_architecture.md`](./docs/phase6_diagnoser_architecture.md)**: Contains the UML class diagram and explains the pure, stateless Diagnoser Rule Engine for generating actionable reports.

---

## 6. Pipeline Runners (Execution Flow)

The system strictly decouples the entry points (`Runners`) from the core business logic (`Pipelines`). To execute the full lifecycle, run the following scripts in order:

### Phase 1: Data Ingestion
```bash
python src/runners/ingestion_runner.py
```
> Extracts data from PDFs, applies configured chunking strategies (e.g., recursive chunking), generates dense vector embeddings, and persists them into the `pgvector` database.

### Phase 2: RAG Inference
```bash
python src/runners/inference_runner.py
```
> Executes queries against the vector store using the RAG Agent. Saves the retrieved contexts and generated answers into `query_history` without evaluating them.

### Phase 3: Automated Evaluation (LLM-as-a-Judge)
```bash
python src/runners/evaluation_runner.py
```
> The **Evaluator**. Dynamically routes queries to polymorphic judges (`GoldenBaselineJudge` for benchmark datasets, `RagTriadJudge` for blind tests). Computes metrics like `Correctness`, `Faithfulness`, and `Context Relevance`, then saves them into the flexible `evaluation_metrics` EAV table.

### Phase 4: Expert Diagnosis
```bash
python src/runners/diagnoser_runner.py
```
> The **Diagnoser**. A pure, stateless analysis layer. Aggregates the Phase 3 metrics and passes them through a Heuristic Rule Engine. If metrics fall below defined thresholds, it outputs actionable optimization recommendations.

---

## 7. Output Files & Artifacts

All final deliverables requested by the assignment are located in the **[`output/`](./output/)** directory:

- 📊 **`case1_benchmark_evaluation.csv`**: Detailed scoring of the 48-query Benchmark Test (Case 1). Includes the original Query, Retrieved Contexts, Generated Answer, Golden Answer (Ground Truth), and the evaluated `Correctness` score.
- 📊 **`case2_blind_test_evaluation.csv`**: Detailed scoring of the 30-query Blind Test (Case 2). Includes the Query, Retrieved Contexts, Generated Answer, and the RAG Triad scores (`Context Relevance`, `Faithfulness`, `Answer Relevance`).
- 📝 **`case1_diagnosis_report.json` & `case2_diagnosis_report.json`**: The final JSON diagnostic reports generated by Phase 4. Structurally adheres to the requested `diagnosis_report_template.json` format, featuring `overall_summary`, `stage_metrics`, and a rule-triggered `diagnosis` array for optimization advice.
- ⚙️ **`optimizer_config.json`**: The hyperparameter search space schema. Defines the configuration format for an experiment orchestrator to run multiple automated sweeps across varying chunking, indexing, reranking, and prompting strategies.
