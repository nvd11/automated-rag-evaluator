# Automated RAG Evaluator and Diagnoser

An enterprise-grade framework designed to evaluate, score, and diagnose Retrieval-Augmented Generation (RAG) systems across multiple pipeline settings. This project implements a fully automated evaluation pipeline using LLM-as-a-Judge patterns and a stateless heuristic rule engine for deep diagnostics.

---

## 1. Setup & Installation

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
   Create a `.env` file in the root directory and configure the following required variables:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   DB_HOST="your_db_host"
   DB_PORT=5432
   DB_NAME="rag_evaluator"
   DB_USER="postgres"
   DB_PASSWORD="your_password"
   ENABLE_PROXY=false  # Set to true in restricted network environments to bypass gRPC deadlocks
   ```

---

## 2. Design Documentation

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

## 3. Pipeline Runners (Execution Flow)

The system strictly decouples the entry points (`Runners`) from the core business logic (`Pipelines`). To execute the full lifecycle, run the following scripts in order:

### Phase 1-3: Data Ingestion
```bash
python src/runners/ingestion_runner.py
```
> Extracts data from PDFs, applies configured chunking strategies (e.g., recursive chunking), generates dense vector embeddings, and persists them into the `pgvector` database.

### Phase 4: RAG Inference
```bash
python src/runners/inference_runner.py
```
> Executes queries against the vector store using the RAG Agent. Saves the retrieved contexts and generated answers into `query_history` without evaluating them.

### Phase 5: Automated Evaluation (LLM-as-a-Judge)
```bash
python src/runners/evaluation_runner.py
```
> The **Evaluator**. Dynamically routes queries to polymorphic judges (`GoldenBaselineJudge` for benchmark datasets, `RagTriadJudge` for blind tests). Computes metrics like `Correctness`, `Faithfulness`, and `Context Relevance`, then saves them into the flexible `evaluation_metrics` EAV table.

### Phase 6: Expert Diagnosis
```bash
python src/runners/diagnoser_runner.py
```
> The **Diagnoser**. A pure, stateless analysis layer. Aggregates the Phase 5 metrics and passes them through a Heuristic Rule Engine. If metrics fall below defined thresholds, it outputs actionable optimization recommendations.

---

## 4. Output Files & Artifacts

All final deliverables requested by the assignment are located in the **[`output/`](./output/)** directory:

- 📊 **`case1_benchmark_evaluation.csv`**: Detailed scoring of the 48-query Benchmark Test (Case 1). Includes the original Query, Retrieved Contexts, Generated Answer, Golden Answer (Ground Truth), and the evaluated `Correctness` score.
- 📊 **`case2_blind_test_evaluation.csv`**: Detailed scoring of the 30-query Blind Test (Case 2). Includes the Query, Retrieved Contexts, Generated Answer, and the RAG Triad scores (`Context Relevance`, `Faithfulness`, `Answer Relevance`).
- 📝 **`case1_diagnosis_report.json` & `case2_diagnosis_report.json`**: The final JSON diagnostic reports generated by Phase 6. Structurally adheres to the requested `diagnosis_report_template.json` format, featuring `overall_summary`, `stage_metrics`, and a rule-triggered `diagnosis` array for optimization advice.
- ⚙️ **`optimizer_config.json`**: The hyperparameter search space schema. Defines the configuration format for an experiment orchestrator to run multiple automated sweeps across varying chunking, indexing, reranking, and prompting strategies.
