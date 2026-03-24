# Automated RAG Evaluator & Diagnoser - Assignment Requirements

## 1. Project Goal
Build a framework that evaluates and diagnoses Retrieval-Augmented Generation (RAG) systems across multiple pipeline settings. The framework must support two different dataset modes (Case 1 with ground truth, Case 2 without) and automatically compare various RAG strategies. Finally, it must produce a structured diagnostic analysis report that identifies likely failure points and suggests optimization directions.

## 2. Core Objectives
1. **Define Evaluation Metrics**: Establish metrics for different phases of the RAG pipeline (Retrieval, Reranking, Generation, End-to-End).
2. **Implement Automated Evaluator**: Design a system that executes multiple RAG pipeline configurations (settings) and records the resulting metrics.
3. **Generate Diagnosis Report**: Develop an expert rule engine that analyzes the metrics for a given setting/dataset to identify issues (e.g., hallucination, poor retrieval) and recommend fixes.

## 3. Dataset Modes Supported
- **Case 1 (Benchmark Test)**: Queries with known Golden Answers (Ground Truth). Evaluated using deterministic or LLM-as-a-Judge correctness metrics.
- **Case 2 (Blind Test)**: Queries without standard answers. Evaluated using the RAG Triad proxy metrics (Context Relevance, Faithfulness, Answer Relevance).

## 4. Expected System Scope (Optimizer Search Space)
The framework should be capable of configuring and evaluating variations in:
- **Chunking**: Fixed-size, recursive, sentence-aware, semantic/structure-aware.
- **Indexing / Retrieval**: Dense, sparse (BM25), hybrid, metadata-aware.
- **Reranking**: None, cross-encoder, LLM-based.
- **Query Prompting**: Raw query, query rewriting, step-back, decomposition.
- **Generation**: Model selection, temperature, citation prompting, constraints.

## 5. Required Deliverables & Output Files
The assignment requires the delivery of a working prototype with the following artifacts:

### 5.1 Architecture & Design Documents
- **Architecture Design Document**: Architecture design for the evaluator and diagnoser.
- **Metric Catalog**: Documentation of the per-stage definitions and formulas (e.g., Faithfulness, Context Relevance, Correctness).

### 5.2 Code & Infrastructure
- **Runnable Prototype**: The actual Python implementation of the pipeline (Inference -> Evaluation -> Diagnosis).
- **Database Schema**: A flexible schema capable of storing evaluation metrics across multiple strategies without rigid constraints.

### 5.3 Output Artifacts (JSON & CSV)
The system must generate the following artifacts:
1. **Evaluation Result Exports (CSVs)**:
  - `case1_benchmark_evaluation.csv`: Detailed evaluation records for Case 1 (including Query, Retrieved Contexts, Generated Answer, Golden Answer, and Correctness Score).
  - `case2_blind_test_evaluation.csv`: Detailed evaluation records for Case 2 (including Query, Retrieved Contexts, Generated Answer, Context Relevance, Faithfulness, and Answer Relevance).
2. **Diagnosis Reports (JSON)**:
  - Structured JSON files adhering to the provided `diagnosis_report_template.json` format.
  - Examples: `case1_diagnosis_report.json` and `case2_diagnosis_report.json`.
  - Must include `overall_summary`, `stage_metrics`, and a rule-triggered `diagnosis` array identifying issues and `recommended_actions`.
3. **Optimizer Configuration (JSON)**:
  - `optimizer_config.json`: An experimental configuration file defining the search space and evaluation thresholds.

## 6. Key Evaluation Criteria & Technical Constraints
- **Design Patterns**: Clear separation of concerns (e.g., Runners vs. Pipelines, DAOs vs. Domain Logic).
- **Database Anti-Patterns**: Avoid Hard FKs and `ON DELETE CASCADE`. Use Soft Links and High-Extensibility Long Tables (Upgraded EAV) for metric storage.
- **Polymorphism & OCP**: Use unified interfaces (e.g., `evaluate_query`) instead of hardcoded IF-ELSE case logic.
- **Stateless Diagnosis**: The Diagnoser should act as a pure read-only analysis layer via an extensible Rule Engine.
- **Network Resilience**: Must gracefully handle proxy environments (e.g., bypassing blocked gRPC protocols using REST fallback).
- **Professionalism**: Architecture documents must use objective, professional terminology (e.g., "Architecture Document").
