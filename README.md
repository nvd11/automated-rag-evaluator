# Automated RAG Evaluator and Diagnoser

This repository contains an enterprise-grade automated framework for evaluating and diagnosing Retrieval-Augmented Generation (RAG) systems across multiple pipeline settings.

## Project Structure
- `infra/`: Infrastructure-as-Code (Terraform) for GCP Cloud SQL (pgvector), VPC, and Envoy Proxy.
- `data/`: Raw corpus (PDFs) and synthetic evaluation datasets (CSV/JSONL).
- `config/`: JSON configuration files for hyperparameter sweeps (`optimizer_config.json`).
- `src/ingestion/`: Document parsing, chunking, and embedding pipelines.
- `src/rag/`: Dynamic RAG execution engine.
- `src/evaluation/`: Metric computation (Recall, nDCG, Semantic Similarity, RAG Triad).
- `src/diagnosis/`: Rule-based diagnostic engine for root cause analysis.
- `docs/`: Architecture diagrams and metric catalogs.
