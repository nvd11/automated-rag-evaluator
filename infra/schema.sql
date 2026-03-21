-- ====================================================================
-- RAG Evaluator & Diagnoser - Core Database Schema (Idempotent)
-- Extension: pgvector
-- ====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Documents (Corpus Metadata)
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    doc_name VARCHAR(255) NOT NULL,
    description TEXT,
    metadata JSONB,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_documents_active ON documents (doc_id) WHERE is_deleted = FALSE;

-- 2. Topics (Domain Categorization)
CREATE TABLE IF NOT EXISTS topics (
    topic_id UUID PRIMARY KEY,
    topic_name VARCHAR(255) NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE UNIQUE INDEX IF NOT EXISTS udx_topics_name ON topics (topic_name) WHERE is_deleted = FALSE;

-- 3. Document to Topics Mapping (Metadata Pre-filtering)
CREATE TABLE IF NOT EXISTS document_topics (
    doc_id VARCHAR(255) REFERENCES documents(doc_id),
    topic_id UUID REFERENCES topics(topic_id),
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (doc_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_doc_topics_reverse ON document_topics (topic_id, doc_id) WHERE is_deleted = FALSE;

-- 4. Document Chunks (The RAG Vector Store)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    doc_id VARCHAR(255) REFERENCES documents(doc_id),
    chunking_strategy VARCHAR(100) NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(768),
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_filter ON document_chunks (doc_id, chunking_strategy) WHERE is_deleted = FALSE;

-- 5. Evaluation Runs (Hyperparameter Sweep Configs)
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id UUID PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    chunking_config VARCHAR(100),
    indexing_config VARCHAR(100),
    reranking_config VARCHAR(100),
    prompting_config VARCHAR(100),
    generation_config VARCHAR(100),
    latency_seconds FLOAT,
    cost_estimate FLOAT,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON evaluation_runs (timestamp DESC) WHERE is_deleted = FALSE;

-- 6. Evaluation Metrics (Diagnoser Input)
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id UUID PRIMARY KEY,
    run_id UUID REFERENCES evaluation_runs(run_id),
    dataset_mode VARCHAR(50) NOT NULL,
    query_id VARCHAR(100) NOT NULL,
    metric_category VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_eval_metrics_diagnoser ON evaluation_metrics (run_id, metric_category, metric_name) WHERE is_deleted = FALSE;
