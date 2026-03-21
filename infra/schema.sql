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

-- ====================================================================
-- 7. Role-Based Access Control (RBAC) - Least Privilege Principle
-- ====================================================================
-- In a production environment, the application service account (e.g., 'nvd11')
-- should only have DML access (SELECT, INSERT, UPDATE, DELETE) and not DDL.

-- Create the application role if it doesn't exist (Idempotent wrapper)
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'nvd11') THEN
      CREATE ROLE nvd11 LOGIN PASSWORD 'placeholder_password';
   END IF;
END
$do$;

-- Grant connection and schema usage
GRANT CONNECT ON DATABASE rag_evaluation_db TO nvd11;
GRANT USAGE ON SCHEMA public TO nvd11;

-- Grant strict DML permissions on all current tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nvd11;

-- Ensure future tables created by the admin also inherit these exact DML permissions for 'nvd11'
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO nvd11;

-- Grant sequence usage (if any auto-incrementing IDs/UUID generators are used)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nvd11;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO nvd11;
