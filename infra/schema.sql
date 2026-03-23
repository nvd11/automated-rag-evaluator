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
    doc_id VARCHAR(255),
    topic_id UUID,
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
    doc_id VARCHAR(255),
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

-- 5. Inference Runs (RAG Generation Configs)
CREATE TABLE IF NOT EXISTS inference_runs (
    run_id UUID PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    chunking_config VARCHAR(100),
    indexing_config VARCHAR(100),
    reranking_config VARCHAR(100),
    prompting_config VARCHAR(100),
    generation_config VARCHAR(100),
    cost_estimate FLOAT,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_inf_runs_timestamp ON inference_runs (start_time DESC) WHERE is_deleted = FALSE;

-- 5.5 Evaluation Jobs (Evaluator Configs - Many-to-One with Inference Runs)
CREATE TABLE IF NOT EXISTS evaluation_jobs (
    job_id UUID PRIMARY KEY,
    inference_run_id UUID,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    evaluator_model VARCHAR(100) NOT NULL,
    evaluator_prompt_version VARCHAR(100),
    cost_estimate FLOAT,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_eval_jobs_timestamp ON evaluation_jobs (start_time DESC) WHERE is_deleted = FALSE;

-- 6. Evaluation Metrics (Diagnoser Input)
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id UUID PRIMARY KEY,
    job_id UUID,
    inference_run_id UUID,
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

CREATE INDEX IF NOT EXISTS idx_eval_metrics_diagnoser ON evaluation_metrics (job_id, metric_category, metric_name) WHERE is_deleted = FALSE;


-- 7. Query History (RAG Interaction Logs)
CREATE TABLE IF NOT EXISTS query_history (
    query_id UUID PRIMARY KEY,
    queried_by VARCHAR(100) NOT NULL,
    question TEXT NOT NULL,
    retrieved_contexts JSONB,
    generated_answer TEXT,
    query_time TIMESTAMP NOT NULL,
    retrieval_time TIMESTAMP NOT NULL,
    response_time TIMESTAMP NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_query_hist_requester ON query_history (queried_by) WHERE is_deleted = FALSE;

-- 8. Golden Records (Evaluation Benchmark Dataset)
CREATE TABLE IF NOT EXISTS golden_records (
    id UUID PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    expected_topics JSONB,
    complexity VARCHAR(50),
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_golden_records_dataset ON golden_records (dataset_name) WHERE is_deleted = FALSE;

-- 9. Evaluation Query Mappings (Ground Truth to Query Resolution)
CREATE TABLE IF NOT EXISTS evaluation_query_mappings (
    query_id UUID,
    golden_record_id UUID,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (query_id, golden_record_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_query_mappings ON evaluation_query_mappings (golden_record_id, query_id) WHERE is_deleted = FALSE;

-- 10. Run Queries (Inference Run to Query Mapping)
CREATE TABLE IF NOT EXISTS run_queries (
    run_id UUID,
    query_id UUID,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (run_id, query_id)
);

CREATE INDEX IF NOT EXISTS idx_run_queries_reverse ON run_queries (query_id, run_id) WHERE is_deleted = FALSE;

-- 11. RAG Triad Analysis Pivot View
CREATE OR REPLACE VIEW v_evaluation_metrics_pivot AS
SELECT 
    job_id,
    query_id,
    MAX(CASE WHEN metric_name = 'context_relevance' THEN metric_value END) AS context_relevance_score,
    MAX(CASE WHEN metric_name = 'faithfulness' THEN metric_value END) AS faithfulness_score,
    MAX(CASE WHEN metric_name = 'answer_relevance' THEN metric_value END) AS answer_relevance_score,
    MAX(CASE WHEN metric_name = 'recall_at_k' THEN metric_value END) AS recall_at_k_score
FROM evaluation_metrics
WHERE is_deleted = FALSE
GROUP BY job_id, query_id;

-- ====================================================================
-- 12. Role-Based Access Control (RBAC) - Least Privilege Principle
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
