# ---------------------------------------------------------
# 1. Existing Enterprise Cloud SQL Instance (Data Source)
# ---------------------------------------------------------
# We reference the existing corporate PostgreSQL instance (with pgvector enabled).
data "google_sql_database_instance" "existing_pgvector_instance" {
  name = var.db_instance_name
}

# ---------------------------------------------------------
# 2. Logical Database & Admin User
# ---------------------------------------------------------
# Provisions a dedicated database and user strictly for the RAG Evaluator on the shared cluster.
resource "google_sql_database" "rag_db" {
  name     = "rag_evaluation_db"
  instance = data.google_sql_database_instance.existing_pgvector_instance.name
}

resource "google_sql_user" "rag_user" {
  name     = "rag_admin"
  instance = data.google_sql_database_instance.existing_pgvector_instance.name
  password = var.db_password
}

# ---------------------------------------------------------
# 3. Database Schema Initialization
# ---------------------------------------------------------
# DDL (Tables, Indexes, Keys) is defined in `schema.sql`.
