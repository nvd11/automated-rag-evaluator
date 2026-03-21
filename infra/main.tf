# ---------------------------------------------------------
# 1. Logical Database & Admin User
# ---------------------------------------------------------
# Provisions a dedicated database and user strictly for the RAG Evaluator on the shared cluster.
# We map directly to the existing instance name via variable injection.

resource "google_sql_database" "rag_db" {
  name     = "rag_evaluation_db"
  instance = var.db_instance_name
}

resource "google_sql_user" "rag_user" {
  name     = "rag_admin"
  instance = var.db_instance_name
  password = var.db_password
}

# ---------------------------------------------------------
# 2. Database Schema Initialization
# ---------------------------------------------------------
# DDL (Tables, Indexes, Keys) is defined in `schema.sql`.
