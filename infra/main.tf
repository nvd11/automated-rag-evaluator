# ---------------------------------------------------------
# 1. Logical Database Provisioning
# ---------------------------------------------------------
# Provisions a dedicated database strictly for the RAG Evaluator on the shared cluster.
# Note: Database users (accounts/passwords) are strictly managed via HashiCorp Vault 
# or GCP Secret Manager in our enterprise environment, not via Terraform state.

resource "google_sql_database" "rag_db" {
  name     = "rag_evaluation_db"
  instance = var.db_instance_name
}

# ---------------------------------------------------------
# 2. Database Schema Initialization
# ---------------------------------------------------------
# DDL (Tables, Indexes, Keys) is defined in `schema.sql`.
