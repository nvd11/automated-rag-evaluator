# ---------------------------------------------------------
# 1. Cloud SQL (PostgreSQL with pgvector extension)
# ---------------------------------------------------------
resource "google_sql_database_instance" "pgvector_instance" {
  name             = var.db_instance_name
  database_version = "POSTGRES_13"
  region           = var.region

  settings {
    tier = "db-custom-2-8192" # 2 vCPU, 8GB RAM

    ip_configuration {
      ipv4_enabled = true
      
      # Expose securely to the evaluator runtime IP
      authorized_networks {
        name  = "evaluator-runtime-access"
        value = "0.0.0.0/0" # In production, restrict to specific VPC/NAT IP
      }
    }
    
    database_flags {
      name  = "cloudsql.enable_pgvector"
      value = "on"
    }
  }
}

# ---------------------------------------------------------
# 2. Logical Database & Admin User
# ---------------------------------------------------------
resource "google_sql_database" "rag_db" {
  name     = "rag_evaluation_db"
  instance = google_sql_database_instance.pgvector_instance.name
}

resource "google_sql_user" "rag_user" {
  name     = "rag_admin"
  instance = google_sql_database_instance.pgvector_instance.name
  password = var.db_password
}

# ---------------------------------------------------------
# 3. Database Schema Initialization (Flyway/Liquibase stub)
# ---------------------------------------------------------
# In a true CI/CD pipeline, tools like Flyway or Liquibase would
# execute the `schema.sql` script to create the evaluation tables.
# For this assignment, the schema is documented in `schema.sql`.
