variable "project_id" {
  description = "The GCP Project ID where resources will be deployed"
  type        = string
  default     = "jason-hsbc"
}

variable "region" {
  description = "The GCP region for the RAG infrastructure"
  type        = string
  default     = "europe-west2"
}

variable "zone" {
  description = "The GCP zone for the RAG infrastructure"
  type        = string
  default     = "europe-west2-c"
}

variable "db_instance_name" {
  description = "The name of the Cloud SQL instance"
  type        = string
  default     = "rag-eval-pgvector-db"
}
