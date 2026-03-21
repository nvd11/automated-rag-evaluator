terraform {
  backend "gcs" {
    bucket = "jason-hsbc"
    prefix = "terraform/automated-rag-evaluator/state"
  }
}
