# ---------------------------------------------------------
# 1. Existing Enterprise Network & NAT (Data Sources)
# ---------------------------------------------------------
# In an enterprise environment (e.g., HSBC), the core network infrastructure
# is managed by a central platform team. We reference the existing VPC and Subnet.
data "google_compute_network" "tf_vpc0" {
  name = "tf-vpc0"
}

data "google_compute_subnetwork" "tf_vpc0_subnet0" {
  name   = "tf-vpc0-subnet0"
  region = var.region
}

# ---------------------------------------------------------
# 2. Cloud SQL (PostgreSQL with pgvector extension)
# ---------------------------------------------------------
resource "google_sql_database_instance" "pgvector_instance" {
  name             = var.db_instance_name
  database_version = "POSTGRES_13"
  region           = var.region

  settings {
    tier = "db-custom-2-8192" # 2 vCPU, 8GB RAM

    ip_configuration {
      # In a strict production environment, this would be private IP only.
      # For the take-home assignment, we expose it behind an authorized network (Envoy Proxy IP).
      ipv4_enabled = true
      
      authorized_networks {
        name  = "envoy-proxy-access"
        value = "${google_compute_address.envoy_static_ip.address}/32"
      }
    }
    
    database_flags {
      name  = "cloudsql.enable_pgvector"
      value = "on"
    }
  }
}

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
# 3. Envoy Proxy Gateway (Compute Engine)
# ---------------------------------------------------------
resource "google_compute_address" "envoy_static_ip" {
  name   = "static-ip-envoy-proxy"
  region = var.region
}

resource "google_compute_instance" "envoy_proxy" {
  name         = "my-envoy-proxy-vm"
  machine_type = "e2-medium"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 20
    }
  }

  network_interface {
    network    = data.google_compute_network.tf_vpc0.id
    subnetwork = data.google_compute_subnetwork.tf_vpc0_subnet0.id

    access_config {
      nat_ip = google_compute_address.envoy_static_ip.address
    }
  }

  tags = ["envoy-proxy", "postgres-inbound"]
}

# ---------------------------------------------------------
# 4. Security & Firewall Rules
# ---------------------------------------------------------
resource "google_compute_firewall" "allow_postgres_to_envoy" {
  name    = "allow-postgres-5432"
  network = data.google_compute_network.tf_vpc0.id

  allow {
    protocol = "tcp"
    ports    = ["5432"]
  }

  # In reality, restrict to specific interviewers' IPs. Exposed here for evaluation portability.
  source_ranges = ["0.0.0.0/0"] 
  target_tags   = ["postgres-inbound"]
}
