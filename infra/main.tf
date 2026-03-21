# ---------------------------------------------------------
# 1. VPC Network & Subnets
# ---------------------------------------------------------
resource "google_compute_network" "rag_vpc" {
  name                    = "rag-eval-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "rag_subnet" {
  name          = "rag-eval-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.rag_vpc.id
}

# ---------------------------------------------------------
# 2. Cloud NAT (For secure outbound traffic & LLM API Calls)
# ---------------------------------------------------------
resource "google_compute_router" "nat_router" {
  name    = "rag-eval-nat-router"
  network = google_compute_network.rag_vpc.name
  region  = var.region
}

resource "google_compute_router_nat" "nat_gateway" {
  name                               = "rag-eval-cloud-nat"
  router                             = google_compute_router.nat_router.name
  region                             = google_compute_router.nat_router.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# ---------------------------------------------------------
# 3. Cloud SQL (PostgreSQL with pgvector extension)
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
# 4. Envoy Proxy Gateway (Compute Engine)
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
    network    = google_compute_network.rag_vpc.name
    subnetwork = google_compute_subnetwork.rag_subnet.name

    access_config {
      nat_ip = google_compute_address.envoy_static_ip.address
    }
  }

  tags = ["envoy-proxy", "postgres-inbound"]
}

# ---------------------------------------------------------
# 5. Security & Firewall Rules
# ---------------------------------------------------------
resource "google_compute_firewall" "allow_postgres_to_envoy" {
  name    = "allow-postgres-5432"
  network = google_compute_network.rag_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["5432"]
  }

  # In reality, restrict to specific interviewers' IPs. Exposed here for evaluation portability.
  source_ranges = ["0.0.0.0/0"] 
  target_tags   = ["postgres-inbound"]
}
