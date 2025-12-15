provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_service" "api" {
  name     = var.service_name
  location = var.region
  template {
    spec {
      containers {
        image = "docker.io/fatchan/cmip6_price_forecaster:latest"
        ports { container_port = 8000 }
        env { name = "MODEL_PATH" value = "models/xgb.pkl" }
        resources {
          limits = { cpu = "1", memory = "2Gi" }
        }
      }
    }
  }
  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.api.name
  location = google_cloud_run_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
