terraform {
  required_version = ">= 1.7, < 2"
  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = "1.0.1"
    }
  }
}

provider "runpod" {}

variable "enable_compute" {
  description = "Create the study pod. False removes only this managed pod and retains network storage."
  type        = bool
  default     = false
}

variable "execution_file" {
  description = "Execution JSON, relative to this Terraform directory or absolute."
  type        = string
  default     = "../config/execution.json"
}

locals {
  execution = jsondecode(file(var.execution_file))
  settings  = local.execution.runpod
}

# An existing volume is referenced, never imported or modified by this module.
resource "runpod_network_volume" "research" {
  count          = local.settings.network_volume_id == "" ? 1 : 0
  name           = "dsic-3007-path-b"
  size           = local.settings.volume_gb
  data_center_id = local.settings.data_center
  lifecycle {
    prevent_destroy = true
    precondition {
      condition     = local.settings.data_center != ""
      error_message = "Set runpod.data_center before provisioning storage."
    }
  }
}

resource "runpod_pod" "research" {
  count                 = var.enable_compute ? 1 : 0
  name                  = "dsic-3007-path-b"
  image_name            = local.settings.image
  gpu_count             = 1
  gpu_type_ids          = [local.settings.gpu_type]
  cloud_type            = "SECURE"
  compute_type          = "GPU"
  data_center_ids       = [local.settings.data_center]
  min_ram_per_gpu       = local.settings.min_ram_gb
  allowed_cuda_versions = local.settings.allowed_cuda_versions
  container_disk_in_gb  = local.settings.container_disk_gb
  network_volume_id     = local.settings.network_volume_id != "" ? local.settings.network_volume_id : runpod_network_volume.research[0].id
  volume_mount_path     = "/workspace"
  ports                 = ["22/tcp"]
  env = {
    PATH_B_MODEL_PATH = "/workspace/models/OLMoE-fp32"
    PUBLIC_KEY        = file(pathexpand(local.settings.ssh_public_key_path))
  }
  lifecycle {
    precondition {
      condition     = local.settings.image != "" && local.settings.data_center != ""
      error_message = "Set the built runtime image and selected data center in config/execution.json."
    }
  }
}

output "pod_id" {
  value = var.enable_compute ? runpod_pod.research[0].id : null
}
output "public_ip" {
  value = var.enable_compute ? runpod_pod.research[0].public_ip : null
}
output "hourly_usd" {
  value = var.enable_compute ? runpod_pod.research[0].cost_per_hr : null
}
output "network_volume_id" {
  value = local.settings.network_volume_id != "" ? local.settings.network_volume_id : runpod_network_volume.research[0].id
}
