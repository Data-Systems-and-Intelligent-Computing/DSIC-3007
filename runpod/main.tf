resource "runpod_pod" "research" {
  count = var.enable_compute ? 1 : 0

  name                       = var.pod_name
  image_name                 = var.container_image
  container_registry_auth_id = var.container_registry_auth_id
  gpu_type_ids               = var.gpu_type_id != null ? [var.gpu_type_id] : []
  gpu_count                  = var.gpu_count
  cloud_type                 = var.cloud_type
  compute_type               = var.compute_type
  data_center_ids            = var.data_center_ids != null ? var.data_center_ids : []
  min_ram_per_gpu            = var.min_ram_per_gpu
  min_vcpu_per_gpu           = var.min_vcpu_per_gpu
  allowed_cuda_versions      = length(var.allowed_cuda_versions) > 0 ? var.allowed_cuda_versions : null
  container_disk_in_gb       = var.container_disk_in_gb
  volume_in_gb               = var.storage_policy == "pod-volume" ? var.volume_in_gb : null
  volume_mount_path          = var.volume_mount_path
  network_volume_id          = var.storage_policy == "existing-network-volume" ? var.network_volume_id : null
  env                        = var.container_image != null ? { RESEARCH_IMAGE_DIGEST = var.container_image } : {}
  ports                      = var.ports

  # Start idle. Input review, plan freezing, and explicit operator action are
  # required before any collection or intervention command runs.
  docker_start_cmd = ["/bin/sh", "-lc", "sleep infinity"]

  lifecycle {
    precondition {
      condition     = local.pod_inputs_present
      error_message = "When enable_compute is true, explicitly supply GPU, image@sha256, data center, RAM, disk, and storage settings."
    }
  }
}

output "pod_id" {
  description = "Runpod pod identifier when compute is enabled."
  value       = var.enable_compute ? runpod_pod.research[0].id : null
}
