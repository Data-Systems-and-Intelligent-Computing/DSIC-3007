variable "enable_compute" {
  description = "Create the single Runpod pod. Keep false until paid compute is authorized."
  type        = bool
  default     = false
}

variable "pod_name" {
  description = "Name for the research pod."
  type        = string
  default     = "dsic-3007-research"
}

variable "gpu_type_id" {
  description = "Runpod GPU type selected for the study; no GPU is assumed by this configuration."
  type        = string
  default     = null
  nullable    = true
}

variable "gpu_count" {
  description = "One GPU for the unsharded FP32 runner."
  type        = number
  default     = null
  nullable    = true
}

variable "cloud_type" {
  description = "Runpod cloud type chosen for the study."
  type        = string
  default     = null
  nullable    = true
}

variable "compute_type" {
  description = "Runpod compute type chosen for the study."
  type        = string
  default     = null
  nullable    = true
}

variable "data_center_ids" {
  description = "Runpod data center IDs chosen as the study region."
  type        = list(string)
  default     = []
}

variable "min_ram_per_gpu" {
  description = "Minimum host RAM in GB per GPU, explicitly selected for the model/runtime."
  type        = number
  default     = null
  nullable    = true
}

variable "min_vcpu_per_gpu" {
  description = "Optional minimum host vCPUs allocated per GPU."
  type        = number
  default     = null
  nullable    = true
  validation {
    condition     = var.min_vcpu_per_gpu == null ? true : var.min_vcpu_per_gpu > 0
    error_message = "min_vcpu_per_gpu must be positive when supplied."
  }
}

variable "allowed_cuda_versions" {
  description = "Acceptable host CUDA versions for the selected container runtime."
  type        = list(string)
  default     = []
  nullable    = false
  validation {
    condition = alltrue([
      for version in var.allowed_cuda_versions : contains([
        "13.0", "12.9", "12.8", "12.7", "12.6", "12.5",
        "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"
      ], version)
    ])
    error_message = "Use CUDA versions accepted by the Runpod REST v1 Pod schema; 13.2 is not currently accepted."
  }
}

variable "container_image" {
  description = "Published container reference pinned by sha256 digest."
  type        = string
  default     = null
  nullable    = true
}

variable "container_registry_auth_id" {
  description = "Existing Runpod registry credential ID for the private container image; never the credential itself."
  type        = string
  default     = null
  nullable    = true
}

variable "container_disk_in_gb" {
  description = "Ephemeral container disk size in GB."
  type        = number
  default     = null
  nullable    = true
}

variable "volume_in_gb" {
  description = "Persistent pod volume size in GB when storage_policy is pod-volume."
  type        = number
  default     = null
  nullable    = true
}

variable "storage_policy" {
  description = "Use this pod's persistent volume or attach an existing Runpod network volume."
  type        = string
  default     = "pod-volume"
  validation {
    condition     = contains(["pod-volume", "existing-network-volume"], var.storage_policy)
    error_message = "storage_policy must be pod-volume or existing-network-volume."
  }
}

variable "network_volume_id" {
  description = "Existing Runpod network volume ID when storage_policy is existing-network-volume."
  type        = string
  default     = null
  nullable    = true
}

variable "volume_mount_path" {
  description = "Mount path for the persistent pod volume."
  type        = string
  default     = "/workspace"
}

variable "ports" {
  description = "Container ports to expose, if the selected workflow needs them."
  type        = list(string)
  default     = []
}

locals {
  pod_inputs_present = alltrue([
    var.gpu_type_id != null ? trimspace(var.gpu_type_id) != "" : false,
    var.gpu_count != null ? var.gpu_count == 1 : false,
    var.cloud_type != null ? trimspace(var.cloud_type) != "" : false,
    var.compute_type != null ? trimspace(var.compute_type) != "" : false,
    var.data_center_ids != null ? length(var.data_center_ids) > 0 : false,
    var.min_ram_per_gpu != null ? var.min_ram_per_gpu > 0 : false,
    var.container_image != null ? can(regex("@sha256:[0-9a-f]{64}$", var.container_image)) : false,
    var.container_disk_in_gb != null ? var.container_disk_in_gb > 0 : false,
    (var.storage_policy == "pod-volume" && (var.volume_in_gb != null ? var.volume_in_gb > 0 : false)) ||
    (var.storage_policy == "existing-network-volume" && var.network_volume_id != null ? trimspace(var.network_volume_id) != "" : false),
    var.volume_mount_path != null ? startswith(var.volume_mount_path, "/") : false
  ])
}
