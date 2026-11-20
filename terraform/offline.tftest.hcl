# Mock resources: never call RunPod. Exercise compute cleanup with retained storage.
mock_provider "runpod" {}

variables {
  execution_file = "../tests/fixtures/execution.json"
}

run "create_study" {
  command = plan
  variables {
    enable_compute = true
  }
  assert {
    condition     = length(runpod_pod.research) == 1 && length(runpod_network_volume.research) == 1
    error_message = "One study pod and one persistent network volume are required."
  }
}

run "release_compute" {
  command = plan
  variables {
    enable_compute = false
  }
  assert {
    condition     = length(runpod_pod.research) == 0 && length(runpod_network_volume.research) == 1
    error_message = "Cleanup must remove compute and preserve network storage."
  }
}

run "use_configured_existing_volume" {
  command = plan
  variables {
    execution_file = "../config/execution.json"
    enable_compute = true
  }
  assert {
    condition     = length(runpod_network_volume.research) == 0 && runpod_pod.research[0].network_volume_id == local.settings.network_volume_id
    error_message = "An existing configured network volume must be attached without creating storage."
  }
  assert {
    condition     = length(runpod_pod.research[0].gpu_type_ids) == 1 && contains(runpod_pod.research[0].gpu_type_ids, local.settings.gpu_type) && runpod_pod.research[0].env.PATH_B_MODEL_PATH == local.execution.model_path
    error_message = "GPU selection and model path must come from execution config."
  }
}

run "release_compute_with_existing_volume" {
  command = plan
  variables {
    execution_file = "../config/execution.json"
    enable_compute = false
  }
  assert {
    condition     = length(runpod_pod.research) == 0 && length(runpod_network_volume.research) == 0
    error_message = "Compute cleanup must not manage an existing external volume."
  }
}
