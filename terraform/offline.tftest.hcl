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
