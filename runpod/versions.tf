terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = "= 1.0.1"
    }
  }
}

provider "runpod" {}
