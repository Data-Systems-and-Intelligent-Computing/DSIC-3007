# Runpod compute

For the currently authorized single L40S diagnostic screen, use the local command
`python runpod/screen.py --run` from the repository root. The user runs this long
command; default invocation without `--run` is read-only. See
[the user-run procedure](../WORKFLOW.md#user-run-bounded-l40s-screen) for the
30-minute/$0.75 planning bound, progress/log paths, teardown and retained volume.
This controller uses the REST v2 API and does not apply the existing Terraform
state, whose historical pod/volume references may be stale.

This configuration describes one idle Runpod pod. `enable_compute` defaults to
`false`; no GPU, image, region, disk, volume, or experiment command is selected
by default. The README research specification requires native FP32 model
weights, so supply a container image that runs that exact path without dtype
fallback. The image must be pinned by immutable `@sha256:` digest.
When compute is explicitly enabled, supply GPU type/count, Runpod cloud and
compute types, data center IDs, minimum RAM per GPU, container disk, and a
storage policy. `pod-volume` uses the pod's persistent volume; selecting
`existing-network-volume` also requires the ID of an already available network
volume. The configuration does not create a second storage resource.

`runpod/Dockerfile` builds a local image from a caller-supplied, digest-pinned
Python base image and the root `uv.lock`. It pins uv to 0.11.16 and installs the
locked project runtime. The build does not publish the image. The default PyPI
PyTorch wheel selection has not been verified against an eventual GPU type;
native FP32 model, CUDA, and GPU execution still require the selected Runpod
hardware and runtime verification.

```sh
docker build -f runpod/Dockerfile \
  --build-arg PYTHON_BASE_IMAGE='python:3.14.6-slim@sha256:<verified-digest>' \
  -t dsic-3007:local .
```

Terraform authenticates with `RUNPOD_API_KEY` from the process environment.
Do not put R2 credentials in Terraform variables or pod `env`: Terraform state
stores provider resource inputs. Configure short-lived/private R2 credentials
inside the running pod through the Runpod console or another approved runtime
secret mechanism before transfer. R2 stays private; the transfer CLI uses no
ACLs or presigned URLs.

After the scientific settings, GPU type, and immutable image are approved, set
the explicit pod values in an ignored `terraform.tfvars` file. The minimum
workflow is:

```sh
cd runpod
terraform init
terraform plan -var='enable_compute=true' -out=research.tfplan
terraform apply research.tfplan
```

These commands are documentation only. Do not run `apply` until paid compute is
authorized. The pod starts idle and does not download data or launch an
experiment. Keep the persistent volume until every result bundle has been
retrieved locally and verified.

Use `python -m src.transfer push --kind prepared-inputs --bundle <sealed-bundle>`
to stage approved inputs. Retrieve measurements with
`python -m src.transfer pull --kind measurements --bundle-id <manifest-sha256> --destination <new-local-directory>`.
Other kinds are `inventory`, `experiment-plan`,
`calibration`, and `local-analysis`. Each transfer handles one sealed bundle;
its manifest is published last and every downloaded file is rechecked locally.

Code is installed in /opt/dsic3007 and the image's virtual environment is on
PATH. Persistent inputs/results belong under /workspace; mounting that volume
does not hide the application. Enter /opt/dsic3007 before using the Python CLI.
The unsharded runner uses one GPU; select hardware with enough memory for the
entire original FP32 model and 4096-token eager attention, then establish fit
with native technical measurements. No GPU type is assumed.
