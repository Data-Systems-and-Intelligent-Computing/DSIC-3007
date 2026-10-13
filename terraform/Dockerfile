# Build once locally/CI, then set the published image in config/execution.json.
# Torch's locked Linux wheels supply CUDA runtime libraries; no pip at pod boot.
FROM python:3.12.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server rsync libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.4
WORKDIR /opt/path-b
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra figures
ENV PATH="/opt/path-b/.venv/bin:${PATH}" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
COPY terraform/bootstrap.sh /usr/local/bin/path-b-bootstrap
RUN chmod +x /usr/local/bin/path-b-bootstrap
ENTRYPOINT ["/usr/local/bin/path-b-bootstrap"]
