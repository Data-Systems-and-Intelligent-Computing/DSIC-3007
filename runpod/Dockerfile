# Supply the exact, verified Python base image digest at build time. This keeps
# the OS/runtime base explicit without selecting a GPU or CUDA architecture.
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE} AS dependencies

WORKDIR /opt/dsic3007

# Keep the package manager version stable and the application dependencies
# locked by the repository's uv.lock.
RUN python -m pip install --no-cache-dir "uv==0.11.16"
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# Copy the installed environment, keeping downloaded package caches out of the
# image Runpod pulls. Both stages use the same immutable Python base.
FROM ${PYTHON_BASE_IMAGE} AS runtime
WORKDIR /opt/dsic3007
RUN python -m pip install --no-cache-dir "uv==0.11.16"
COPY pyproject.toml uv.lock .python-version ./
COPY --from=dependencies /opt/dsic3007/.venv/ ./.venv/
ENV PATH="/opt/dsic3007/.venv/bin:${PATH}"

COPY src/ ./src/
COPY configs/ ./configs/

# A pod starts idle. Research commands are always explicit operator actions.
CMD ["sleep", "infinity"]
