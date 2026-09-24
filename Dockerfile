FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ----------------------------------------------------------------------
# Build args (compose passes the host UID/GID so files in the mounted repo keep the host owner)
# ----------------------------------------------------------------------
ARG UID=1000
ARG GID=1000
ARG USER_NAME=user

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ----------------------------------------------------------------------
# System packages
# ----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# User
# ----------------------------------------------------------------------
# Reuse the group when the host GID already exists in the image (e.g. GID 20 on macOS).
RUN (getent group ${GID} > /dev/null || groupadd -g ${GID} ${USER_NAME}) && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER_NAME} && \
    mkdir -p /app /opt/venv && \
    chown -R ${UID}:${GID} /app /opt/venv

# ----------------------------------------------------------------------
# uv (Python package manager)
# ----------------------------------------------------------------------
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT="/opt/venv" \
    UV_PYTHON_INSTALL_DIR="/opt/uv-python" \
    UV_LINK_MODE=copy

# ----------------------------------------------------------------------
# Python 3.12 + dependencies (uv fetches them according to requires-python in pyproject)
# ----------------------------------------------------------------------
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project && \
    chown -R ${UID}:${GID} /opt/venv /opt/uv-python

# ----------------------------------------------------------------------
# Project source (compose also mounts the repository over /app)
# ----------------------------------------------------------------------
COPY ./src/      /app/src/
COPY ./main.py   /app/
COPY ./README.md /app/

RUN mkdir -p /app/data /app/models /app/outputs && \
    chown -R ${UID}:${GID} /app

USER ${USER_NAME}

CMD ["/bin/bash"]
