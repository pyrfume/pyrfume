ARG PYTHON_VERSION_BASE="3.9"
ARG OS_RELEASE="bullseye"


FROM python:$PYTHON_VERSION_BASE-$OS_RELEASE
ARG POETRY_VERSION="1.2.0b3"
ARG PROJECT_NAME="pyrfume"
ARG PYTHON_VERSION_BASE

ENTRYPOINT [ "bash" ]
CMD [ "poetry", "shell" ]
SHELL ["/bin/bash", "-eux", "-o", "pipefail", "-c"]
WORKDIR "/workspace/$PROJECT_NAME"

# Install and configure poetry
RUN python -m pip install "poetry==$POETRY_VERSION"

# Install project dependencies
COPY pyproject.toml poetry.lock ./
RUN export PIP_DEFAULT_TIMEOUT=100 && \
    poetry install --no-root

# Install project
COPY pyrfume .
RUN poetry install
