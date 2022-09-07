ARG python_version_base="3.9"
ARG os_release="bullseye"


FROM python:$python_version_base-$os_release
ARG poetry_version="1.2.0b3"
ARG project="pyrfume"
ARG username="$project"
ARG workdir="/workspace/$project"

SHELL ["/bin/bash", "-eux", "-o", "pipefail", "-c"]

# Create user and group, and grant permissions
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends sudo && \
    apt-get clean && \
    useradd --user-group --create-home --groups users "$username" && \
    echo "$username ALL=(ALL:ALL) NOPASSWD: ALL" | tee "/etc/sudoers.d/$username"

USER "$username"
WORKDIR "$workdir"
ENV PATH="/home/$username/.local/bin:$PATH"

# Install and configure poetry
RUN python -m pip install --upgrade pip && \
    python -m pip install "poetry==$poetry_version"

# Install project dependencies
COPY pyproject.toml poetry.lock ./
RUN export PIP_DEFAULT_TIMEOUT=100 && \
    poetry install --no-root

# Install project
COPY pyrfume .
RUN poetry install

CMD [ "bash" ]

