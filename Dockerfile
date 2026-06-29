FROM prefecthq/prefect:3-latest

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    git+https://github.com/MaRDI4NFDI/mardiclient.git \
    ".[dev]"

WORKDIR /opt/prefect
ENV PREFECT_LOGGING_LEVEL=INFO
