FROM prefecthq/prefect:3-latest

RUN pip install --no-cache-dir \
    anthropic \
    requests \
    beautifulsoup4 \
    html2text \
    lakefs \
    git+https://github.com/MaRDI4NFDI/mardiclient.git

WORKDIR /opt/prefect
ENV PREFECT_LOGGING_LEVEL=INFO
