# evalkit: production image (API + OTLP receiver + dashboard + CLI).
#
#   docker build -t evalkit:0.2.0 --build-arg EXTRAS="server,openai,azure" .
#   docker run -p 8080:8080 -v $PWD/deploy/examples:/etc/evalkit:ro -v evalkit-data:/data evalkit:0.2.0
#
# EXTRAS picks the integrations baked into the image (see pyproject.toml), so
# an AWS-only deployment carries no Azure or Google SDKs.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ARG EXTRAS="server,openai"
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml EVALKIT.md LICENSE ./
COPY evalkit ./evalkit
RUN python -m venv /opt/evalkit \
 && /opt/evalkit/bin/pip install --upgrade pip \
 && /opt/evalkit/bin/pip install ".[${EXTRAS}]"

FROM python:${PYTHON_VERSION}-slim
LABEL org.opencontainers.image.title="evalkit" \
      org.opencontainers.image.description="Vendor-neutral evaluation platform for AI agents" \
      org.opencontainers.image.licenses="LicenseRef-Proprietary"
ENV PATH=/opt/evalkit/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
    EVALKIT_AUDIT_LOG=/data/audit/audit.jsonl
RUN groupadd --system --gid 10001 evalkit \
 && useradd --system --uid 10001 --gid evalkit --home-dir /data --shell /usr/sbin/nologin evalkit \
 && mkdir -p /data /etc/evalkit && chown evalkit:evalkit /data
COPY --from=build /opt/evalkit /opt/evalkit
USER 10001:10001
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"
ENTRYPOINT ["evalkit"]
CMD ["serve", "--config", "/etc/evalkit/server.yaml", "--host", "0.0.0.0", "--port", "8080"]
