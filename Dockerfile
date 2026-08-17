# PyVec — single-node vector database.
#
# Slim rather than alpine: NumPy on musl means either compiling from source or
# living without the prebuilt wheels, and the image-size saving is not worth it.
FROM python:3.12-slim

# Vector data is the point of this container, so make the volume explicit.
ENV PYVEC_DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata first, so a source edit does not invalidate the pip layer.
COPY pyproject.toml README.md ./
COPY pyvec ./pyvec
RUN pip install --no-cache-dir .

# Run unprivileged. The volume has to be writable by that user, so create and
# chown it before dropping down.
RUN useradd --create-home --uid 10001 pyvec \
    && mkdir -p /data \
    && chown -R pyvec:pyvec /data /app
USER pyvec

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).status == 200 else 1)"

# Single worker, deliberately. A Collection's RWLock and mmap live in one process;
# multiple uvicorn workers would each open the same files with no coordination
# between them, which is a correctness problem, not a tuning knob. Scaling out
# means sharding, and sharding is an explicit non-goal (PRD N1 / ADR-004).
CMD ["uvicorn", "pyvec.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
