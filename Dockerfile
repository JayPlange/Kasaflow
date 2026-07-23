# ---- Builder stage ----
# Installs dependencies into an isolated location. Keeping this separate from
# the final stage means build tools never end up in the image you actually
# ship, only the installed packages do.
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Final stage ----
# This is the image that actually runs. Deliberately does not install
# requirements-dev.txt (pytest, httpx) -- test tooling has no reason to exist
# in a production container, it's dead weight and unnecessary attack surface.
FROM python:3.12-slim

# Run as a non-root user. If this container is ever compromised, the attacker
# doesn't get root inside it for free.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Copy only the installed packages from the builder stage, not its build cache.
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY app/ ./app/
COPY services/ ./services/
COPY data/ ./data/

USER appuser

EXPOSE 8000

# Lets Docker (and later, Kubernetes) know if the app is actually responding,
# not just that the process happens to still be running.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
