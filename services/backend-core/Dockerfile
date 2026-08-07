FROM python:3.12-slim

# Install uv
RUN pip install uv

WORKDIR /app

COPY services/backend-core/requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY services/backend-core/app/ ./app/
COPY services/__init__.py ./services/__init__.py
COPY services/url_safety.py ./services/url_safety.py
COPY services/observability.py ./services/observability.py
COPY services/cors.py ./services/cors.py
COPY config/ ./config/
# `export.html.j2` -- without this, `Exporter.export_html` throws inside its
# try/except on every deployed request and silently returns the plain-HTML
# fallback instead. The template was dead in every container while alive in a
# source checkout.
COPY templates/ ./templates/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
