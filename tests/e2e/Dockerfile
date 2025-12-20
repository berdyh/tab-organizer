FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Install dependencies
RUN uv pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-timeout \
    requests \
    selenium \
    playwright

# Install playwright browsers
RUN playwright install chromium

# Copy test files
COPY . /app/tests

CMD ["pytest", "-v", "--tb=short"]
