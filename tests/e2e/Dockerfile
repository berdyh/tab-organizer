FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
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
