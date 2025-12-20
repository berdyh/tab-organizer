FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8089

CMD ["streamlit", "run", "app.py", "--server.port=8089", "--server.address=0.0.0.0"]
