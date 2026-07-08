FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install standard Linux build requirements for Postgres connection drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache requirements install layer
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy Django codebase
COPY innerbalance/ /app/

EXPOSE 8000

CMD ["gunicorn", "innerbalance.wsgi:application", "--bind", "0.0.0.0:8000"]
