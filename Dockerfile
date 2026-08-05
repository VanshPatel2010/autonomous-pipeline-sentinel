FROM python:3.11-slim

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1
# Set PYTHONPATH to the project root
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (e.g., for psycopg2)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose port (Render will dynamically set PORT env var, but this is a fallback)
EXPOSE 8000

# Default command (can be overridden by render.yaml)
# By default, runs the FastAPI server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
