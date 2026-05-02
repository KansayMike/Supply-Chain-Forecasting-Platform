# Dockerfile
# This is the "recipe" for building our Python container

FROM python:3.11-slim


# Install system dependencies (tools that Python libraries need)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
    # apt-get update = "check what packages are available"
    # apt-get install = "download and install these packages"
    # gcc = a compiler (some Python libraries need to compile code)
    # libpq-dev = PostgreSQL client libraries (so Python can talk to our database)
    # rm -rf = "clean up temporary files to keep the image small"

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first (Docker caches this step if it doesn't change)
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt
    # pip = Python's package installer
    # --no-cache-dir = "don't save temporary files, keep the image small"
    # -r requirements.txt = "install everything listed in this file"

# Copy all project files into the container
COPY . .