# Use a lightweight Linux version with Python 3.10
FROM python:3.10-slim

# Install system tools needed for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (to cache dependencies)
COPY requirements.txt .

# --- OPTIMIZATION START ---
# 1. Install CPU-only PyTorch first (Fast & Small)
# This prevents downloading 2GB+ of NVIDIA GPU drivers
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the dependencies
# Sentence-Transformers will see that torch is already installed and skip the heavy GPU version
RUN pip install --no-cache-dir -r requirements.txt
# --- OPTIMIZATION END ---

# Copy the rest of your code
COPY . .

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# By default, run the backend (overridden in docker-compose for frontend)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]