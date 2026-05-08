# Stage 1: Build React Frontend
FROM node:18-alpine as build-frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python Backend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV and other C-based libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and training code (needed for model structure)
COPY backend/ ./backend/
COPY 2stream_rppg/ ./2stream_rppg/

# Copy built frontend from Stage 1 into a 'static' directory
COPY --from=build-frontend /frontend/dist ./static

# Ensure the model directory exists (even if weights are loaded via LFS)
RUN mkdir -p backend/models

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/2stream_rppg"
ENV PORT=7860

# Expose the default Hugging Face port
EXPOSE 7860

# Run the application
# We use uvicorn to serve the FastAPI app, which will also serve the static frontend
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
