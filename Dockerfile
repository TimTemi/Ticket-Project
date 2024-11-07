# First stage: Training
FROM python:3.9-slim AS training

WORKDIR /app

# Install dependencies only for training
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt


# Copy all the code and run the training script
COPY . .
RUN python train.py

# Second stage: Inference
FROM python:3.9-slim AS inference

WORKDIR /app

# Copy only the essentials for inference
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the saved model and FastAPI code
COPY --from=training /app/saved_model ./saved_model
COPY app ./app

# Expose port and run FastAPI
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
