# First stage: Training
FROM python:3.9-slim AS training

WORKDIR /app

# Install dependencies only for training
COPY requirements.txt . 
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Copy nltk data into the container
COPY nltk_data /app/nltk_data
ENV NLTK_DATA=/app/nltk_data

# Copy only the necessary code for training
COPY train.py . 
COPY preprocess.py . 
COPY data_overview.py .

# Run the training script (dataset should be mounted or downloaded at runtime)
RUN python train.py

# Second stage: Inference
FROM python:3.9-slim AS inference

WORKDIR /app

# Install dependencies for inference
COPY requirements.txt . 
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Copy nltk data into the container
COPY --from=training /app/nltk_data /app/nltk_data
ENV NLTK_DATA=/app/nltk_data

# Copy the saved model and FastAPI code
COPY --from=training /app/saved_model ./saved_model
COPY app ./app

# Expose port and run FastAPI
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
