# Support Ticket Queue Prediction API

## Overview
This project simulates an end-to-end Data Science pipeline for a multilingual support ticket classifier. Using a transformer model, it predicts the department queue most relevant to each incoming support ticket, based on the ticket text. This pipeline includes data preparation, model training, and an API service for inference, containerized using Docker.

The task covers:
- Loading and preprocessing a multilingual support ticket dataset.
- Training a transformer-based model to classify support tickets by department.
- Serving the model via a FastAPI endpoint.
- Dockerizing the solution with multi-stage builds for training and inference.

## Setup Instructions

### Prerequisites
- Python 3.9
- Docker
- Kaggle account and API access to download the dataset

### Step 1: Clone the Repository
```bash
git clone https://github.com/TimTemi/Ticket-Project.git
cd Ticket-Project
```
### Build and Run the Docker Containers:
- Training Stage:
```bash
docker build --target training -t support-ticket-api-train .
```
- Inference Stage:
```bash
docker build --target inference -t support-ticket-api .
docker run -p 8000:8000 support-ticket-api
```
### Code Structure
- ```train.py```: Contains the code to train the transformer model for classifying customer support tickets.
- ```app/main.py```: The FastAPI application for serving the model and providing a user-friendly prediction endpoint.
- ```preprocess.py```: Handles data cleaning and preprocessing steps.
- ```Dockerfile```: Multi-stage Docker setup for separate training and inference environments.
- ```requirements.txt```: List of dependencies needed to run the project.
- ```data_overview.py``` and ```evaluation.py```: Scripts for data exploration and model evaluation.

### Design Choices and Assumptions
## Data Preparation
1. ### Choice:
- I loaded and explored the dataset using basic EDA techniques to understand its structure and distributions.
Data preprocessing includes language standardization, removal of stop words, tokenization, and any necessary text cleaning.
2. ### Assumptions:
- The dataset is clean and properly formatted.
- The text data is assumed to contain sufficient information for classification.

## Model Selection
1. ### Choice:
- A pre-trained transformer model from HuggingFace was chosen for its efficiency in handling multilingual text data.
- Transformers excel in natural language understanding and support multiple languages.
- Since accuracy isn't the primary concern, I did not fine-tune the model with GPU-based optimizations, prioritizing simplicity instead.
2. ### Assumptions:
- The model can generalize well to unseen support tickets, even with minimal fine-tuning.
- The transformer model has sufficient multilingual capability for this task.

##Docker and FastAPI

1. ### Choice:
- Docker Multi-Stage Build: To create separate environments for training and inference, optimizing the final container image size and reducing build complexity.
- FastAPI: Chosen for its performance and ease of use when serving ML models. FastAPI provides easy API setup and asynchronous handling, making it ideal for production-like environments.
2. ### Assumptions:
- Model artifacts can be reliably saved and loaded between Docker stages.
- FastAPI’s asynchronous processing can handle concurrent requests effectively.

## Limitations
- The model may not be optimized for high-performance, real-time inference due to limited resources (no GPU).
- The training and inference stages were simplified, and hyperparameter tuning was not performed.

## How to Run Inference

After building the Docker image, you can access the API at ```http://127.0.0.1:8000```. Use ```/predict``` to send a POST request with the text of a support ticket to receive the predicted department (queue).

Example Request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -d '{
  "text": "Your support ticket text here"
}'
```
Example Request:

```bash
{
  "predicted_queue_id": 5,
  "predicted_queue_name": "Returns and Exchanges"
}
```
