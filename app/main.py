# from fastapi import FastAPI
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# app = FastAPI()

# # Load the model and tokenizer from the saved path
# model_path = './results/checkpoint-225'
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# @app.post("/predict")
# async def predict(text: str):
#     # Tokenize the input text
#     inputs = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Get the prediction
#         prediction = torch.argmax(outputs.logits, dim=1).item()
#     return {"predicted_queue": prediction}

# @app.get("/")
# async def root():
#     return {"message": "FastAPI is working!"}

from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load the model and tokenizer
model_path = './results/checkpoint-225'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Map of label IDs to department names
queue_mapping = {
    0: "Technical Support",
    1: "Product Support",
    2: "Customer Service",
    3: "IT Support",
    4: "Billing and Payments",
    5: "Returns and Exchanges",
    6: "Human Resources",
    7: "Service Outages and Maintenance",
    8: "Sales and Pre-Sales",
    9: "General Inquiry"
}

@app.get("/")
async def root():
    return {"message": "FastAPI is working!"}

@app.post("/predict")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction_id = torch.argmax(outputs.logits, dim=1).item()
        
        # Map the predicted label ID to the department name
        predicted_queue = queue_mapping.get(prediction_id, "Unknown")

    # Return both the ID and the name of the department for clarity
    return {
        "predicted_queue_id": prediction_id,
        "predicted_queue_name": predicted_queue
    }

