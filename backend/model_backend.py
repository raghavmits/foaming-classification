from dotenv import load_dotenv
import os
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image, ExifTags
import io
import boto3

# Initialize FastAPI app
app = FastAPI()

# Device configuration
device = torch.device("cpu")

# Load environment variables from .env file
load_dotenv()

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
bucket_name = "learning-aws-rami-bucket"
model_key = "resnet_reactor_model.pth"

# Download and load the model
def load_model():
    # Download the model file from S3
    s3.download_file(bucket_name, model_key, "./resnet_reactor_model.pth")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)  # Modify for 2 classes
    model.load_state_dict(torch.load("./resnet_reactor_model.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Correct image orientation
def correct_image_orientation(image):
    try:
        # Check for EXIF data
        exif = image._getexif()
        if exif is not None:
            for orientation_tag, value in ExifTags.TAGS.items():
                if value == "Orientation":
                    orientation = exif.get(orientation_tag)
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
    except AttributeError:
        pass  # Ignore if image has no EXIF data
    return image

# API endpoint to check the health of the backend
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "FastAPI backend is running!"}


# API endpoint to predict image
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Correct orientation if needed
        image = correct_image_orientation(image)

        # Transform the image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Map prediction to class
        result = "Foaming" if predicted.item() == 0 else "Non-Foaming"
        return JSONResponse(content={"prediction": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

