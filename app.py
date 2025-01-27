import torch
import gradio as gr
from PIL import Image
from torchvision import models, transforms


# Load the model
device = torch.device("cpu")
model = models.resnet18(pretrained=False)  # Load model architecture without pretrained weights
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)  # Modify for 2 classes
model.load_state_dict(torch.load("./model_weights/resnet_reactor_model.pth", map_location=device))  # Load task-specific weights
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Correct image orientation
def correct_image_orientation(img):
    try:
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(274)  # 274 is the orientation tag
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except AttributeError:
        pass
    return img


# Preprocess image
def preprocess_image(file_path):
    image = Image.open(file_path)
        
    # Correct image orientation
    image = correct_image_orientation(image)

    # Transform image
    return transform(image).unsqueeze(0).to(device)

# Define prediction function
def predict_image(image):
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return "Foaming" if predicted.item() == 0 else "Non-Foaming"
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath"),  # Accept file input to handle .HEIC and other formats
    outputs="text",
    live=True
)

# Launch the Gradio interface
iface.launch(share=True)
