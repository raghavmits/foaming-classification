import gradio as gr
import requests
from io import BytesIO  # Import BytesIO

# FastAPI endpoint URL (adjust to match your FastAPI server's address)
API_URL = "http://localhost:8000/predict/"

# Function to send the image to the FastAPI backend for prediction
def predict_image(image):
    if image is None:
        return "No image uploaded"  # Return a message if no image is uploaded

    try:
        # Convert the image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")  # Save as JPEG, matching your cURL example
        img_byte_arr = img_byte_arr.getvalue()

        # Send the image to FastAPI endpoint using multipart/form-data
        files = {'file': ('image.jpeg', img_byte_arr, 'image/jpeg')}  # Ensure the correct MIME type
        headers = {'accept': 'application/json'}
        response = requests.post(API_URL, files=files, headers=headers)

        # If the response is successful, extract the prediction result
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            return prediction
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error processing image: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),  # Accept image input as PIL image object
    outputs="text",
    live=True
)

# Launch the Gradio interface
iface.launch(server_name="0.0.0.0", server_port=7860)
