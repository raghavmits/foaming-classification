import gradio as gr
import requests
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting Gradio application...")

# FastAPI endpoint URL (adjust to match your FastAPI server's address)
API_URL = "http://54.83.142.161:8000/predict/"

# Function to send the image to the FastAPI backend for prediction
def predict_image(image):
    if image is None:
        logger.warning("No image uploaded")
        return "No image uploaded"

    try:
        # Convert the image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        # Send the image to FastAPI endpoint using multipart/form-data
        files = {'file': ('image.jpeg', img_byte_arr, 'image/jpeg')}
        headers = {'accept': 'application/json'}
        
        logger.info("Sending request to API")
        response = requests.post(API_URL, files=files, headers=headers)

        # If the response is successful, extract the prediction result
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            logger.info(f"Received prediction: {prediction}")
            return prediction
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    live=True
)

if __name__ == "__main__":
    logger.info("Launching interface on http://0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)
    logger.info("Interface closed")



