# Use a base Python image (choose a version that matches your app's requirements)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies inside the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY model_frontend.py .

# Expose the port that the Gradio app will use (Gradio by default uses port 7860)
EXPOSE 7860

# Run the app when the container starts, ensuring Gradio works in headless mode
CMD ["python", "model_frontend.py"]
