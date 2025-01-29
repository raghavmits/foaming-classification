# Use a base Python image (choose a version that matches your app's requirements)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Set the environment variable for the virtual environment
ENV VIRTUAL_ENV=/app/venv

# Create the virtual environment
RUN python -m venv $VIRTUAL_ENV

# Update the PATH to include the virtual environment's bin directory
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies inside the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the port that the Gradio app will use (Gradio by default uses port 7860)
EXPOSE 7860

# Run the app when the container starts, ensuring Gradio works in headless mode
CMD ["python", "model_frontend.py"]
