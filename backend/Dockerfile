# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files (e.g., model_backend.py and the foaming_fastapi directory)
COPY model_backend.py /app/

# Optionally, if you want to copy your .env file as well, you can add:
COPY .env /app/

# Expose port for FastAPI to run
EXPOSE 8000

# Set the command to run the FastAPI application
CMD ["uvicorn", "model_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
