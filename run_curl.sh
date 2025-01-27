# #!/bin/bash

# # Directory containing the JPEG files
# DIRECTORY="/Users/raghav/Downloads/NoFoaming"

# # FastAPI endpoint
# URL="http://127.0.0.1:8000/predict/"

# # Iterate through all JPEG files in the directory
# for file in "$DIRECTORY"/*.jpeg; do
#   echo "Processing file: $file"

#   # Execute the curl command
#   response=$(curl -X 'POST' \
#     "$URL" \
#     -F "file=@$file")
  
#   # Output the response
#   echo "Response: $response"
# done


#!/bin/bash

# Directory containing the JPEG files
DIRECTORY="/Users/raghav/Downloads/NoFoaming"

# FastAPI endpoint inside Docker
URL="http://localhost:8000/predict/"

# Iterate through all JPEG files in the directory
for file in "$DIRECTORY"/*.jpeg; do
  echo "Processing file: $file"

  # Execute the curl command to send the file to the FastAPI endpoint
  response=$(curl -X 'POST' \
    "$URL" \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F "file=@$file")
  
  # Output the response
  echo "Response: $response"
done
