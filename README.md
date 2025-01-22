# PyTorch Image Classification with Gradio Frontend

This repository contains a PyTorch-based image classification model designed to identify reactor images as either "Foaming" or "Non-Foaming." The project includes a user-friendly frontend built with Gradio, allowing users to upload images and receive predictions instantly.

## Features

- **Deep Learning Model:** Utilizes a ResNet-18 architecture customized for binary classification (foaming vs. non-foaming).
- **Gradio Frontend:** Provides an intuitive and interactive interface for users to test the model with their own images.
- **Custom Preprocessing:** Includes preprocessing steps for resizing, normalization, and orientation correction to handle diverse image inputs.

## Demo

You can interact with the model using the Gradio app. Once launched, the app allows you to upload an image and get a prediction.

## Project Structure

```
.
├── model_weights
│   └── resnet_reactor_model.pth  # Pre-trained model weights
├── app.py                        # Gradio application code
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

## Getting Started

Follow these steps to set up the project on your local machine:

### Prerequisites

- Python 3.8+
- PyTorch (GPU support is optional but recommended for faster inference)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Ensure the pre-trained model weights (`resnet_reactor_model.pth`) are in the `model_weights` directory.

2. Run the Gradio application:

    ```bash
    python app.py
    ```

3. Access the application in your web browser at the URL displayed in the terminal (e.g., `http://127.0.0.1:7860`).

## Model Details

- **Architecture:** ResNet-18
- **Input Size:** Images are resized to 224x224 pixels.
- **Preprocessing:** Includes resizing, normalization, and orientation correction for better compatibility.

## Usage

1. Launch the app.
2. Upload an image in formats like JPEG, PNG, or HEIC.
3. View the prediction displayed as either "Foaming" or "Non-Foaming."

## Key Code Snippets

### Model Loading

```python
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)
model.load_state_dict(torch.load("resnet_reactor_model.pth", map_location=device))
model = model.to(device)
model.eval()
```

### Gradio Interface

```python
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    live=True
)
iface.launch(share=True)
```

## Future Work

- Enhance the model with more diverse training data for improved accuracy.
- Add support for multi-class classification if needed.
- Deploy the application to cloud platforms like AWS or GCP.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)
- [ResNet](https://arxiv.org/abs/1512.03385)

