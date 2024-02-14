from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import sys
import os
from werkzeug.utils import secure_filename


# Adjust the path to include your model definitions
sys.path.append('../training')

# Directly import the model classes
from models.lenet import LeNet
from models.alexnet import AlexNet
from models.compactnet import CompactCNN
from models.skipconnect import SkipConnectionCNN

from torchvision import datasets, transforms
from PIL import Image

def process_image(image_path):
    """
    Process an image path into a PyTorch tensor ready for model inference.

    Args:
        image_path (str): The file path to the image.

    Returns:
        torch.Tensor: The processed image tensor.
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])

    # Load the image
    image = Image.open(image_path)

    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply the transformations to the image
    processed_image = transform(image)

    # Add a batch dimension (B x C x H x W)
    processed_image = processed_image.unsqueeze(0)

    return processed_image


# Initialize your models
models = {
    'lenet': LeNet(),
    'alexnet': AlexNet(),
    'compactnet': CompactCNN(),
    'skipconnect': SkipConnectionCNN(),
}

# Adjust model_paths to point to the correct location
model_paths = {
    'lenet': '../training/models/checkpoints/LeNet_best_model.pth',
    'alexnet': '../training/models/checkpoints/AlexNet_best_model.pth',
    'compactnet': '../training/models/checkpoints/CompactCNN_best_model.pth',
    'skipconnect': '../training/models/checkpoints/SkipConnectionCNN_best_model.pth',
}

# Load the state dicts
for name, model in models.items():
    model_path = model_paths[name]
    # Ensure the path is absolute to avoid issues
    absolute_model_path = os.path.join(os.getcwd(), model_path)
    model.load_state_dict(torch.load(absolute_model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Save the uploaded file
        image = request.files['image']
        filename = secure_filename(image.filename)
        filepath = os.path.join('uploads', filename)
        image.save(filepath)
        
        # Get the selected model
        model_name = request.form['model']
        model = models[model_name]
        
        # Process the image and predict
        # Assuming you have a function to handle image preprocessing
        input_tensor = process_image(filepath)
        prediction = model(input_tensor).argmax().item()

        if prediction == 0:
            prediction = "Normal"
        else:
            prediction = "Pneumonia"
        
        # Render the result template with prediction
        return render_template('result.html', model_name=model_name.capitalize(), prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


