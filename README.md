# Machine Learning Model Deployment

This project demonstrates the training and deployment of machine learning models for classifying images in the Chest X-Ray dataset. The web server, built with Flask, allows users to upload images and select a model for classification. This README outlines the structure of the project and the steps to run it.

## Project Structure

- `serving/`: Main directory for the Flask server and associated files.
  - `app.py`: The Flask application.
  - `templates/`: HTML templates for the web interface.
  - `static/`: Directory for static files like CSS and JavaScript (if applicable).
  - 'uploads/': Directory with images uploaded by server users for testing.
- `training/`: Contains all training scripts and notebooks.
  - `models/`: Directory containing the trained machine learning models.
  - 'utils/': Contains the function to preprocess and load the data.
  - 'train.py': Training script.
  - 'data/': Dataset.
  - 'mlruns/': Contains files necessary for MLFlow server.
- `requirements.txt`: Python dependencies required for the project.
- 'venv9/': Directory with the Python 3.9 virtual environment files.
- '.gitattributes': File for the storage of large files.
- '.gitignore': File for ignoring large files for commits.

## Detailed Setup Instructions

Each command in the setup process prepares the environment for running the Flask web server that serves the machine learning models. Here’s a breakdown of what each command accomplishes:

1. **`cd serving`**:
   - This command changes the directory to `serving`, where the Flask application and its related files are located. It's essential because the subsequent commands should be run in the context of this directory.

2. **`python3 -m venv venv9`**:
   - This command we used to create a new virtual environment named `venv9` in the current directory.

3. **`source venv9/bin/activate`**:
   - Activates the virtual environment `venv9` that we created in the previous step. Activation changes the shell’s environment to use the Python and pip executables and libraries from the `venv9` instead of the global Python installation.

4. **`pip3 install -r requirements.txt`**:
   - Installs the Python dependencies listed in `requirements.txt` into the activated virtual environment. These dependencies include Flask and any libraries the project needs to run, such as libraries for loading machine learning models, processing images, etc. This ensures that all necessary Python packages are available to the application.

5. **`python3 app.py`**:
   - Runs the Flask application by executing the `app.py` script with Python. This command starts the Flask development server, making the web application accessible via a web browser. The server listens for requests, such as image uploads for classification, and handles them according to your Flask app's routes and logic.

By following these commands in order, we set up an isolated development environment tailored for this project, install all necessary dependencies, and start the web server ready to classify images using trained machine learning models.
