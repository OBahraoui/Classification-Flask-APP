# CIFAR10-Classification-Flask-APP

## Overview

This project is a web application that allows users to upload images for classification using a pre-trained convolutional neural network (CNN) model implemented in PyTorch.

## Features

- Image upload functionality
- Real-time image classification
- Display of predicted class name

## Installation

1. Clone the repository:

git clone https://github.com/OBahraoui/Classification-Flask-APP.git
cd repository

2. Install dependencies:
    pip install -r requirements.txt

3. Run the application:
    python main.py

or 
    gunicorn -b 0.0.0.0:5000 app.main:app

4. Access the application web browser at http://localhost:5000.

## Usage

1. Upload an image using the provided form.
2. Wait for the classification result to appear below the form.

## Deployment
This application is deployed on Render. To deploy your own instance:

1. Create an account on Render and set up a new web service.
2. Link your GitHub repository.
3. Configure the build command (pip install -r requirements.txt) and start command (gunicorn -b 0.0.0.0:5000 app.main:app).
4. Deploy your application.


## Technologies Used

1. Python
2. Flask
3. PyTorch
4. HTML/CSS/JavaScript


## Future Enhancements

1. Add user authentication
2. Improve frontend design
3. Support for batch image processing

## License

This project is licensed under the MIT License.


