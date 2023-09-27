# SketchDetect AI

SketchDetect AI is a deep learning model created using tensorflow and keras. It is a convolutional neural network that is trained to detect hand made sketches. It was trained on 3000 unique hand-drawn images, but after data augmentation there were 6000 hand-drawn images for the model to train on. The model is accessible through a simple flask application that allows the user to draw and have the model guess what the user has drawn.

# Features

- Users can draw live on the website
- The flask application will display the result from the model for the user to see
- In depth description of how the model was made and reasoning for 
- Over 85% accuracy on over 300 images the model has never seen

# How to Use Model

Ensure you have python and pip installed on your device.

Clone the repository:

`git clone https://github.com/joeschueren/SketchDetect.git`

Download required packages: 

`pip install -r requirements.txt`

Run the app locally: 

`python3 main.py`

Find the application running locally at:

http://localhost:5000

## Live Demo

The live demo for SketchDetect is available here: https://sketchdetect.onrender.com/


