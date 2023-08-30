import random
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import os

# initiates flask app
app = Flask(__name__)

tf.get_logger().setLevel('ERROR')

model = None

model = tf.keras.models.load_model("prod_model.h5")

# loads in the weights of the model
model.load_weights("new_model_3000_0.h5")


# defines home route
@app.route("/")
def home():
    send = ""
    return render_template("index.html", send="")


# defines route to submit user image
@app.route("/guess", methods=["POST"])
def guess():
    # gets the data from the image drawn by user
    image_data = request.form["image_data"]

    # saves the full image data to be used later before it is manipulated
    image_data_full = image_data

    # splits the data into the values needed to make an array
    image_data = image_data.split(",")[1]

    # decodes the data to make it usable
    decoded_data = base64.b64decode(image_data)

    # creates a PIL image
    image = Image.open(BytesIO(decoded_data)).convert('L')

    # turns image into a numpy array and preprocesses the array in the same way as the training images
    image_array = np.reshape(np.array(image).astype(float) / 255, (1,400,400,1))

    # defines the parameters of the model
    lambda_ = 0.01
    dropout_enter = 0
    dropout_exit = 0.25

    #sets the model to be used to predict what the user drew if the model couldn't be loaded
    global model
    if model is None:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(12, (6, 6), strides=(1, 1), padding="valid", activation="relu",
                                   input_shape=(400, 400, 1), kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            tf.keras.layers.Dropout(dropout_enter),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(12, (8, 8), strides=(1, 1), padding="valid", activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            tf.keras.layers.Dropout(dropout_enter),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(12, (10, 10), strides=(1, 1), padding="valid", activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            tf.keras.layers.Dropout(dropout_exit),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(12, (12, 12), strides=(1, 1), padding="valid", activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            tf.keras.layers.Dropout(dropout_exit),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(20, activation="softmax")
        ])

    prediction = model.predict(tf.convert_to_tensor(image_array))

    # turns the output of the model into a human-readable response
    index = prediction.argmax()

    categories = ["umbrella", "house", "sun", "apple", "envelope", "star", "heart",
                  "lightning bolt", "cloud", "spoon", "balloon", "mug", "mountains",
                  "fish", "bowtie", "ladder", "ice cream cone", "bow", "moon", "smiley"]

    # gets the path need to display an example image on the front end
    image_paths = ["umbrella", "house", "sun", "apple", "envelope", "star", "heart",
                  "lightning", "cloud", "spoon", "balloon", "mug", "mountains",
                  "fish", "bowtie", "ladder", "icecream", "bow", "moon", "smiley"]

    # randomly picks on of 3 images to show
    num = random.randint(1, 3)

    image_url = "Images/" + image_paths[index] + str(num) + ".png"

    send = categories[index]

    # renders a template with the guess from the model
    return render_template("guess.html", send=send, index=index, image=image_url, imagedata=image_data_full)


app.run(host="0.0.0.0", port=5000)
