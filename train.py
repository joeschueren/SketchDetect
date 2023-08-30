import tensorflow as tf
import os
import numpy as np

# set up path to images
folder_path = "training-images"
# gets how many images are in the folder
num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# temporary arrays to accumulate the images in a list
training_images = []
training_labels = []

# get all the images and store them in the training arrays along with them flipped across y-axis
for i in range(1, num_files, 1):
            loaded_image = (np.load("training-images/"+f"training_image{i}.npy").astype(float) / 255)
            loaded_label = (np.load("training-labels/"+f"training_label{i}.npy").astype(float))
            flipped_image = np.fliplr(loaded_image)
            training_images.append(flipped_image)
            training_labels.append(loaded_label)
            training_images.append(loaded_image)
            training_labels.append(loaded_label)

# Reshape the array to fit into the model and turn them into numpy arrays
training_images = np.reshape(np.asarray(training_images), (-1, 400, 400, 1))
training_labels = np.asarray(training_labels)

# gets the test images and put them in an array to test
folder_path = "test-images"
num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

test_images = []
test_labels = []

for i in range(1, num_files, 1):
        test_images.append(np.load("test-images/"+f"test_image{i}.npy").astype(float) / 255)
        test_labels.append(np.load("test-labels/"+f"test_label{i}.npy").astype(float))


test_images = np.reshape(np.array(test_images), (len(test_images), 400, 400, 1))
test_labels = np.array(test_labels)

# sets the value for lambda and the dropout for each layer
lambda_ = 0.01
dropout_enter = 0
dropout_exit = 0.25

# Defines the model to train on
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(12, (6, 6), strides=(1, 1), padding="valid", activation="relu",
        input_shape=(400, 400, 1), kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dropout(dropout_enter),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(12, (8, 8), strides=(1, 1), padding="valid", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dropout(dropout_enter),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(12, (10, 10), strides=(1, 1), padding="valid", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dropout(dropout_exit),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(12, (12, 12), strides=(1, 1), padding="valid", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dropout(dropout_exit),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(20, activation="softmax")
    ])

# compiles model
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# prints summary of model and parameters
model.summary()

# loads the current weights and trains on them
model.load_weights("new_model_3000_0.h5")
model.fit(training_images, training_labels, epochs=15, validation_data=(test_images, test_labels))

# saves the weights to be trained on again
model.save_weights("new_model_3000_0.h5")

# tests unseen images against the model
y_pred = model.predict(test_images)

# defines an array to view what the model is actually predicting
categories = ["umbrella", "house", "sun", "apple", "envelope", "star", "heart",
              "lightning bolt", "cloud", "spoon", "balloon", "mug", "mountains",
              "fish", "bowtie", "ladder", "ice cream cone", "bow", "moon", "smiley"]

total_wrong = {}

# Print the input and output arrays side by side
for i in range(len(y_pred)):
    index1 = np.argmax(test_labels[i])
    index2 = np.argmax(y_pred[i])
    if index1 != index2:
        total_wrong.setdefault(categories[index1], 0)
        total_wrong[categories[index1]] += 1
    print("Input:", categories[index1], "Output:", categories[index2])


test_loss, test_acc = model.evaluate(test_images, test_labels)

print("test accuracy: ", test_acc)
print(total_wrong,)



