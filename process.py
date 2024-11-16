import os
import cv2
import logging
import requests
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier


def aquire_image(url):
    response = requests.get(url, headers = {'User-agent': 'machasins'})
    image_bytes = response.content
    
    # Convert image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

# Function to preprocess the input image
def preprocess_image(image):
    # Resize image to a uniform size (e.g., 300x300 pixels)
    resized_image = cv2.resize(image, (100, 100))
    # Return preprocessed image
    return resized_image

# Function to extract features from the image
def extract_features(image):
    hog_features = []
    for i in range(3):  # Iterate over R, G, B channels
        channel = image[:, :, i]
        channel_hog = hog(channel, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        hog_features.extend(channel_hog)
    features = np.array(hog_features)
    return features

# Aquire, preprocess and extract image features from a url
def process_features(url):
    # Aquire
    image = aquire_image(url)
    # Preprocess
    altered_image = preprocess_image(image)
    # Feature extraction
    features = extract_features(altered_image)
    return features

'''
def loss_leyline(true, pred):
    # Split the outputs for yellow and blue
    y_true_yellow, y_true_blue = true
    y_pred_yellow, y_pred_blue = pred

    # Compute the categorical cross-entropy for each output
    loss_yellow = tf.keras.losses.categorical_crossentropy(y_true_yellow, y_pred_yellow)
    loss_blue = tf.keras.losses.categorical_crossentropy(y_true_blue, y_pred_blue)

    # Add a penalty for overlapping predictions
    overlap_penalty = tf.reduce_sum(tf.multiply(y_pred_yellow, y_pred_blue), axis=-1)

    # Combine the losses with the penalty
    total_loss = loss_yellow + loss_blue + 0.1 * overlap_penalty
    return tf.reduce_mean(total_loss)  # Take the mean across the batch
'''

# Split data into training and testing
def test_split(features, labels) -> list:
    return train_test_split(features, labels, test_size=0.2, random_state=42)

# Function to train the classifier
def train_svc(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = test_split(features, labels)
    # Initialize and train the classifier (e.g., SVM)
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    # Calculate accuracy
    accuracy = classifier.score(X_test, y_test)
    # Return trained classifier and accuracy
    return classifier, accuracy

# Function to train the classifier
def train_mrfc(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = test_split(features, labels)
    # Initialize and train the classifier (e.g., SVM)
    classifier = MultiOutputRegressor(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2))
    classifier.fit(X_train, y_train)
    # Calculate accuracy
    accuracy = classifier.score(X_test, y_test)
    # Return trained classifier and accuracy
    return classifier, accuracy

'''
# Function to train the classifier
def train_nn(features, labels, epoch_count):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = test_split(features, labels)

    num_classes = labels.max() + 1

    # Assuming `labels` has shape (num_samples, 2) for yellow and blue leylines
    y_train_cat = [tf.keras.utils.to_categorical(y_train[:, i], num_classes=num_classes) for i in range(labels.shape[1])]
    y_test_cat = [tf.keras.utils.to_categorical(y_test[:, i], num_classes=num_classes) for i in range(labels.shape[1])]

    # Input layer
    inputs = tf.keras.Input(shape=(features.shape[1],))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # Two output layers for yellow and blue leylines
    output = [tf.keras.layers.Dense(num_classes, activation='softmax', name=f'{i}_output')(x) for i in range(labels.shape[1])]

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    # Step 4: Compile the model with a suitable loss function and optimizer
    model.compile(
        optimizer='adam',
        loss={f'{i}_output': 'categorical_crossentropy' for i in range(labels.shape[1])},
        metrics=['accuracy' for _ in range(labels.shape[1])]
    )

    # Step 5: Train the model
    classifier = model.fit(
        X_train, 
        {f'{i}_output': y_train_cat[i] for i in range(labels.shape[1])}, 
        validation_data=(X_test, {f'{i}_output': y_test_cat[i] for i in range(labels.shape[1])}),
        epochs=epoch_count,
        batch_size=32,
        verbose=0
    )

    # Step 6: Evaluate the model
    accuracy = model.evaluate(X_test, {f'{i}_output': y_test_cat[i] for i in range(labels.shape[1])}, verbose=0)
    # Return trained classifier and accuracy
    return classifier, np.average(accuracy[1:])
'''