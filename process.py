import os
import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier

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
    classifier = MultiOutputRegressor(RandomForestClassifier(random_state=42))
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