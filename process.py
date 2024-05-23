import cv2
import requests
import numpy as np
import logging as log

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier

def write_log(message, func, is_extra, verbose, end = "\n"):
    if not is_extra or is_extra == verbose:
        log.StreamHandler().terminator = end
        func(message)

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
    resized_image = cv2.resize(image, (300, 300))
    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Return preprocessed image
    return gray_image

# Function to extract features from the image
def extract_features(image):
    # Compute HOG features
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
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

# Function to train the classifier
def train_svc(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Initialize and train the classifier (e.g., SVM)
    classifier = MultiOutputRegressor(RandomForestClassifier(n_estimators=100, random_state=42))
    classifier.fit(X_train, y_train)
    # Calculate accuracy
    accuracy = classifier.score(X_test, y_test)
    # Return trained classifier and accuracy
    return classifier, accuracy