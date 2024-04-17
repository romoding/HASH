import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
from pathlib import Path
import shutil
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Activation, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D,Dropout,Layer,BatchNormalization
from keras.applications import VGG16
from keras import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import mlflow
import mlflow.keras
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries

experiment_name = 'Myoma Classification'
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    

# Simple login mechanism
def login(user, password):
    # Placeholder for actual login mechanism
    if user == "admin" and password == "Test2024#":
        return True
    else:
        return False

def normalize(input):

    return input / 255.0

def preprocess_image(input_image):
    # Resize the image to 224x224
    image = input_image.resize((224,224))
    # Convert the image to a NumPy array
    array = np.array(image)
    # Expand dimensions
    array = np.expand_dims(array,axis=0)
    # Normalize image array
    array = normalize(array)

    return array
    
def model_predictions(array):
    # Load trained model
    model = load_model('um_modelv1.h5')
    # Make predictions
    predictions = model.predict(array)

    return predictions

# Define the segmentation function
def segment_fn(image):
    segments = slic(image, n_segments=100, compactness=10)
    return segments

def lime_xai(image_path,sample_size=1000,target_size=(224,224)):

    # Load trained model
    model = load_model('um_modelv1.h5')

    explainer = lime_image.LimeImageExplainer()
    img = tf.keras.preprocessing.image.load_img(image_path,target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(img)

    # Explain the prediction using LimeImageExplainer with the custom segmentation function
    explanation = explainer.explain_instance(image.astype('double'), model.predict,
                                            top_labels=3, hide_color=0, num_samples=sample_size,
                                            segmentation_fn=segment_fn)
    prediction = explanation.top_labels[0]

    # Visualize the explanations
    explanation_image_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0],
                                                            positive_only=True,
                                                            negative_only=False,
                                                            num_features=5,
                                                            hide_rest=True)
    # Visualize the explanations
    explanation_image_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0],
                                                            positive_only=False,
                                                            negative_only=False,
                                                            num_features=5,
                                                            hide_rest=False)

    return explanation_image_1,explanation_image_2,mask_1,mask_2,img,prediction
