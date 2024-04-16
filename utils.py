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

experiment_name = 'LUS Pathology Classification'
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

def train_model(zip_file, learning_rate, epochs, batch_size, target_size, labels):
    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('data')
    # Extract the zip file name without extension
    zip_name = Path(zip_file.name).stem
    # Define data directories
    train_dir = f'data/{zip_name}/train'
    test_dir = f'data/{zip_name}/test'
    val_dir = f'data/{zip_name}/val'

    # Create data generators
    train_data_generator = ImageDataGenerator(
        rescale = 1./255,
        zoom_range = 0.2,
        #shear_range = 0.2,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip = True,
        #vertical_flip = True,
        fill_mode='nearest')
    
    test_val_data_generator = ImageDataGenerator(
        rescale = 1./255)
    
    train_data = train_data_generator.flow_from_directory(
        directory = train_dir,
        classes = labels,
        class_mode = 'categorical',
        target_size=target_size,
        batch_size = batch_size,
        shuffle=True)

    test_data = test_val_data_generator.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        batch_size = 1,
        shuffle=False,
        class_mode = None)

    val_data = test_val_data_generator.flow_from_directory(
        directory = val_dir,
        classes = labels,
        class_mode = 'categorical',
        target_size=target_size,
        shuffle=True,
        batch_size = 1)
    
    model = Sequential()

    pretrained_model = VGG16(include_top=False,
                            input_shape=(target_size[0],target_size[1],3),
                            classes=len(labels),
                            weights='imagenet')

    for layer in pretrained_model.layers:
        if layer.name in ['block5_conv1','block5_conv2','block5_conv3']:
            layer.trainable = True  # Unfreeze block5 layers
        else:
            layer.trainable = False

    # Add the classification top
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('models/model_{val_accuracy:.3f}.h5',
                                save_best_only=True,
                                monitor='val_accuracy',
                                mode="max")

    history = model.fit(train_data,
                            epochs=epochs,
                            validation_data=val_data,
                            callbacks=[checkpoint],
                            verbose=1)
     
    # Model Evaluation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'tab:blue', label='Training acc')
    plt.plot(epochs, val_acc, 'tab:orange', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('accuracy.png',dpi=600)

    plt.figure()

    plt.plot(epochs, loss, 'tab:blue', label='Training loss')
    plt.plot(epochs, val_loss, 'tab:orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss.png',dpi=600)

    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)

    labels = list(test_data.class_indices.keys())
    cm = confusion_matrix(test_data.classes, y_pred)
    plt.figure(figsize=(5,4))

    sns.heatmap(cm ,cmap='Blues',annot=True,fmt=".0f",xticklabels=labels, yticklabels=labels, annot_kws={
                    'fontsize': 12,
                    'fontweight': 'bold',
                    'fontfamily': 'sans-serif'
                })

    plt.xlabel('Predicted')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.ylabel('Ground Truth')
    plt.ioff()
    plt.savefig('confusion_matrix.png',dpi=600)

    # Using MLFlow to monitor model
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name = run_name) as mlflow_run:
        
        mlflow.set_experiment_tag("base_model", "VGG16")
        mlflow.set_tag("optimizer", "keras.optimizers.Adam")
        mlflow.set_tag("loss", "categorical_crossentropy")

        mlflow.keras.log_model(model, "model")

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_shape", target_size)

        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("train_acc", history.history["accuracy"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("val_acc", history.history["val_accuracy"][-1])

        mlflow.log_artifact("accuracy.png", "training_accuracy_curves")
        mlflow.log_artifact("loss.png", "training_loss_curves")
        mlflow.log_artifact("confusion_matrix.png", "confusion_matrix")

        mlflow_run_id = mlflow_run.info.run_id
    # Clean up extracted data
    shutil.rmtree('data')
    return history, mlflow_run_id

def register_trained_model(run_id,model_name):
    print(f'---{run_id}---')
    # Logged model in MLFlow
    logged_model_path = f"runs:/{run_id}/model"

    with mlflow.start_run(run_id=run_id) as run:
        mlflow.register_model(
            logged_model_path,
            model_name)
        
def move_model_to_production(model_name,model_version):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production")

def deploy_trained_model(model_name):
    # Load model as a Keras model
    model = mlflow.keras.load_model(
        model_uri=f"models:/{model_name}/production")
    
    # Convert the model to TensorFlow Serving format
    tf.saved_model.save(model, 'saved_model')

    pass

def model_predictions(array):
    # Load trained model
    model = load_model('um_model.h5')
    # Preprocess the input image for the model
    preprocessed_image = preprocess_input(array)
    print(f'Processed Image Shape: {preprocessed_image.shape}')
    # Make predictions
    predictions = model.predict(preprocessed_image)
    # Decode predictions
    #decoded_predictions = decode_predictions(predictions, top=3)[0]

    # for i, (id, label, score) in enumerate(decoded_predictions):
    #     print(f"{i + 1}: {label} ({score:.2f})")

    return predictions

# Define the segmentation function
def segment_fn(image):
    segments = slic(image, n_segments=100, compactness=10)
    return segments

def lime_xai(image_path,sample_size=1000,target_size=(224,224)):

    # Load trained model
    model = load_model('um_model.h5')

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

    # fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,18))
    # ax1.imshow(mark_boundaries(explanation_image_1/255,mask_1))
    # ax2.imshow(mark_boundaries(explanation_image_2/255,mask_2))
    # ax3.imshow(img)
    # ax1.set_title('Top Positive Patches')
    # ax2.set_title('Top 5 Patches')
    # ax3.set_title('Input Image')
    # ax1.axis('off')
    # ax2.axis('off')
    # ax3.axis('off')
        
