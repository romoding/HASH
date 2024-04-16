import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from PIL import Image
import requests
from utils import lime_xai, login, model_predictions, preprocess_image, train_model, register_trained_model, move_model_to_production, deploy_trained_model
from skimage.segmentation import slic, mark_boundaries

#model_endpoint = "http://localhost:7777/invocations"
model_endpoint = "http://127.0.0.1:7777/invocations"

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Define a session state variable to track training status, model name, and deployment status
if 'training_status' not in st.session_state:
    st.session_state.training_status = False

if 'model_name' not in st.session_state:
    st.session_state.model_name = ""

if 'registration_status' not in st.session_state:
    st.session_state.registration_status = False

if 'production_status' not in st.session_state:
    st.session_state.production_status = False

if 'mlflow_run_id' not in st.session_state:
    st.session_state.mlflow_run_id = None

if 'history' not in st.session_state:
    st.session_state.history = None


       
# Title
st.title('HASH Project Dashboard')

# Sidebar
page = st.sidebar.selectbox('Navigation', ["Model Prediction", "Train Model", "Model Analysis"])
st.sidebar.markdown("""---""")
#st.sidebar.write("Created by [MARCONI LAB@MAK](https://marconilab.org/)")
#st.sidebar.image("marc.jpg", width=200)
st.sidebar.write("PROJECT PARTNERS")
st.sidebar.image("marc.jpg", width=100)
st.sidebar.image("ailab.jpg", width=100)
st.sidebar.image("mak.jpg", width=100)
st.sidebar.image("hash.jpg", width=100)

# Parameter initialization
submit = None
uploaded_file = None

if page == "Model Prediction":
    # Inputs
    st.markdown("Select input ultrasound image.")
    upload_columns = st.columns([2, 1])
    
    try:
        # File upload
        file_upload = upload_columns[0].expander(label="Upload an image file.")
        uploaded_file = file_upload.file_uploader("Choose an image file", type=['jpg','png','jpeg'])

        # Validity Check
        if uploaded_file is None:
            st.error("No image uploaded :no_entry_sign:")
        if uploaded_file is not None:
            st.info("Image uploaded successfully :ballot_box_with_check:")

            # Open the image using Pillow
            image = Image.open(uploaded_file)
            upload_columns[1].image(image,caption="Uploaded Image")
            submit = upload_columns[1].button("Submit Image")

    except Exception as e:
        st.error(f"Error during file upload: {str(e)}") 

    # Data Submission
    st.markdown("""---""")
    if submit:
        try:
            with st.spinner(text="Fetching model prediction..."):
                # Preprocess Input Image
                array = preprocess_image(image)
                # Predictions
                probabilities = model_predictions(array)
                # # Image Request
                # image_request = {
                # "instances":array.tolist()}
                # # Response
                # response = requests.post(model_endpoint, json=image_request)
                # # Model Predictions
                # probabilities = eval(response.text)["predictions"]
                #print(f"Probabilities: {probabilities}")
                #prob = np.argmax(probabilities,axis=1)
                prediction = [1 if pred > 0.5 else 0 for pred in probabilities]
                print("Probability: ",probabilities)

            # ----------- Ouputs
            outputs = st.columns([2, 1])
            outputs[0].markdown("Pathology Prediction: ")

            if prediction[0] == 0:
                outputs[1].success("No Myoma")
            elif prediction[0] == 1:
                outputs[1].success("Myoma Detected")
            else:
                outputs[1].error("Error: Invalid Outcome")

            prediction_details = st.expander(label="XAI using LIME")
            details = prediction_details.columns([3, 1])
            with st.spinner(text="Fetching prediction explanations..."):
                # All of this is mocked
                explanation_image_1,explanation_image_2,mask_1,mask_2,img,prediction = lime_xai(uploaded_file,sample_size=100)
                
                # Image and Mask display (Example with matplotlib and Streamlit)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
                plt.subplots_adjust(wspace=0.01, hspace=0)
                ax1.imshow(img)
                ax1.set_title('Input Image')
                ax1.axis('off')
                details[0].image(img,caption="Uploaded Image")
            
                ax2.imshow(mark_boundaries(explanation_image_2/255,mask_2))
                ax2.set_title('LIME Explanation')
                ax2.axis('off')

                ax3.imshow(mark_boundaries(explanation_image_1/255,mask_1))
                ax3.set_title('Regions of Focus')
                ax3.axis('off')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

elif page == "Train Model":
    st.header("Train Model")
    st.markdown("This page will be available soon :no_entry_sign:")
    # # Text input for model name and version
    # model_name = st.text_input("Enter a model name: ")
    # model_version = st.text_input("Enter a model version: ")
    # st.session_state.model_name = model_name
    # st.session_state.model_version = model_version

    # # Text inputs for training parameters
    # learning_rate = st.text_input("Enter learning rate: ", value="0.001")
    # epochs = st.text_input("Enter number of epochs: ", value="10")
    # batch_size = st.text_input("Enter batch size: ", value="32")
    # target_size = st.text_input("Enter target size: ", value="224,224")
    
    # # Text input for label names
    # label_names = st.text_input("Enter label names (comma-separated): ", value="Covid,Healthy,Other")
    # label_names = [label.strip() for label in label_names.split(',')]

    # # File upload for batch training
    # st.markdown("Upload a batch of images for training and testing the model.")
    # train_upload = st.file_uploader("Choose a zip file containing images", type=['zip'])
    
    # if train_upload:
    #     try:
    #         mlflow_run_id = None
    #         if st.button("Train Model"):
    #             with st.spinner(text="Training model..."):
    #                 history, mlflow_run_id = train_model(train_upload,
    #                                             learning_rate=float(learning_rate),
    #                                             epochs=int(epochs),
    #                                             batch_size=int(batch_size),
    #                                             target_size=tuple(map(int, target_size.split(','))),
    #                                             labels=label_names)  
    #                 st.session_state.training_status = True
    #                 st.session_state.mlflow_run_id = mlflow_run_id
    #                 st.session_state.history = history
    #             st.success("Model training completed successfully!")

    #         # Display "Registration" button only if training is completed
    #         if st.session_state.training_status and st.button("Registration"):
    #             with st.spinner(text="Registering model..."):
    #                 mlflow_run_id = st.session_state.mlflow_run_id
    #                 register_trained_model(run_id=mlflow_run_id,model_name=st.session_state.model_name)
    #                 st.session_state.registration_status = True
    #             st.success("Model Registration completed successfully!")

    #         # Display "Production" button only if registration is completed
    #         if st.session_state.registration_status and st.button("Production"):
    #             with st.spinner(text="Transitioning registered model to production stage..."):
    #                 move_model_to_production(model_name=st.session_state.model_name,model_version=st.session_state.model_version)
    #                 st.session_state.production_status = True
    #             st.success("Model Production completed successfully!")

    #         # Display "Deploy" button only if production is completed
    #         if st.session_state.production_status and st.button("Deploy"):
    #             # Set deployment URI
    #             deployment_uri = f'models:/{st.session_state.model_name}/production'
    #             # Deploy the model using MLflow
    #             mlflow_serve_command = f"mlflow models serve --model-uri {deployment_uri} -p 7070 --no-conda"
    #             with st.spinner(text="Deploying model using MLflow..."):
    #                 print(f"{mlflow_serve_command}")
    #                 process = subprocess.Popen(mlflow_serve_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #                 # Wait for the process to complete and get the return code
    #                 #return_code = process.wait()
    #                 print(f"---{process.returncode}---")
    #                 # Capture the output of the command
    #                 out, err = process.communicate(timeout=60)
    #                 print(f"---{out}---{err}---")
    #                 if process.returncode == 0:
    #                     print("---d---")
    #                     st.session_state.production_status = True
    #                     print("---e---")
    #                     deployment_url = f"http://localhost:7070/{model_name}/invocations"
    #                     st.success(f"Model deployment using MLflow completed successfully!\nDeployment URL: {deployment_url}")
    #                 else:
    #                     st.error(f"Error during model deployment: {err.decode()}")

    #     except Exception as e:
    #         st.error(f"Error during model training: {str(e)}")
elif page == "Model Analysis":
    st.header("Model Comparison and Analysis")
    st.markdown("This page will be available soon :no_entry_sign:")

else:
    st.markdown("This page will be available soon :no_entry_sign:")


