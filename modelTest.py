import streamlit as st
import pandas as pd
from io import StringIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gdown
import requests

# Set page configuration
st.set_page_config(layout="wide")


st.title("Diabetic Retinopathy Classification Application")

st.markdown("""
Welcome to the Diabetic Retinopathy Classification App!  
- **Sample Prediction**: Test the model on a predefined sample image.  
- **Try It Yourself**: Upload your own image to get a prediction.  
- **Test Model Accuracy**: Upload a ZIP file with images to evaluate the model's accuracy.  
""")

# Define the mapping of class indices to severity levels
severity_levels = ["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"]


# CSS for consistent height across columns
st.markdown(
    """
    <style>
    .column-container {
        display: flex;
        justify-content: space-between;
    }
    .column-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        flex: 1;
        min-height: 100px; /* Adjust this height to match your needs */
        height: auto;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .column-box h3 {
        text-align: center;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def extract_images(zip_path, extract_to, valid_extensions):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            # Check if the file is an image
            if file.lower().endswith(valid_extensions) and not file.startswith("__MACOSX/") and not file.startswith("._"):
                # Extract the file while preserving the subdirectory structure
                zip_ref.extract(file, extract_to)

def test_model(model_name, test_dir):
    with st.spinner('Evaluating model accuracy on uploaded images...'):
        # Load the model
        model = load_model(model_name)

        # Create the test data generator with only rescaling
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)

        # Create the test generator
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False  # No shuffling for evaluation
        )

        # Predict on the test set
        predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
        predicted_classes = np.argmax(predictions, axis=1)  # Get the indices of the highest predicted probability
        actual_classes = test_generator.classes  # True class indices
        class_labels = list(test_generator.class_indices.keys())  # Map indices to class labels

        # Generate a detailed report for each image
        results = []
        for i, image_path in enumerate(test_generator.filenames):
            actual_class = class_labels[actual_classes[i]]
            predicted_class = class_labels[predicted_classes[i]]
            confidence = predictions[i][predicted_classes[i]]
            results.append((image_path, actual_class, predicted_class, confidence))

        # Display the results
        st.write("Prediction Results:")
        for image_path, actual, predicted, confidence in results:
                   st.markdown(
            f"""
            <div class="column-box">
                <ul>
                <li>
                    <strong>Image:</strong> {image_path} |
                    <strong>Actual Category:</strong> {actual_class} |
                    <strong>Predicted Category:</strong> {predicted_class} |
                    <strong>Confidence:</strong> {confidence:.2%}
                </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
            # st.write(f"Image: {image_path}, Actual Category: {actual}, Predicted Category: {predicted}, Confidence: {confidence:.2%}")
            # st.write("---")

        # Calculate and display overall accuracy and loss
        test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // 32)

        st.markdown(
            f"""
            <div class="column-box">
                <ul>
                    <li><strong>Test accuracy:</strong> {test_accuracy:.4f}</li>
                    <li><strong>Test loss:</strong> {test_loss:.4f}%</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

def test_model_Single(model_name, test_dir):
    """Predicts the class of images in an existing test directory."""
    # Load the model
    model = load_model(model_name)

    # Initialize ImageDataGenerator for consistent preprocessing
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Use flow_from_directory to process the image
    single_image_generator = datagen.flow_from_directory(
        test_dir,  # Directory where the uploaded image resides
        target_size=(224, 224),
        batch_size=1,  # Only one image at a time
        class_mode=None,  # No class labels
        shuffle=False  # Ensure order is preserved
    )

    # Predict the image
    prediction = model.predict(single_image_generator)
    predicted_class_index = np.argmax(prediction[0])  # Get index of the highest probability
    confidence = prediction[0][predicted_class_index]  # Confidence score for the prediction

    # Map the predicted index to the severity level (replace with your actual mapping)
    severity_levels = {0: "Mild", 1: "Moderate", 2: "Severe"}  # Example mapping
    predicted_severity = severity_levels.get(predicted_class_index, "Unknown")

    # Display the results
    st.markdown(
        f"""
        <div class="column-box">
            <strong>Prediction Results:</strong><br>
            <ul>
                <li><strong>Severity Level:</strong> {predicted_severity}</li>
                <li><strong>Confidence:</strong> {confidence:.2%}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

model_file = "dr_classification_model.h5"

# **1. Sample Prediction**
st.header("Sample Prediction")
st.image("bf9cba745efc.png", caption="Sample Image")
if st.button("Run Prediction", key="sample"):
     with st.spinner('Running prediction on the sample image...'):
        if os.path.exists("test"):
            import shutil
            shutil.rmtree("test")
        os.makedirs("test")

            # Save the uploaded image to the `test` directory
        image_path = os.path.join("test", "uploaded_image.png")
        shutil.copy("bf9cba745efc.png", image_path)

        model_file = "dr_classification_model.h5"  # Replace with your model file path
        test_model_Single(model_file, "test")


# **2. Try It Yourself**
st.header("Try It Yourself")
uploaded_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.success("Image successfully uploaded")
    st.image(uploaded_image, caption="Uploaded Image")
    if st.button("Predict", key="user-upload"):
        with st.spinner('Running prediction on the uploaded image...'):
            if os.path.exists("test"):
                import shutil
                shutil.rmtree("test")
            os.makedirs("test")

            # Save the uploaded image to the `test` directory
            image_path = os.path.join("test", "uploaded_image.png")
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            test_model_Single(model_file, "test")


# **3. Test Model Accuracy**

st.header("Test Model Accuracy")
uploaded_zip = st.file_uploader("Upload a ZIP file with images:", type=["zip"])
if uploaded_zip:
    st.success("ZIP file successfully uploaded")

    if st.button("Evaluate Accuracy", key="test-accuracy"):

            zip_path = "uploaded.zip"
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            # Clear or recreate the `test` directory
            if os.path.exists("test"):
                import shutil
                shutil.rmtree("test")

            # Extract only image files
            valid_extensions = (".jpg", ".jpeg", ".png")
            extract_images(zip_path, "", valid_extensions)
            st.success("Images extracted to 'test' directory")

            # Display extracted images
            images = [img for img in os.listdir("test") if img.lower().endswith(valid_extensions)]

            # Load and test the model on the extracted images
            test_model(model_file, "test")
             

st.text("")
st.text("")
st.text("")

st.link_button("Go to Model", "https://www.kaggle.com/code/abdulabedin/cps843-project/edit")


url = "https://detectionmodel.s3.us-east-1.amazonaws.com/dr_classification_model.h5"
model_path = "dr_classification_model.h5"

if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        response = requests.get(url)
        f.write(response.content)


# Usage example
if __name__ == "__main__":
    test_dir = 'test'
    model_file = "dr_classification_model.h5"
    
