import streamlit as st
import pandas as pd
from io import StringIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gdown


st.title("Welcome to the DR Classification App")

st.link_button("Go to Model", "https://www.kaggle.com/code/abdulabedin/cps843-project/edit")

# Define the mapping of class indices to severity levels
severity_levels = ["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"]

# def test_model(model_name, test_dir):
#     model = load_model(model_name)
#     # print(model.summary())

#     # Load and predict for each transformed image in test_dir
#     images = os.listdir(test_dir)

#     for img_name in images:
#         # Only process transformed images (assuming they have a suffix)
#         print(img_name)
#         image_path = os.path.join(test_dir, img_name)

#         # Load and preprocess the image
#         img = image.load_img(image_path, target_size=(224, 224))
#         img_array = image.img_to_array(img) / 255.0  # Convert to array and scale
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # Predict the category
#         prediction = model.predict(img_array)
#         predicted_class_index = np.argmax(prediction[0])  # Get index of the highest probability
#         accuracy = prediction[0][predicted_class_index]  # Get the highest probability (confidence)

#         # Map the predicted class index to the severity level
#         predicted_severity = severity_levels[predicted_class_index]

#         # Print the desired output

#         st.write(f"Image: {img_name} - Predicted Severity: {predicted_severity} - Accuracy: {accuracy:.2%}")


# def test_model(model_name, test_dir):

#     with st.spinner('Wait for it...'):

#         model = load_model(model_name)
#         # print(model.summary())

#         # Create the test data generator with only rescaling
#         test_datagen = ImageDataGenerator(rescale=1.0/255.0)

#         # Create the test generator
#         test_generator = test_datagen.flow_from_directory(
#             test_dir,
#             target_size=(224, 224),
#             batch_size=32,
#             class_mode='categorical',
#             shuffle=False  # No shuffling for evaluation
#         )

#         # Evaluate the model on the test set
#         test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // 32)
#     st.write(f'Test accuracy: {test_accuracy:.4f}')
#     st.write(f'Test loss: {test_loss:.4f}')


def download_model_from_drive():
    file_id = "1RIeCpOTzQTrTKE-jT-Vtpjx5vlsRRzoP"  # Replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dr_classification_model.h5"

    if not os.path.exists(output):
        with st.spinner('Downloading Latest model...'):
            gdown.download(url, output, quiet=False)
            st.success("Download complete")
    return output


model_file = download_model_from_drive() 

def test_model(model_name, test_dir):
    with st.spinner('Wait for it...'):
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
            st.write(f"Image: {image_path}, Actual Category: {actual}, Predicted Category: {predicted}, Confidence: {confidence:.2%}")
            st.write("---")

        # Calculate and display overall accuracy and loss
        test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // 32)
        st.write(f'Test accuracy: {test_accuracy:.4f}')
        st.write(f'Test loss: {test_loss:.4f}')


# Usage example
if __name__ == "__main__":
    test_dir = 'test'
    model_file = "dr_classification_model.h5"
    
    # Run predictions on transformed images
    # test_model(model_file, test_dir)



    

# File uploader for ZIP files
uploaded_file = st.file_uploader("Upload a ZIP file with images", type=["zip"])


# def extract_images(zip_path, extract_to, valid_extensions):
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         for file in zip_ref.namelist():
#             # Check if the file is an image
#             if file.lower().endswith(valid_extensions) and not file.startswith("__MACOSX/") and not file.startswith("._"):
#                 # Extract the file to a temporary location
#                 extracted_path = zip_ref.extract(file, extract_to)
                
#                 # Move the file to the main directory, ignoring subfolders
#                 file_name = os.path.basename(extracted_path)  # Get only the file name
#                 final_path = os.path.join(extract_to, file_name)
                
#                 # Rename/move the file to remove subdirectory structure
#                 os.rename(extracted_path, final_path)

def extract_images(zip_path, extract_to, valid_extensions):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            # Check if the file is an image
            if file.lower().endswith(valid_extensions) and not file.startswith("__MACOSX/") and not file.startswith("._"):
                # Extract the file while preserving the subdirectory structure
                zip_ref.extract(file, extract_to)


if uploaded_file is not None:
    # Save ZIP file to current directory
    zip_path = "uploaded.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("ZIP file successfully uploaded")

    # Clear or recreate the `test` directory
    if os.path.exists("test"):
        import shutil
        shutil.rmtree("test")
    # os.makedirs("test")

    # Extract only image files
    valid_extensions = (".jpg", ".jpeg", ".png")
    extract_images(zip_path, "", valid_extensions)
    st.success("Images extracted to 'test' directory")

    # Display extracted images
    images = [img for img in os.listdir("test") if img.lower().endswith(valid_extensions)]
    # st.write("Uploaded Images:")
    # for img_name in images:
    #     img_path = os.path.join("test", img_name)
        # st.image(img_path, caption=img_name, use_column_width=True)

    # Load and test the model on the extracted images
    model_file = "dr_classification_model.h5"  # Replace with your model file path
    test_model(model_file, "test")


# data = pd.read_csv('train.csv')
# st.write(data)
