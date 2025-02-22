import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from transformers import pipeline

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Load a text generation model from Hugging Face
generator = pipeline("text-generation", model="gpt2")  # Use GPT-2 for text generation

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match ResNet50 input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array

# Function to classify the image
def classify_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
    return decoded_predictions

# Function to generate wildlife description
def generate_wildlife_description(species):
    prompt = f"Provide a detailed description of the wildlife species '{species}'. Include its habitat, behavior, diet, and conservation status."
    try:
        response = generator(prompt, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error generating description: {e}"

# Function to answer user questions
def answer_question(species, question):
    prompt = f"The wildlife species is '{species}'. Answer the following question: {question}"
    try:
        response = generator(prompt, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error answering question: {e}"

# Function to generate fun facts
def generate_fun_facts(species):
    prompt = f"Generate 3 fun facts about the wildlife species '{species}'."
    try:
        response = generator(prompt, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error generating fun facts: {e}"

# Streamlit app
st.title("Wildlife Classification App with Generative AI")
st.write("Upload an image of wildlife, and the app will classify it and provide detailed information.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Classify the image
    predictions = classify_image(image)

    # Display predictions
    st.write("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    # Get the top prediction
    top_prediction = predictions[0][1]  # Get the label of the top prediction

    # Generate and display wildlife description
    st.write("")
    st.write(f"Detailed Information about {top_prediction}:")
    description = generate_wildlife_description(top_prediction)
    st.write(description)

    # Generate and display fun facts
    st.write("")
    st.write(f"Fun Facts about {top_prediction}:")
    fun_facts = generate_fun_facts(top_prediction)
    st.write(fun_facts)

    # Allow user to ask questions
    st.write("")
    st.write("Ask a question about the predicted species:")
    user_question = st.text_input("Enter your question:")
    if user_question:
        answer = answer_question(top_prediction, user_question)
        st.write(f"Answer: {answer}")

# Option to load an image from a URL
st.write("Or load an image from a URL:")
image_url = st.text_input("Enter the image URL:")
if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Image from URL', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Classify the image
        predictions = classify_image(image)

        # Display predictions
        st.write("Predictions:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i + 1}: {label} ({score:.2f})")

        # Get the top prediction
        top_prediction = predictions[0][1]  # Get the label of the top prediction

        # Generate and display wildlife description
        st.write("")
        st.write(f"Detailed Information about {top_prediction}:")
        description = generate_wildlife_description(top_prediction)
        st.write(description)

        # Generate and display fun facts
        st.write("")
        st.write(f"Fun Facts about {top_prediction}:")
        fun_facts = generate_fun_facts(top_prediction)
        st.write(fun_facts)

        # Allow user to ask questions
        st.write("")
        st.write("Ask a question about the predicted species:")
        user_question = st.text_input("Enter your question:")
        if user_question:
            answer = answer_question(top_prediction, user_question)
            st.write(f"Answer: {answer}")
    except:
        st.write("Error: Unable to load image from URL.")