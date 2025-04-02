import streamlit as st
import requests
import base64
from PIL import Image
import io

# Flask API URL
API_URL = "http://localhost:5000/predict"

st.title("Political Bias Classifier")
st.write("Enter text to classify its political bias.")

# User input
text_input = st.text_area("Enter your text here:")

if st.button("Classify"):
    if text_input.strip():
        # Send request to Flask API
        response = requests.post(API_URL, 
                                 json={"text": text_input})
        
        if response.status_code == 200:
            # Extract predictions
            data = response.json()
            predictions = data.get("predictions", [])
            base64_image = data.get("spider_chart", "")

            # Display predictions
            if predictions:
                st.subheader("Classification Results")
                for prediction in predictions:
                    st.write(f"**{prediction['label']}**: {prediction['percentage']}%")

                # Decode and display the image
                if base64_image:
                    image_data = base64.b64decode(base64_image)
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, 
                             caption="Predicted Biases", 
                             use_column_width=True)
                else:
                    st.error("Error: Image data not received.")
            else:
                st.error("No predictions available.")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
    else:
        st.warning("Please enter some text before classifying.")