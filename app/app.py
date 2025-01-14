
# Importing essential libraries and modules
import streamlit as st
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

# ------------------------- LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# ------------------------- HELPER FUNCTIONS --------------------------------------------------------

def weather_fetch(city_name):
    """
    Fetch and return the temperature and humidity of a city.
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# ------------------------- STREAMLIT APP ----------------------------------------------------------

st.title("Harvestify: Agriculture Assistant")
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Select a Service",
    ["Home", "Crop Recommendation", "Fertilizer Suggestion", "Disease Detection"]
)

if options == "Home":
    st.image("images/homepage.jpg", use_column_width=True)
    st.write("""
        Welcome to Harvestify! This app provides:
        - Crop Recommendations
        - Fertilizer Suggestions
        - Plant Disease Detection
    """)

elif options == "Crop Recommendation":
    st.header("Crop Recommendation System")
    st.write("Enter the details below to get a crop recommendation:")

    N = st.number_input("Nitrogen (N) Content in Soil", min_value=0, max_value=100)
    P = st.number_input("Phosphorus (P) Content in Soil", min_value=0, max_value=100)
    K = st.number_input("Potassium (K) Content in Soil", min_value=0, max_value=100)
    ph = st.number_input("Soil pH Value", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (in mm)", min_value=0.0, max_value=500.0)
    city = st.text_input("Enter City Name for Weather Data")

    if st.button("Predict"):
        if weather_fetch(city) is not None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)
            st.success(f"The recommended crop for your farm is: {prediction[0]}")
        else:
            st.error("Could not fetch weather data. Please check the city name.")

elif options == "Fertilizer Suggestion":
    st.header("Fertilizer Suggestion System")
    st.write("Enter the details below to get a fertilizer recommendation:")

    crop_name = st.selectbox("Select Crop", fertilizer_dic.keys())
    N = st.number_input("Nitrogen (N) Content in Soil", min_value=0, max_value=100)
    P = st.number_input("Phosphorus (P) Content in Soil", min_value=0, max_value=100)
    K = st.number_input("Potassium (K) Content in Soil", min_value=0, max_value=100)

    if st.button("Suggest Fertilizer"):
        df = pd.read_csv('Data/fertilizer.csv')
        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]
        n, p, k = nr - N, pr - P, kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        key = f"{temp[max(temp.keys())]}{'High' if n < 0 or p < 0 or k < 0 else 'low'}"
        recommendation = fertilizer_dic.get(key, "No suggestion available")
        st.success(f"Fertilizer Suggestion: {recommendation}")

elif options == "Disease Detection":
    st.header("Plant Disease Detection")
    st.write("Upload a leaf image to detect disease:")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        prediction = predict_image(img_bytes)
        st.image(uploaded_file, caption=f"Prediction: {prediction}", use_column_width=True)
        st.success(f"Detected Disease: {disease_dic.get(prediction, 'Unknown Disease')}")
