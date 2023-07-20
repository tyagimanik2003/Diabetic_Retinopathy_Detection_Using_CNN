import streamlit as st 
from st_on_hover_tabs import on_hover_tabs
import base64
import tensorflow as tf
import numpy as np
from PIL import Image,ImageFilter

@st.cache_data
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        

    }}
    </style>
    """,
    unsafe_allow_html=True
    )
def convert_rgba_to_rgb(rgba_image):
    if rgba_image.mode == 'RGBA':
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save("converted_image.jpg")  # Save the converted image to a new file
        return rgb_image
    else:
        # The image is not in RGBA format, so no conversion is needed
        return None
    
@st.cache_resource
def load_model():

    model=tf.keras.models.load_model('main_model.h5')
    return model 

def load_image(image_file):
	img = Image.open(image_file)
    
	return img

def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Resize the image to the input size required by the model (e.g., 224x224)
    image = image.resize((224, 224))

    # Convert the image to a NumPy array and normalize the pixel values
    image = np.array(image) / 255.0

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    return image

@st.cache_resource
def predict_diabetic_retinopathy(image_file, _model):
    image = load_image(image_file)
    preprocessed_image = preprocess_image(image)

    # Predict the class probabilities for the image using the loaded model
    predictions = _model.predict(preprocessed_image)

    # Assuming the model predicts the probability for each class (e.g., 0 to 4 for diabetic retinopathy stages),
    # you can get the predicted class as follows:
    predicted_class = np.argmax(predictions[0])

    return predicted_class

st.set_page_config(layout="wide")
add_bg_from_local('image.png')
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)


with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'About', 'DR Predictor','Ongoing Research'], 
                         iconName=['home', 'info', 'smart_toy','history_edu'], default_choice=0)

if tabs =='Home':
    st.title("Understanding Diabetic Retinopathy")
    st.write("Diabetic retinopathy is a diabetes complication that affects the blood vessels in the retina, the light-sensitive tissue at the back of the eye. If left untreated, it can lead to vision loss and even blindness. Educating yourself about this condition is crucial to take timely actions and prevent severe consequences.")

# About Diabetic Retinopathy
    st.header("What is Diabetic Retinopathy?")
    st.write("Diabetic retinopathy occurs when high blood sugar levels damage the blood vessels in the retina. The damaged blood vessels may swell and leak, leading to a condition called 'non-proliferative diabetic retinopathy.' As the disease progresses, new blood vessels may grow on the retina, causing 'proliferative diabetic retinopathy.'")

# Symptoms
    st.header("Symptoms of Diabetic Retinopathy")
    st.write("In the early stages, diabetic retinopathy may not cause noticeable symptoms. However, as the disease progresses, the following symptoms may occur:")
    st.markdown("- Blurred vision")
    st.markdown("- Floaters (dark spots or strings floating in your vision)")
    st.markdown("- Difficulty seeing at night")
    st.markdown("- Dark or empty areas in your vision")
    st.markdown("- Vision loss")

# Prevention and Management
    st.header("Prevention and Management")
    st.write("Taking proactive steps to manage diabetes and control blood sugar levels is essential to prevent or slow down the progression of diabetic retinopathy. Regular eye check-ups are recommended, especially for those with diabetes. Additionally, lifestyle changes, such as maintaining a healthy diet and regular exercise, can significantly reduce the risk of complications.")



elif tabs == 'About':
    st.title("Welcome to the Diabetic Retinopathy Detection App!")
    st.write("This app aims to help diagnose diabetic retinopathy, a complication of diabetes that affects the retina of the eye. Early detection of diabetic retinopathy is crucial for timely treatment and prevention of vision loss.")

# How it works section
    st.header("How it works?")
    st.markdown("1. **Upload Retinal Image**: On the \"Predict\" tab, you can easily upload a retinal image by clicking the \"Upload retinal Image\" button. The app supports images in PNG format.")
    st.markdown("2. **Get Predictions**: Once you've uploaded the image, the advanced deep learning model will analyze it to predict whether diabetic retinopathy is detected or not.")
    st.markdown("3. **Instant Results**: In just a few moments, you'll receive a prediction, indicating whether diabetic retinopathy is detected or not. The app will provide information on the predicted class, allowing you to understand the severity level.")

# Why Use Our App section
    st.header("Why Use This App?")
    st.markdown("1. **Fast and Accurate**: The deep learning model ensures speedy and accurate predictions, saving you valuable time.")
    st.markdown("2. **Accessible**: The app is user-friendly, making it easy for anyone to use, even without any technical expertise.")
    st.markdown("3. **Early Detection**: By detecting diabetic retinopathy early, we can take proactive steps to prevent further complications and preserve vision.")

# Privacy and Security section
    st.header("Privacy and Security")
    st.markdown("We prioritize the privacy and security of your data. All images uploaded to the app are processed solely for diagnostic purposes and are not stored or shared with any third parties.")

# Disclaimer section
    st.header("Disclaimer")
    st.markdown("Please note that the predictions made by this app are for informational purposes only. It is essential to consult with a qualified healthcare professional for a definitive diagnosis and personalized treatment plan.")
    
# Learn More section
    st.header("Learn More")
    st.markdown("For more information about diabetic retinopathy and how this app works, check out the \"About\" section.")

# Contact Us section
    st.header("Contact Us")
    st.markdown("If you have any questions, feedback, or concerns, feel free to reach out to me through the \"Contact Us\" section.")

    st.write("Thank you for choosing this Diabetic Retinopathy Detection App. I am committed to making a positive impact on eye health by promoting early detection and intervention.")



elif tabs == 'DR Predictor':
    st.title("Diabetic Retinopathy Detection")
    image_file=st.file_uploader('Upload retinal Image',type="png")

    if image_file is not None:
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        st.write(file_details)
        st.image(load_image(image_file),width=250)

        model = load_model()
        predicted_class = predict_diabetic_retinopathy(image_file, model)
        if predicted_class==1:
            st.write("Predicted Class: No Diabetic Retinopathy detected")
        else:
            st.write("Predicted Class: Diabetic Retinopathy detected")
