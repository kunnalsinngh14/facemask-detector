import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2

st.title("Face Mask Detector 😷")

model = keras.models.load_model('model.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

input_file = st.file_uploader("upload your img...", type=['jpeg','jpg','png'])
if input_file is not None:
    img = Image.open(input_file)
    img = img.convert('RGB')
    st.image(img,width=200,caption='uploaded img')
    
    # --- OpenCV Face Detection ---
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_crop = img_np[y:y+h, x:x+w]
        img = Image.fromarray(face_crop)
        
    img = img.resize((150,150)) 
    
    img = np.array(img)
    img = img.reshape(1,150,150,3)
    pred = model.predict(img)
    score = pred[0][0]
    
    if (score > 0.5):
        st.header('No Mask!')
        st.subheader(f'Confidence: {score * 100:.2f}%')
    else:
        st.header('Wearing Mask')
        st.subheader(f'Confidence: {(1.0 - score) * 100:.2f}%')