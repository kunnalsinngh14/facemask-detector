import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2

st.title("Face Mask Detector 😷")

model = keras.models.load_model('model.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


inputfile = st.file_uploader("upload your img", type=['jpeg', 'jpg', 'png'])

if inputfile is not None:
    img = Image.open(inputfile)
    img = img.convert('RGB')
    st.image(img, width=200, caption='Uploaded img')
    
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_crop = img_np[y:y+h, x:x+w]
        img = Image.fromarray(face_crop)
        # st.success("Face successfully detected and cropped!")
        # st.image(img, width=150, caption='Detected Face')
    else:
        st.warning("No face detected by OpenCV. Falling back to using the full image.")
        
    img = img.resize((150, 150))
    
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    
    pred = model.predict(img_array)
    score = pred[0][0]
    
    if score > 0.5:
        st.header('No Mask!')
        st.subheader(f'Confidence: {score * 100:.2f}%')
    else:
        st.header('Wearing Mask')
        st.subheader(f'Confidence: {(1.0 - score) * 100:.2f}%')
    
if st.button("open camera"):
    st.subheader("press 'q' in the new window to exit the webcam!")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x,y,w,h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            
            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            
            # Resize and prepare for the model
            resized = cv2.resize(roi_rgb, (150, 150))
            img_array = np.array(resized).reshape(1, 150, 150, 3)
            
            pred = model.predict(img_array)
            score = pred[0][0]
            
            if score > 0.5:
                label = "No Mask!"
                color = (0, 0, 255) 
                conf = f"{score * 100:.1f}%"
            else:
                label = "Wearing Mask"
                color = (0, 255, 0) 
                conf = f"{(1.0 - score) * 100:.1f}%"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {conf}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cv2.imshow("webcam view", frame)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
            
    cap.release()
    cv2.destroyAllWindows()