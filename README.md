# 😷 Real-Time Face Mask Detection using CNN (MobileNetV2)`

A deep learning-based **Face Mask Detection** system built using **TensorFlow**, **Keras**, **MobileNetV2**, **OpenCV**, and **Streamlit**. The application detects a face in an uploaded image or through webcam, classifies whether the person is wearing a face mask, and displays the prediction with a confidence score.

---

## 📌 Features

* 🧠 Transfer Learning using **MobileNetV2**
* 📷 Automatic face detection using **OpenCV Haar Cascade**
* 🎥 **Live Webcam Support:** Real-time face tracking and mask detection through your camera!
* 🎨 Data augmentation during training
* ⚡ Fast inference with TensorFlow/Keras
* 🌐 Simple and interactive Streamlit web application
* 📈 Binary classification:

  * 😷 Wearing Mask
  * ❌ No Mask

---

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy
* Pillow
* Streamlit

---

## 📂 Project Structure

```
facemask-detector/
│
├── Dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── model.keras
├── src.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Model Architecture

The project uses **Transfer Learning** with MobileNetV2.

```
Input Image (150 × 150 × 3)
        │
Data Augmentation
        │
Rescaling (-1 to 1)
        │
Pre-trained MobileNetV2 (Frozen)
        │
GlobalAveragePooling2D
        │
Dropout (0.2)
        │
Dense (1, Sigmoid)
        │
Prediction
```

---

## 🚀 How It Works

**📸 Image Upload Mode:**
1. Upload an image.
2. OpenCV detects the first face in the image and crops it.
3. The face is resized to **150×150**.
4. The trained MobileNetV2 model predicts the mask status.
5. The application displays the prediction along with its confidence score.

**🎥 Live Webcam Mode:**
1. Click the "open camera" button in the Streamlit UI.
2. The app hooks into your device's webcam using OpenCV.
3. It draws a live tracking box around your face.
4. The MobileNetV2 model predicts your mask status frame-by-frame and overlays the live percentage directly on the video feed!

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/face-mask-detection-cnn.git
```

Move into the project directory:

```bash
cd face-mask-detection-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📷 Demo

You can either **upload an image** containing a face, or click **open camera** to test it live!

Example Output:

```
😷 Wearing Mask
Confidence: 98.74%
```

or

```
❌ No Mask
Confidence: 96.41%
```

---


# 🤝 Contributions

Contributions, suggestions, and feature requests are welcome.

Feel free to fork the repository, open issues, and submit pull requests to improve ThermalWatch.

---

© 2026 Kunal Singh