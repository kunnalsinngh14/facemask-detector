# 😷 Real-Time Face Mask Detection using CNN (MobileNetV2)

A deep learning-based **Face Mask Detection** system built using **TensorFlow**, **Keras**, **MobileNetV2**, **OpenCV**, and **Streamlit**. The application detects a face in an uploaded image, classifies whether the person is wearing a face mask, and displays the prediction with a confidence score.

---

## 📌 Features

* 🧠 Transfer Learning using **MobileNetV2**
* 📷 Automatic face detection using **OpenCV Haar Cascade**
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

1. Upload an image.
2. OpenCV detects the first face in the image.
3. The detected face is cropped.
4. The face is resized to **150×150**.
5. The trained MobileNetV2 model predicts:

   * Wearing Mask
   * No Mask
6. The application displays the prediction along with its confidence score.

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

Upload an image containing a face.

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