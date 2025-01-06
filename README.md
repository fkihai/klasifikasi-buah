Tentu! Berikut adalah README lengkap dalam satu halaman yang bisa Anda salin:

```markdown
# 🍎 Fruit Maturity Classification with Image Processing 🍌

This project implements a machine learning-based fruit maturity classification using image processing techniques. The classification process is based on the color features of the fruit images, using **K-Nearest Neighbors (KNN)** as the classification model.

## 🧑‍💻 Project Overview

The goal of this project is to classify fruits into three categories based on their maturity:
- 🌱 **Unmature**
- 🍊 **Partially Mature**
- 🍇 **Mature**

The classification is based on image data processed using color features in the **HSV (Hue, Saturation, Value)** color space.

## 📋 Requirements

Before running the code, make sure you have the following Python libraries installed:

- `opencv-python`
- `numpy`
- `scikit-learn`
- `joblib`
- `csv`

You can install them using pip:

```bash
pip install opencv-python numpy scikit-learn joblib
```

## 📂 Dataset

The dataset should be organized in the following structure:

```
dataset/
    unmature/
        image1.jpg
        image2.jpg
        ...
    partiallymature/
        image1.jpg
        image2.jpg
        ...
    mature/
        image1.jpg
        image2.jpg
        ...
```

## 🚀 Usage

To run the code, simply execute the Python script. It will train the KNN classifier on the fruit images and output the model accuracy.

```bash
python fruit_maturity_classification.py
```

### 📌 Features:
- 🖼 **Image Processing**: Using **OpenCV** for image manipulation.
- 🔍 **Feature Extraction**: Extracting the average color features from images in the **HSV** color space.
- 🤖 **KNN Model**: **K-Nearest Neighbors** algorithm used for fruit maturity classification.
- 💾 **Model Saving**: Saved model for future inference.

---

⭐ Happy coding! ⭐
```

Silakan salin dan tempelkan kode di atas ke file README.md Anda di GitHub!
