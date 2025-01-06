import os
import cv2
import numpy as np
import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


base_path = "dataset/"
classes = ['unmature', 'partiallymature', 'mature']

data = []
labels = []

def write_csv(data):
    # Simpan ke file CSV
    with open('data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = np.mean(hsv_image, axis=(0, 1))
    return features

for label, fruit_class in enumerate(classes):
    folder_path = os.path.join(base_path, fruit_class)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        features = extract_features(image_path)
        data.append(features)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Membuat model KNN
model = KNeighborsClassifier(n_neighbors=3)

# Melatih model KNN
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# simpan model
joblib.dump(model, 'model/knn_model.pkl')

# Evaluasi akurasi model
print("Akurasi Model:", accuracy_score(y_test, y_pred))




