import cv2
import joblib
import numpy as np

# Daftar kelas untuk prediksi
classes = ['unmature', 'partiallymature', 'mature']

# Memuat model
model = joblib.load('model/knn_model.pkl')

# Fungsi untuk ekstraksi fitur dari gambar
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize gambar ke ukuran 128x128
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Mengonversi gambar ke ruang warna HSV
    features = np.mean(hsv_image, axis=(0, 1))  # Mengambil rata-rata nilai HSV
    return image,hsv_image,features

def predict_image(image_path):
    image,hsv_image,features = extract_features(image_path)
    prediction = model.predict([features])
    return image,hsv_image,classes[prediction[0]],features  # Mengembalikan nama kelas yang diprediksi

# Path gambar yang akan diuji
test_image = 'buah-test.jpg'

image, hsv_image, result,features = predict_image(test_image)

image = cv2.resize(image, (360, 360))
hsv_image = cv2.resize(hsv_image, (360, 360))
hue, saturation, value = cv2.split(hsv_image)

print("Prediksi Kematangan Buah:", result)
print("HSV:", features)

cv2.imshow("Gambar Buah", image)
cv2.imshow("Hue", hue)
cv2.imshow("Saturation", saturation)
cv2.imshow("Value", value)

cv2.waitKey(0)
cv2.destroyAllWindows()
