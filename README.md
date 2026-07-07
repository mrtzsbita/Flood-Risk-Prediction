# Flood Risk Prediction Using Deep Learning (LSTM & GRU)

## Deskripsi
Proyek ini merupakan implementasi model Deep Learning untuk memprediksi risiko banjir menggunakan algoritma Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU). Dataset yang digunakan adalah Flood Prediction Dataset dari Kaggle, dengan target regresi (FloodProbability) yang diubah menjadi klasifikasi biner berdasarkan threshold 0.5.

## Dataset
- Dataset: Flood Prediction Dataset
- Jumlah data: 50.000
- Jumlah fitur: 20
- Target: FloodClass (0 = Risiko Rendah, 1 = Risiko Tinggi)

## Metode
- Preprocessing Data
- Data Splitting (80% Training, 10% Validation, 10% Testing)
- StandardScaler
- LSTM
- GRU
- Evaluasi menggunakan Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix

## Hasil
Model LSTM memberikan performa terbaik dengan hasil evaluasi:

- Accuracy : 99.98%
- Precision : 99.96%
- Recall : 100%
- F1-Score : 99.98%

Model terbaik kemudian dikonversi ke format TensorFlow Lite (.tflite) dan diimplementasikan pada aplikasi GUI berbasis Kivy.

## Struktur Project

```
UAS_MLPRAK/
│── MLProject.ipynb
│── main.py
│── flood.csv
│── scaler.pkl
│── model_prediksi_banjir.tflite
│── README.md
```

## Tools
- Python
- Google Colab
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib
- Kivy