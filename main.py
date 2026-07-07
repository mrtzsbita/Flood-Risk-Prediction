# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.core.window import Window
import numpy as np
import tensorflow as tf
import joblib

# Mengubah warna background aplikasi menjadi warna gelap 
Window.clearcolor = (0.1, 0.12, 0.16, 1) 
Window.size = (360, 640)

class FloodApp(App):
    def build(self):
        self.title = 'Aplikasi Deteksi Banjir'
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Load Model TFLite
        self.interpreter = tf.lite.Interpreter(model_path="model_prediksi_banjir.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Judul dengan warna biru
        layout.add_widget(Label(text="Sistem Prediksi Banjir", font_size='22sp', bold=True, color=(0, 1, 1, 1), size_hint=(1, 0.2)))
        
        # --- FUNGSI BANTUAN UNTUK SLIDER ---
        # Menggabungkan Nama Fitur, Slider, dan Angka Real-time dalam 1 baris
        def create_slider_row(text_label, color_rgb):
            # Horizontal untuk mengatur posisi tulisan, slider, dan angka menyamping
            row = BoxLayout(orientation='horizontal', size_hint=(1, 0.15))
            
            # 1. Label Nama Fitur
            lbl = Label(text=text_label, color=color_rgb, size_hint=(0.4, 1))
            
            # 2. Slider
            sld = Slider(min=0, max=20, value=5, size_hint=(0.5, 1))
            
            # 3. Label Angka (Yang akan berubah secara real-time)
            val_lbl = Label(text="5", bold=True, color=color_rgb, size_hint=(0.1, 1))
            
            # Fungsi internal untuk mengupdate tulisan angka saat slider digeser
            def update_label(instance, value):
                val_lbl.text = str(int(value))
                
            # Menghubungkan fungsi update angka ke pergerakan slider
            sld.bind(value=update_label)
            
            # Memasukkan ketiganya ke dalam baris
            row.add_widget(lbl)
            row.add_widget(sld)
            row.add_widget(val_lbl)
            
            return row, sld

        # --- MEMBUAT 3 INPUT DENGAN WARNA BERBEDA ---
        # Input 1: Warna Kuning
        row1, self.s1 = create_slider_row("Monsun (0-20):", (1, 0.8, 0.2, 1)) 
        layout.add_widget(row1)
        
        # Input 2: Warna Hijau Terang
        row2, self.s2 = create_slider_row("Topografi (0-20):", (0.5, 1, 0.5, 1)) 
        layout.add_widget(row2)
        
        # Input 3: Warna Pink / Magenta
        row3, self.s3 = create_slider_row("Sungai (0-20):", (1, 0.5, 0.8, 1)) 
        layout.add_widget(row3)

        # Tombol Prediksi (Warna Biru Terang)
        btn = Button(text="Cek Risiko", bold=True, size_hint=(1, 0.25), background_color=(0.1, 0.6, 1, 1))
        btn.bind(on_press=self.predict)
        layout.add_widget(btn)
        
        # Label Hasil Awal (Putih Keabu-abuan)
        self.result = Label(text="Geser slider & tekan tombol", font_size='18sp', color=(0.8, 0.8, 0.8, 1), size_hint=(1, 0.2))
        layout.add_widget(self.result)
        
        return layout

    def predict(self, instance):
        # Ambil nilai dalam bentuk INTEGER (bilangan bulat) dari slider
        v1, v2, v3 = int(self.s1.value), int(self.s2.value), int(self.s3.value)
        
        # Siapkan array 20 fitur mentah
        input_data = np.array([[v1, v2, v3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]], dtype=np.float32)
        
        # --- KODE YANG DITAMBAHKAN ---
        # Load scaler dan lakukan normalisasi pada input_data agar angkanya menjadi desimal kecil
        scaler = joblib.load("scaler.pkl")
        input_scaled = scaler.transform(input_data)
        
        # Reshape data yang SUDAH di-scale
        input_reshaped = input_scaled.reshape((1, 1, 20))
        # -----------------------------
        
        # Proses Inferensi TFLite
        self.interpreter.set_tensor(self.input_details[0]['index'], input_reshaped)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        
        # Logika Output Warna Saat Ditekan
        if prediction >= 0.5:
            self.result.text = "Risiko Banjir: TINGGI"
            self.result.color = (1, 0.2, 0.2, 1) # Merah Terang
        else:
            self.result.text = "Risiko Banjir: RENDAH"
            self.result.color = (0.2, 1, 0.2, 1) # Hijau Terang

if __name__ == '__main__':
    FloodApp().run()