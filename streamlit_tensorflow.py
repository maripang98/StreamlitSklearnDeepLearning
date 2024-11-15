import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Definisikan jalur model
model_path = r'best_model_tf.h5'

# Muat model
if os.path.exists(model_path):
    try:
        # Mengurangi verbosity dari TensorFlow
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path, compile=False)

        # Nama kelas untuk Fashion MNIST
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Fungsi untuk memproses gambar
        def preprocess_image(image):
            image = image.resize((28, 28))  # Ubah ukuran menjadi 28x28 piksel
            image = image.convert('L')      # Ubah menjadi grayscale
            image_array = np.array(image) / 255.0  # Normalisasi
            image_array = image_array.reshape(1, 28, 28, 1)  # Ubah bentuk menjadi 4D array
            return image_array

        # UI Streamlit
        st.title("Fashion MNIST Image Classifier")
        st.write("Unggah gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelasnya.")

        # File uploader untuk input gambar
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

        # Tampilkan gambar yang diunggah dan tombol "Predict"
        if uploaded_file is not None:
            # Tampilkan gambar yang diunggah
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

            # Tombol "Predict"
            if st.button("Predict"):
                # Proses gambar dan prediksi
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)[0]
                
                # Mendapatkan kelas dan confidence dengan softmax
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100  # Pastikan confidence dalam persentase

                # Tampilkan hasil prediksi
                st.write("### Hasil Prediksi")
                st.write(f"Kelas Prediksi: **{class_names[predicted_class]}**")
                st.write(f"Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.error("File model tidak ditemukan.")
