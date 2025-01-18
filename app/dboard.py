import streamlit as st
from PIL import Image

# Dashboard başlığı
st.title("Model Training Visualization Dashboard")

# Görsel dosyalarını yükleme ve gösterme
def display_image(file_path, caption):
    image = Image.open(file_path)
    st.image(image, caption=caption, use_container_width=True)

# Dashboard bölümleri
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a visualization:", 
                           ["Model Loss", "Weights Heatmaps", "Biases Histograms"])

if option == "Model Loss":
    st.header("Model Loss Over Epochs")
    display_image("figz/summerized_loss_seaborn.png", "Loss vs Epochs (Train & Validation)")

elif option == "Weights Heatmaps":
    st.header("Weights Heatmaps")
    # Belirli katmanların heatmaplerini ekleyin
    for layer_idx in range(1, 4):  # Örneğin 3 katman
        file_path = f"figz/Heatmap-of-Weights-in-Layer-{layer_idx}.png"
        display_image(file_path, f"Heatmap of Weights - Layer {layer_idx}")

elif option == "Biases Histograms":
    st.header("Biases Histograms")
    # Belirli katmanların histogramlarını ekleyin
    for layer_idx in range(1, 4):  # Örneğin 3 katman
        file_path = f"figz/Biases-of-Layer-{layer_idx}.png"
        display_image(file_path, f"Biases Histogram - Layer {layer_idx}")
