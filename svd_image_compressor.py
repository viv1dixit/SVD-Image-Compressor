import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

def compress_image_svd(image_array, k):
    if len(image_array.shape) == 2:  # Grayscale
        U, S, Vt = np.linalg.svd(image_array, full_matrices=False)
        compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        return compressed
    else:  # RGB
        channels = []
        for i in range(3):
            U, S, Vt = np.linalg.svd(image_array[:, :, i], full_matrices=False)
            C = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            channels.append(C)
        return np.stack(channels, axis=2)

def main():
    st.title("🧠 SVD Image Compressor")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image = image.resize((256, 256))  # Resize for speed
        st.image(image, caption="Original Image", use_container_width=True)
        gray = image.mode == "L"
        image_array = np.array(image)

        k = st.slider("Select number of singular values (k)", 1, 150, 50)

        compressed = compress_image_svd(image_array, k)
        compressed = np.clip(compressed, 0, 255).astype(np.uint8)

        # Show compressed image
        st.image(compressed, caption=f"Compressed Image (k={k})", use_container_width=True)

        # Show metrics (for grayscale only)
        if gray:
            mse_val = mean_squared_error(image_array, compressed)
            ssim_val = ssim(image_array, compressed)
            st.write(f"**MSE:** {mse_val:.2f}")
            st.write(f"**SSIM:** {ssim_val:.4f}")

if __name__ == "__main__":
    main()
