import streamlit as st
import requests
import base64
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Diffusion App", layout="centered")

st.title("🧠 Diffusion Model Demo")


def decode_image(b64_string):
    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data))


tab1, tab2 = st.tabs(["🎲 Generate Image", "🧪 Denoise Image"])


# =========================
# GENERATE
# =========================
with tab1:
    st.subheader("Generate from Random Noise")

    if st.button("Generate Image"):
        with st.spinner("Generating..."):

            response = requests.post(API_URL + "/generate")

            if response.status_code == 200:
                data = response.json()

                img = decode_image(data["image"])
                st.image(img, caption="Generated Image", use_column_width=True)

            else:
                st.error("Failed to generate image")


# =========================
# DENOISE
# =========================
with tab2:
    st.subheader("Upload Image → Add Noise → Reconstruct")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Run Denoising"):

            with st.spinner("Processing..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(API_URL + "/denoise", files=files)

                if response.status_code == 200:

                    data = response.json()

                    original = decode_image(data["original"])
                    noisy = decode_image(data["noisy"])
                    reconstructed = decode_image(data["reconstructed"])

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.image(original, caption="Original", use_column_width=True)

                    with col2:
                        st.image(noisy, caption="Noisy Image (x_t)", use_column_width=True)

                    with col3:
                        st.image(reconstructed, caption="Reconstructed", use_column_width=True)

                    st.success("PSNR: " + str(round(data["psnr"], 2)))
                    st.success("SSIM: " + str(round(data["ssim"], 4)))