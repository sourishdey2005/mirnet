import streamlit as st
import numpy as np
from huggingface_hub import from_pretrained_keras
import keras
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Low-Light Image Enhancer",
    page_icon="🌃",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

model = load_model()

# App Header
st.markdown("<h1 style='text-align: center;'>🌃 Low-Light Image Enhancer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Boost visibility of dark images using deep learning</p>", unsafe_allow_html=True)
st.divider()

# Slider
passes = st.slider("🔁 Enhancement Passes", min_value=1, max_value=5, value=2, step=1, help="Number of times the image is enhanced")

# File uploader
uploaded_file = st.file_uploader("📷 Upload a low-light image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    low_light_img = Image.open(uploaded_file).convert("RGB")
    st.image(low_light_img, caption="📉 Original Image", use_container_width=True)

    # Preprocess image
    low_light_img = low_light_img.resize((256, 256), Image.NEAREST)
    image = keras.preprocessing.image.img_to_array(low_light_img)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    with st.spinner("🪄 Enhancing image..."):
        # Iterative enhancement
        output = image
        for _ in range(passes):
            output = model.predict(output)

        # Postprocess
        output_image = output[0] * 255.0
        output_image = output_image.clip(0, 255).astype('uint8')
        final_img = Image.fromarray(output_image, 'RGB')

        st.image(final_img, caption=f"✨ Enhanced Image (x{passes} pass{'es' if passes > 1 else ''})", use_container_width=True)

        # Download button
        img_buffer = io.BytesIO()
        final_img.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=img_bytes,
            file_name="enhanced_image.png",
            mime="image/png"
        )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; font-size: 14px;'>
        Made with ❤️ by <strong>Sourish</strong> (Team <strong>CodeKarma</strong>)<br>
        Under <strong>Bharatiya Antariksh Hackathon 2025</strong> – An <strong>ISRO</strong> Initiative 🚀
    </div>
    """,
    unsafe_allow_html=True
)
