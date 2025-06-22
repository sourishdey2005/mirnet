import streamlit as st
import numpy as np
from huggingface_hub import from_pretrained_keras
import keras
from PIL import Image
import io
import tensorflow as tf
import gc

# Streamlit Page Config
st.set_page_config(
    page_title="Low-Light Image Enhancer",
    page_icon="🌃",
    layout="centered"
)

# Load MIRNet Model
@st.cache_resource
def load_model():
    return from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

model = load_model()

# Wrap prediction in @tf.function for efficiency
@tf.function
def enhance_image(img, passes):
    for _ in tf.range(passes):
        img = model(img)
    return img

# UI Header
st.markdown("<h1 style='text-align: center;'>🌃 Low-Light Image Enhancer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Boost visibility of dark images using deep learning</p>", unsafe_allow_html=True)
st.divider()

# User Control: Enhancement Passes
passes = st.slider("🔁 Enhancement Passes", 1, 3, 1, help="Number of times the model will re-enhance the image")

# Upload and Enhance Image
uploaded_file = st.file_uploader("📷 Upload a low-light image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    low_light_img = Image.open(uploaded_file).convert("RGB")
    st.image(low_light_img, caption="📉 Original Image", use_container_width=True)

    # Resize + Normalize
    low_light_img = low_light_img.resize((256, 256), Image.LANCZOS)
    image = keras.preprocessing.image.img_to_array(low_light_img)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    with st.spinner("🪄 Enhancing image..."):
        # Efficient multi-pass enhancement
        output = enhance_image(tf.convert_to_tensor(image), passes)
        output_image = output[0].numpy() * 255.0
        output_image = output_image.clip(0, 255).astype('uint8')
        final_img = Image.fromarray(output_image, 'RGB')

        # Display output
        st.image(final_img, caption=f"✨ Enhanced Image (x{passes} pass{'es' if passes > 1 else ''})", use_container_width=True)

        # Download image
        buffer = io.BytesIO()
        final_img.save(buffer, format="PNG")
        st.download_button("⬇️ Download Enhanced Image", data=buffer.getvalue(), file_name="enhanced_image.png", mime="image/png")

        # Clean up memory
        del image, output, final_img, buffer
        gc.collect()

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
