import streamlit as st
import numpy as np
from huggingface_hub import from_pretrained_keras
import keras  # Must be standalone keras==2.13.1
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

model = load_model()

st.title("Low-Light Image Enhancer")

uploaded_file = st.file_uploader("Upload a low-light image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    low_light_img = Image.open(uploaded_file).convert("RGB")
    st.image(low_light_img, caption="Original Image", use_container_width=True)

    # Preprocess image
    low_light_img = low_light_img.resize((256, 256), Image.NEAREST)
    image = keras.preprocessing.image.img_to_array(low_light_img)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    with st.spinner("Enhancing image..."):
        output = model.predict(image)
        output_image = output[0] * 255.0
        output_image = output_image.clip(0, 255).astype('uint8')

        final_img = Image.fromarray(output_image, 'RGB')
        st.image(final_img, caption="Enhanced Image", use_container_width=True)
