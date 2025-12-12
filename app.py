import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import string

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_PATH = 'letter_recognition_cnn.h5'
IMG_SIZE = 32
LABELS = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

# Disable GPU for Inference (optional, prevents conflicts if training runs simultaneously)
# tf.config.set_visible_devices([], 'GPU') 

@st.cache_resource
def load_model():
    """Loads model once and caches it."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def preprocess_image(image_pil):
    """
    1. Convert to Grayscale
    2. Invert (if black text on white bg -> white text on black bg)
    3. Resize to 32x32
    4. Normalize
    """
    img = image_pil.convert('L') # Grayscale
    
    # Smart Inversion:
    # A-Z Dataset is White letters on Black background.
    # Canvas/PDFs are usually Black letters on White background.
    # We invert the colors so it matches the dataset.
    img = ImageOps.invert(img)
    
    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return img_array, img

def predict_result(image_array):
    if model is None: return "Error", 0.0
    
    pred = model.predict(image_array, verbose=0)
    idx = np.argmax(pred)
    conf = np.max(pred)
    return LABELS[idx], conf

# ==========================================
# 🖥️ STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(layout="wide", page_title="AI Letter Recognition")

st.title("Letter Recognition")
st.markdown("Draw a letter or upload an image to detect the character.")

# Layout: Left (3/4) and Right (1/4)
col_draw, col_opt = st.columns([3, 1])

# --- LEFT COLUMN: CANVAS ---
with col_draw:
    st.subheader("Writing Board")
    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=15,
        stroke_color="#000000",      # Black ink
        background_color="#FFFFFF",  # White paper
        height=500,
        width=800,                   # Wide canvas
        drawing_mode="freedraw",
        key="canvas",
    )

# --- RIGHT COLUMN: OPTIONS & RESULT ---
with col_opt:
    st.subheader("Controls")
    
    # 1. READ CANVAS BUTTON
    if st.button("Read Written Letter", type="primary", use_container_width=True):
        if canvas_result.image_data is not None:
            # Convert canvas data (RGBA) to PIL
            raw_numpy = np.array(canvas_result.image_data)
            
            # Check if empty (pure white)
            if np.mean(raw_numpy) > 250: # Assuming white background
                st.warning("Please write something first.")
            else:
                input_image = Image.fromarray(raw_numpy.astype('uint8'), 'RGBA')
                # Process
                processed_arr, view_img = preprocess_image(input_image)
                
                # Predict
                letter, confidence = predict_result(processed_arr)
                
                # Store result in session state to display at bottom
                st.session_state['result'] = (letter, confidence, view_img)

    st.markdown("---")
    
    # 2. FILE UPLOAD
    st.subheader("Upload File")
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Preview", width=150)
        if st.button("Read Uploaded File", use_container_width=True):
            input_image = Image.open(uploaded_file)
            processed_arr, view_img = preprocess_image(input_image)
            letter, confidence = predict_result(processed_arr)
            st.session_state['result'] = (letter, confidence, view_img)

    # --- BOTTOM RIGHT: RESULTS SECTION ---
    st.markdown("---")
    st.subheader("Result")
    
    if 'result' in st.session_state:
        res_letter, res_conf, res_img = st.session_state['result']
        
        # Display large letter
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50; font-size: 80px;'>{res_letter}</h1>", unsafe_allow_html=True)
        st.metric(label="Confidence", value=f"{res_conf*100:.2f}%")
        
        with st.expander("See what model sees"):
            st.image(res_img, width=100, caption="Inverted Input")
    else:
        st.info("Result will appear here.")