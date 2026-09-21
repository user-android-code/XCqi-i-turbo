import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import pipeline
from simple_lama_inpainting import SimpleLama

st.set_page_config(page_title="2D Parallax Effect Previewer", layout="wide")

st.title("2D Parallax Effect Previewer")

@st.cache_resource
def load_depth_estimator():
    # device=-1 で明示的にCPUを使うように指定するよ
    return pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=-1)

@st.cache_resource
def load_lama_inpainter():
    # SimpleLamaもCPUで動くように設定
    return SimpleLama(device=torch.device('cpu'))

depth_estimator = load_depth_estimator()
lama_inpainter = load_lama_inpainter()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Original Image")
        st.image(input_image, use_container_width=True)
    
    depth_result = depth_estimator(input_image)
    depth_map = depth_result["depth"]
    
    with col2:
        st.write("Depth Map")
        st.image(depth_map, use_container_width=True)
    
    depth_array = np.array(depth_map)
    
    st.write("Layer Separation Settings")
    threshold = st.slider("Foreground Threshold", min_value=0, max_value=255, value=128)
    
    mask_array = np.where(depth_array > threshold, 255, 0).astype(np.uint8)
    mask_image = Image.fromarray(mask_array)
    
    inpainted_bg = lama_inpainter(input_image, mask_image)
    
    fg_array = np.array(input_image.convert("RGBA"))
    fg_array[:, :, 3] = mask_array
    fg_image = Image.fromarray(fg_array)
    
    st.write("Parallax Controls")
    shift_x = st.slider("Horizontal Shift (X)", min_value=-50, max_value=50, value=0)
    shift_y = st.slider("Vertical Shift (Y)", min_value=-50, max_value=50, value=0)
    
    def render_parallax(bg, fg, sx, sy, scale=1.1):
        w, h = bg.size
        sw, sh = int(w * scale), int(h * scale)
        bg_scaled = bg.resize((sw, sh), Image.Resampling.LANCZOS)
        
        cx = (sw - w) // 2 - int(sx * 0.4)
        cy = (sh - h) // 2 - int(sy * 0.4)
        cx = max(0, min(cx, sw - w))
        cy = max(0, min(cy, sh - h))
        
        canvas = bg_scaled.crop((cx, cy, cx + w, cy + h)).convert("RGBA")
        
        fg_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fg_x = int(sx * 0.6)
        fg_y = int(sy * 0.6)
        fg_canvas.paste(fg, (fg_x, fg_y), fg)
        
        canvas.paste(fg_canvas, (0, 0), fg_canvas)
        return canvas

    parallax_result = render_parallax(inpainted_bg, fg_image, shift_x, shift_y)
    
    st.write("Parallax Preview")
    st.image(parallax_result, use_container_width=True)
