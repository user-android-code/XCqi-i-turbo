import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import pipeline
import os
from urllib.request import Request, urlopen

st.set_page_config(page_title="2D Parallax Effect Previewer", layout="wide")

st.title("2D Parallax Effect Previewer")

@st.cache_resource
def load_depth_estimator():
    return pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=-1)

@st.cache_resource
def load_lama_model():
    model_path = os.path.expanduser("~/.cache/simple_lama/big-lama.pt")
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Hugging Faceの安定したURLを使用
        url = "https://huggingface.co/anyines/big-lama/resolve/main/big-lama.pt"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req) as response, open(model_path, 'wb') as out_file:
            out_file.write(response.read())
    
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model

def run_lama_inpainting(model, image, mask):
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor, mask_tensor)
    
    out_np = output[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0
    return Image.fromarray(out_np.astype(np.uint8))

depth_estimator = load_depth_estimator()
lama_model = load_lama_model()

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
    
    inpainted_bg = run_lama_inpainting(lama_model, input_image, mask_image)
    
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
