import gc
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

st.set_page_config(
    page_title="Xcqi i-air",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# サイドバー削除CSS
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        .block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("Xcqi i-air")

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

@st.cache_resource
def load_depth_model():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.eval()
    return image_processor, model

try:
    image_processor, model = load_depth_model()
except Exception as e:
    st.error(f"モデルロード失敗: {e}")
    st.stop()

# 画像生成ロジック
def generate_warped_image(img_256, depth_norm, shift_x, shift_y):
    img_np = np.array(img_256)
    h, w, c = img_np.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    offset_x = (grid_x + shift_x * depth_norm).astype(np.float32)
    offset_y = (grid_y + shift_y * depth_norm).astype(np.float32)
    
    offset_x = np.clip(offset_x, 0, w - 1).astype(np.int32)
    offset_y = np.clip(offset_y, 0, h - 1).astype(np.int32)
    
    warped_img = img_np[offset_y, offset_x]
    return Image.fromarray(warped_img)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_256 = ImageOps.fit(raw_img, (256, 256), Image.Resampling.LANCZOS)

    # 深度マップ計算（キャッシュ化）
    file_id = uploaded_file.name + str(uploaded_file.size)
    if "depth_norm" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("深度を解析中..."):
            inputs = image_processor(images=img_256, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(256, 256),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            st.session_state.depth_norm = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            st.session_state.file_id = file_id
            st.session_state.shift_x = 0
            st.session_state.shift_y = 0

    st.markdown("##### 🖱️ 画像の上でドラッグしてアングルを傾けてね（離すと高画質生成！）")

    # 2次元アングル指定スライダー（ドラッグして「手を離した瞬間」に生成が走るStreamlit標準の挙動）
    col1, col2 = st.columns(2)
    with col1:
        shift_x = st.slider("左右アングル (Yaw)", -25, 25, st.session_state.shift_x, step=1, key="slider_x")
    with col2:
        shift_y = st.slider("上下アングル (Pitch)", -25, 25, st.session_state.shift_y, step=1, key="slider_y")

    # 値が変わったとき（つまみを離したとき）に生成
    with st.spinner("2.5D空間視点を再構成中..."):
        generated_img = generate_warped_image(img_256, st.session_state.depth_norm, shift_x, shift_y)

    st.image(generated_img, caption=f"生成された視点 (X: {shift_x}, Y: {shift_y})", use_container_width=True)

    if st.button("リセット (正面に戻す)", use_container_width=True):
        st.session_state.shift_x = 0
        st.session_state.shift_y = 0
        st.rerun()
