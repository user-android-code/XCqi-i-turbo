import sys
import os
import gc
import base64
import io
import urllib.request
import zipfile
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor

# ---------------------------------------------------------
# 1. GitHubコードの自動取得（2ファイル完結用）
# ---------------------------------------------------------
OVIE_CODE_DIR = os.path.abspath("./ovie_repo")

if not os.path.exists(OVIE_CODE_DIR):
    os.makedirs(OVIE_CODE_DIR, exist_ok=True)
    zip_path = os.path.join(OVIE_CODE_DIR, "ovie.zip")
    url = "https://github.com/kyutai-labs/ovie/archive/refs/heads/main.zip"
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OVIE_CODE_DIR)

extracted_folder = os.path.join(OVIE_CODE_DIR, "ovie-main")
if extracted_folder not in sys.path:
    sys.path.insert(0, extracted_folder)

# ---------------------------------------------------------
# 2. ページ構成 & デザイン（サイドバー消去）
# ---------------------------------------------------------
st.set_page_config(
    page_title="Xcqi i-air",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        .block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("Xcqi i-air")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 3. OVIEモデルロード
# ---------------------------------------------------------
@st.cache_resource
def load_ovie_model():
    from models.models import OVIEModel
    from utils.pose_enc import extri_intri_to_pose_encoding
    
    model = OVIEModel.from_pretrained("kyutai/ovie", revision="v1.0").to(device)
    model.eval()
    return model, extri_intri_to_pose_encoding

try:
    with st.spinner("OVIEモデルをロード中..."):
        model, extri_intri_to_pose_encoding = load_ovie_model()
except Exception as e:
    st.error(f"OVIEモデルのロードに失敗したよ: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. メイン処理 & UI
# ---------------------------------------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    image_size = getattr(model, "image_size", 256)
    img_pil = ImageOps.fit(raw_img, (image_size, image_size), Image.Resampling.LANCZOS)

    file_id = uploaded_file.name + str(uploaded_file.size)
    
    # 画像アップロード時に8方向を一括推論してキャッシュ
    if "ovie_cache" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("OVIEで空間視点を一括生成中..."):
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)
            dummy_intrinsics = torch.zeros(1, 1, 3, 3, device=device)

            angles = {
                "center": (0.0, 0.0),
                "left": (-1.25, 0.0),
                "right": (1.25, 0.0),
                "up": (0.0, 0.5),
                "down": (0.0, -0.5),
            }

            rendered_images = {"center": img_pil}

            for key, (pos_x, pos_y) in angles.items():
                if key == "center":
                    continue
                
                extrinsics = torch.tensor([[[1.0, 0.0, 0.0, pos_x],
                                            [0.0, 1.0, 0.0, pos_y],
                                            [0.0, 0.0, 1.0, -2.0]]], device=device)

                camera = extri_intri_to_pose_encoding(
                    extrinsics=extrinsics.unsqueeze(0),
                    intrinsics=dummy_intrinsics,
                    image_size_hw=(image_size, image_size),
                )
                cam_token = camera[..., :7].squeeze(0)

                with torch.no_grad():
                    pred = model(x=img_tensor, cam_params=cam_token)
                    out_tensor = pred.squeeze(0).cpu()
                    out_img_np = out_tensor.numpy().transpose(1, 2, 0)
                    out_img_np = (np.clip(out_img_np, 0.0, 1.0) * 255).astype(np.uint8)
                    rendered_images[key] = Image.fromarray(out_img_np)

            st.session_state.ovie_cache = rendered_images
            st.session_state.current_key = "center"
            st.session_state.file_id = file_id
            gc.collect()

    # 表示コントロールボタン（十字キー配置）
    col_u1, col_u2, col_u3 = st.columns([1, 1, 1])
    with col_u2:
        if st.button("▲ 上", use_container_width=True):
            st.session_state.current_key = "up"

    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    with col_m1:
        if st.button("◀ 左", use_container_width=True):
            st.session_state.current_key = "left"
    with col_m2:
        if st.button("正面 (0)", use_container_width=True):
            st.session_state.current_key = "center"
    with col_m3:
        if st.button("右 ▶", use_container_width=True):
            st.session_state.current_key = "right"

    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    with col_d2:
        if st.button("▼ 下", use_container_width=True):
            st.session_state.current_key = "down"

    st.markdown("---")

    # 現在選択されている視点の画像を表示
    curr_key = st.session_state.get("current_key", "center")
    display_img = st.session_state.ovie_cache[curr_key]
    
    st.image(display_img, caption=f"視点: {curr_key.upper()}", use_container_width=True)
