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

# ---------------------------------------------------------
# 1. GitHubからリポジトリを自動取得＆安全解凍
# ---------------------------------------------------------
REPO_DIR = os.path.abspath("./dvlt_github_repo")

if not os.path.exists(REPO_DIR):
    os.makedirs(REPO_DIR, exist_ok=True)
    zip_path = os.path.join(REPO_DIR, "repo.zip")
    url = "https://github.com/nv-tlabs/dvlt/archive/refs/heads/main.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(REPO_DIR)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        st.error(f"GitHubからのコードダウンロードに失敗しました: {e}")

extracted_folder = os.path.join(REPO_DIR, "dvlt-main")
if extracted_folder not in sys.path and os.path.exists(extracted_folder):
    sys.path.insert(0, extracted_folder)

# ---------------------------------------------------------
# 2. ページ構成
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
# 3. DVLTモデル構造の動的ロード
# ---------------------------------------------------------
@st.cache_resource
def load_github_dvlt():
    import importlib
    
    model = None
    # リポジトリ内のモデル定義を安全に読み込み
    try:
        from models.dvlt import DVLT
        model = DVLT()
    except Exception:
        try:
            from models import dvlt
            model = dvlt()
        except Exception:
            try:
                mod = importlib.import_module("models")
                model_cls = getattr(mod, "DVLT", getattr(mod, "dvlt", None))
                if model_cls is not None:
                    model = model_cls()
            except Exception:
                pass

    if model is None:
        raise RuntimeError("モデル構造の読み込みに失敗しました。")

    model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model

try:
    with st.spinner("DVLTモデルを初期化中..."):
        model = load_github_dvlt()
except Exception as e:
    st.warning(f"モデルの直接ロードを準備中: {e}")
    model = None

# ---------------------------------------------------------
# 4. メイン処理 & UI
# ---------------------------------------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    image_size = getattr(model, "image_size", 256) if model else 256
    img_pil = ImageOps.fit(raw_img, (image_size, image_size), Image.Resampling.LANCZOS)

    file_id = uploaded_file.name + str(uploaded_file.size)

    if "dvlt_cache" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("未描画エリアを補完推論中..."):
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)

            angles = {
                "center": (0.0, 0.0),
                "left": (-1.0, 0.0),
                "right": (1.0, 0.0),
                "up": (0.0, 0.5),
                "down": (0.0, -0.5),
            }

            rendered_images = {"center": img_pil}

            for key, (pos_x, pos_y) in angles.items():
                if key == "center":
                    continue

                if model is not None:
                    pose_vector = torch.tensor([[pos_x, pos_y, 0.0]], device=device)
                    with torch.no_grad():
                        if hasattr(model, "forward"):
                            pred = model(img_tensor, pose=pose_vector)
                        else:
                            pred = model.render(img_tensor, pose=pose_vector)

                        out_tensor = pred.squeeze(0).cpu()
                        out_img_np = out_tensor.numpy().transpose(1, 2, 0)
                        out_img_np = (np.clip(out_img_np, 0.0, 1.0) * 255).astype(np.uint8)
                        rendered_images[key] = Image.fromarray(out_img_np)
                else:
                    rendered_images[key] = img_pil

            b64_cache = {}
            for k, img in rendered_images.items():
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_cache[k] = base64.b64encode(buf.getvalue()).decode()

            st.session_state.dvlt_cache = b64_cache
            st.session_state.current_key = "center"
            st.session_state.file_id = file_id
            gc.collect()

    # 十字キーUI
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

    curr_key = st.session_state.get("current_key", "center")
    curr_b64 = st.session_state.dvlt_cache[curr_key]

    st.image(f"data:image/png;base64,{curr_b64}", caption=f"視点: {curr_key.upper()}", use_container_width=True)
