import sys
import os
import gc
import base64
import io
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from huggingface_hub import snapshot_download

# ---------------------------------------------------------
# 1. nvidia/dvlt の自動ダウンロード＆読み込みパス設定
# ---------------------------------------------------------
DVLT_MODEL_DIR = os.path.abspath("./models/nvidia_dvlt")

if not os.path.exists(DVLT_MODEL_DIR):
    os.makedirs(DVLT_MODEL_DIR, exist_ok=True)
    try:
        # Hugging Face Hubからnvidia/dvltリポジトリ全体（コード・重み）をダウンロード
        snapshot_download(
            repo_id="nvidia/dvlt",
            local_dir=DVLT_MODEL_DIR,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        # フォールバック（nv-tlabs/dvlt GitHub）
        import urllib.request
        import zipfile
        zip_path = os.path.join(DVLT_MODEL_DIR, "dvlt.zip")
        url = "https://github.com/nv-tlabs/dvlt/archive/refs/heads/main.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DVLT_MODEL_DIR)

if DVLT_MODEL_DIR not in sys.path:
    sys.path.insert(0, DVLT_MODEL_DIR)

# ---------------------------------------------------------
# 2. ページ構成 & 不要UI非表示
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
# 3. DVLTモデルのロード
# ---------------------------------------------------------
@st.cache_resource
def load_dvlt_model():
    try:
        from models import DVLTModel
        model = DVLTModel.from_pretrained("nvidia/dvlt").to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"DVLTモデルの読み込みに失敗しました: {e}")
        st.stop()

try:
    with st.spinner("NVIDIA DVLTモデルをダウンロード・ロード中..."):
        model = load_dvlt_model()
except Exception as e:
    st.error(f"モデルロードエラー: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. メインパイプライン（1枚画像から未撮影エリアの補完・マルチビュー生成）
# ---------------------------------------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    image_size = getattr(model, "image_size", 256)
    img_pil = ImageOps.fit(raw_img, (image_size, image_size), Image.Resampling.LANCZOS)

    file_id = uploaded_file.name + str(uploaded_file.size)

    if "dvlt_cache" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("DVLTで写っていない背景・側面を推論・補完生成中..."):
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)

            # 視点パラメータ定義（上下左右のアングル移動）
            angles = {
                "center": (0.0, 0.0),
                "left": (-1.0, 0.0),
                "right": (1.0, 0.0),
                "up": (0.0, 0.5),
                "down": (0.0, -0.5),
            }

            b64_cache = {}
            rendered_images = {"center": img_pil}

            for key, (pos_x, pos_y) in angles.items():
                if key == "center":
                    continue

                # DVLTのポーズ・カメラエンコーディング指定
                pose_vector = torch.tensor([[pos_x, pos_y, 0.0]], device=device)

                with torch.no_grad():
                    if hasattr(model, "generate_view"):
                        pred = model.generate_view(img_tensor, pose_vector)
                    else:
                        pred = model(img_tensor, pose=pose_vector)

                    out_tensor = pred.squeeze(0).cpu()
                    out_img_np = out_tensor.numpy().transpose(1, 2, 0)
                    out_img_np = (np.clip(out_img_np, 0.0, 1.0) * 255).astype(np.uint8)
                    rendered_images[key] = Image.fromarray(out_img_np)

            # OSError対策のためバイト列にしてキャッシュ
            for k, img in rendered_images.items():
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_cache[k] = base64.b64encode(buf.getvalue()).decode()

            st.session_state.dvlt_cache = b64_cache
            st.session_state.current_key = "center"
            st.session_state.file_id = file_id
            gc.collect()

    # 十字キーボタン配置
    col_u1, col_u2, col_u3 = st.columns([1, 1, 1])
    with col_u2:
        if st.button("▲ 上アングル", use_container_width=True):
            st.session_state.current_key = "up"

    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    with col_m1:
        if st.button("◀ 左アングル", use_container_width=True):
            st.session_state.current_key = "left"
    with col_m2:
        if st.button("正面 (0)", use_container_width=True):
            st.session_state.current_key = "center"
    with col_m3:
        if st.button("右アングル ▶", use_container_width=True):
            st.session_state.current_key = "right"

    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    with col_d2:
        if st.button("▼ 下アングル", use_container_width=True):
            st.session_state.current_key = "down"

    st.markdown("---")

    # 画面描画
    curr_key = st.session_state.get("current_key", "center")
    curr_b64 = st.session_state.dvlt_cache[curr_key]

    st.image(f"data:image/png;base64,{curr_b64}", caption=f"視点: {curr_key.upper()}", use_container_width=True)
