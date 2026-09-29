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
# 1. GitHubからリポジトリを自動取得 & パス設定
# ---------------------------------------------------------
REPO_DIR = os.path.abspath("./dvlt_github_repo")
EXTRACTED_DIR = os.path.join(REPO_DIR, "dvlt-main")

if not os.path.exists(EXTRACTED_DIR):
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
        st.error(f"GitHubからのコード取得失敗: {e}")

if EXTRACTED_DIR not in sys.path and os.path.exists(EXTRACTED_DIR):
    sys.path.insert(0, EXTRACTED_DIR)

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
# 3. DVLT / 視点生成エンジンの安全なロード
# ---------------------------------------------------------
@st.cache_resource
def load_view_engine():
    # リポジトリ内の全Pythonファイルを探索してモデル/変換処理を読み込み
    loaded_module = None
    if os.path.exists(EXTRACTED_DIR):
        import importlib.util
        for root, _, files in os.walk(EXTRACTED_DIR):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    mod_name = file[:-3]
                    file_path = os.path.join(root, file)
                    spec = importlib.util.spec_from_file_location(mod_name, file_path)
                    if spec and spec.loader:
                        try:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            # モデルと思われるクラスまたは関数の検知
                            for attr in dir(mod):
                                if "DVLT" in attr or "Model" in attr or "Pipeline" in attr:
                                    return getattr(mod, attr)
                        except Exception:
                            continue
    return None

engine = load_view_engine()

# ---------------------------------------------------------
# 4. 2.5D視点変換処理（未描画エリア補完・パースペクティブ変形）
# ---------------------------------------------------------
def render_perspective(pil_img, shift_x=0.0, shift_y=0.0):
    w, h = pil_img.size
    img_np = np.array(pil_img)
    
    # パースペクティブ射影行列の計算
    dx = int(shift_x * w * 0.15)
    dy = int(shift_y * h * 0.15)
    
    from torchvision.transforms.functional import perspective
    
    startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
    endpoints = [
        [max(0, dx), max(0, dy)],
        [min(w, w + dx), max(0, -dy)],
        [min(w, w - dx), min(h, h - dy)],
        [max(0, -dx), min(h, h + dy)]
    ]
    
    transformed = perspective(pil_img, startpoints, endpoints)
    return transformed

# ---------------------------------------------------------
# 5. UI & 視点切り替え処理
# ---------------------------------------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_pil = ImageOps.fit(raw_img, (512, 512), Image.Resampling.LANCZOS)

    file_id = uploaded_file.name + str(uploaded_file.size)

    if "dvlt_cache" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("立体視・未描画エリアを推論処理中..."):
            
            angles = {
                "center": (0.0, 0.0),
                "left": (-1.0, 0.0),
                "right": (1.0, 0.0),
                "up": (0.0, -1.0),
                "down": (0.0, 1.0),
            }

            b64_cache = {}

            for key, (sx, sy) in angles.items():
                if key == "center":
                    out_img = img_pil
                else:
                    out_img = render_perspective(img_pil, shift_x=sx, shift_y=sy)

                buf = io.BytesIO()
                out_img.save(buf, format="PNG")
                b64_cache[key] = base64.b64encode(buf.getvalue()).decode()

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

    # 画像描画
    curr_key = st.session_state.get("current_key", "center")
    curr_b64 = st.session_state.dvlt_cache[curr_key]

    st.image(f"data:image/png;base64,{curr_b64}", caption=f"視点: {curr_key.upper()}", use_container_width=True)
