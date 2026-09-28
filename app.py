import gc
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ---------------------------------------------------------
# ページ設定（サイドバー非表示）
# ---------------------------------------------------------
st.set_page_config(
    page_title="Xcqi i-air",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# サイドバーおよび不要なUI要素の完全非表示CSS
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("Xcqi i-air")

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ---------------------------------------------------------
# モデルロード
# ---------------------------------------------------------
@st.cache_resource
def load_depth_model():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.eval()
    return image_processor, model

try:
    image_processor, model = load_depth_model()
except Exception as e:
    st.error(f"モデルのロードに失敗しました: {e}")
    st.stop()

# ---------------------------------------------------------
# 視点合成関数（ワープ処理）
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 画像アップロード & セッション状態管理
# ---------------------------------------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    # 新しい画像がアップロードされた場合、状態を初期化
    file_id = uploaded_file.name + str(uploaded_file.size)
    if "current_file" not in st.session_state or st.session_state.current_file != file_id:
        st.session_state.current_file = file_id
        st.session_state.pos_x = 0
        st.session_state.pos_y = 0
        st.session_state.generated_cache = {}

    raw_img = Image.open(uploaded_file).convert("RGB")
    img_256 = ImageOps.fit(raw_img, (256, 256), Image.Resampling.LANCZOS)

    # ---------------------------------------------------------
    # 計8枚（上下左右の50/100）を事前生成
    # ---------------------------------------------------------
    if "pregenerated" not in st.session_state.generated_cache:
        with st.spinner("画像を生成中..."):
            # 1. 深度推定
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

            depth_norm = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)

            # 2. 0, 上下左右の50, 100 計9パターンを作成してキャッシュ
            cache = {}
            cache[(0, 0)] = img_256
            
            # 各方向のパターン (shift_x, shift_y)
            patterns = {
                ("up", 50): (0, -10),
                ("up", 100): (0, -20),
                ("down", 50): (0, 10),
                ("down", 100): (0, 20),
                ("left", 50): (-10, 0),
                ("left", 100): (-20, 0),
                ("right", 50): (10, 0),
                ("right", 100): (20, 0),
            }

            for key, (sx, sy) in patterns.items():
                cache[key] = generate_warped_image(img_256, depth_norm, sx, sy)

            st.session_state.generated_cache = cache
            st.session_state.generated_cache["pregenerated"] = True
            gc.collect()

    # ---------------------------------------------------------
    # ボタン操作ロジック
    # ---------------------------------------------------------
    curr_x = st.session_state.pos_x
    curr_y = st.session_state.pos_y

    # 各ボタンの有効/無効判定（100以上は押せない）
    can_up = curr_y < 100 and curr_x == 0
    can_down = curr_y > -100 and curr_x == 0
    can_left = curr_x > -100 and curr_y == 0
    can_right = curr_x < 100 and curr_y == 0

    # ---------------------------------------------------------
    # 十字型ボタンコントローラー
    # ---------------------------------------------------------
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    
    with col_ctrl2:
        if st.button("▲ 上", disabled=not can_up, use_container_width=True):
            st.session_state.pos_y += 50
            st.rerun()

    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
    with col_b1:
        if st.button("◀ 左", disabled=not can_left, use_container_width=True):
            st.session_state.pos_x -= 50
            st.rerun()
    with col_b2:
        if st.button("リセット (0)", use_container_width=True):
            st.session_state.pos_x = 0
            st.session_state.pos_y = 0
            st.rerun()
    with col_b3:
        if st.button("右 ▶", disabled=not can_right, use_container_width=True):
            st.session_state.pos_x += 50
            st.rerun()

    col_ctrl4, col_ctrl5, col_ctrl6 = st.columns([1, 1, 1])
    with col_ctrl5:
        if st.button("▼ 下", disabled=not can_down, use_container_width=True):
            st.session_state.pos_y -= 50
            st.rerun()

    # ---------------------------------------------------------
    # 表示する画像の特定
    # ---------------------------------------------------------
    cache = st.session_state.generated_cache
    display_img = img_256

    if curr_x == 0 and curr_y == 0:
        display_img = cache[(0, 0)]
    elif curr_y > 0:
        display_img = cache[("up", curr_y)]
    elif curr_y < 0:
        display_img = cache[("down", abs(curr_y))]
    elif curr_x < 0:
        display_img = cache[("left", abs(curr_x))]
    elif curr_x > 0:
        display_img = cache[("right", curr_x)]

    # ステータス表示テキストの生成
    if curr_x == 0 and curr_y == 0:
        status_text = "位置: 0 (元画像)"
    elif curr_y != 0:
        direction = "上" if curr_y > 0 else "下"
        status_text = f"位置: {direction} {abs(curr_y)}"
    else:
        direction = "右" if curr_x > 0 else "左"
        status_text = f"位置: {direction} {abs(curr_x)}"

    # 画像表示
    st.image(display_img, caption=status_text, use_container_width=True)
