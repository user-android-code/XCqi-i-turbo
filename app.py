import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"

import streamlit as st
import numpy as np
from PIL import Image
from transformers import pipeline
import plotly.graph_objects as go

st.set_page_config(page_title="2.5D Scene Generator", layout="wide")
st.title("⚡ 2.5D 空間シーンジェネレーター")

# --- Hugging Face から画像AI（深度推定）を1行でロード ---
@st.cache_resource
def load_ai():
    # Hugging Face上の軽量な深度推定AIモデルを読み込む
    return pipeline(task="depth-estimation", model="onnx-community/depth-anything-v2-small")

try:
    with st.spinner("AIモデルを読み込み中..."):
        depth_ai = load_ai()
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")

# --- 画面構成 ---
st.sidebar.header("設定")
depth_scale = st.sidebar.slider("奥行きの強さ", 0.1, 2.0, 0.5, step=0.1)
downsample = st.sidebar.slider("3D点群の間引き", 1, 5, 2)

uploaded_file = st.file_uploader("画像をアップロードしてね", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. 画像の読み込みとリサイズ
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    img_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像")
        st.image(image, use_container_width=True)

    # 2. AIで奥行き（深度）を推定
    with st.spinner("奥行きを推定中..."):
        result = depth_ai(image)
        depth_map = np.array(result["depth"]).astype(np.float32)
        # 0〜1に正規化
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    with col2:
        st.subheader("奥行きデータ（深度マップ）")
        st.image(depth_map, use_container_width=True)

    # 3. 3D描画
    st.subheader("🎮 2.5D 空間シーン")
    
    h, w = depth_map.shape
    y_grid, x_grid = np.meshgrid(np.arange(0, h, downsample), np.arange(0, w, downsample), indexing='ij')

    x_flat = x_grid.flatten()
    y_flat = (h - y_grid).flatten() # 上下反転
    z_flat = depth_map[y_grid, x_grid].flatten() * depth_scale * 256

    colors_flat = img_np[y_grid, x_grid].reshape(-1, 3)
    color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors_flat]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_flat, y=y_flat, z=z_flat,
            mode='markers',
            marker=dict(size=downsample + 1, color=color_strings, opacity=1.0)
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=h/w, z=depth_scale),
            camera=dict(eye=dict(x=0, y=-0.2, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success("完成！")
