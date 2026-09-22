import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
import plotly.graph_objects as go

# --- ページ設定 ---
st.set_page_config(page_title="2.5D Scene Generator", layout="wide")
st.title("⚡ 爆速版：一枚の画像から2.5D空間を作るAI")

# --- 爆速・超軽量モデルのロード ---
@st.cache_resource
def load_depth_pipeline():
    # Depth Anything の小型軽量モデル(約90MB)を使用！精度が良くて高速
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=-1)
    return pipe

with st.spinner("軽量モデルを準備中...（初回のみダウンロードが入ります）"):
    depth_pipe = load_depth_pipeline()

# --- サイドバー ---
st.sidebar.header("設定")
max_size = st.sidebar.slider("処理サイズ（小さいほど爆速）", 128, 384, 256, step=64)
depth_scale = st.sidebar.slider("奥行きの強さ", 0.1, 2.0, 0.4, step=0.1)
downsample = st.sidebar.slider("3Dメッシュの間引き", 1, 6, 2)

# --- 画像アップロード ---
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 爆速化のために画像をリサイズ
    img_resized = image.resize((max_size, max_size))
    img_np = np.array(img_resized)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像")
        st.image(img_resized, use_container_width=True)

    # 深度推定の実行（一瞬で終わる）
    with st.spinner("奥行きを推測中..."):
        result = depth_pipe(img_resized)
        
        # PIL Image か Numpy 配列で返ってくるのを処理
        raw_depth = result["depth"]
        if isinstance(raw_depth, Image.Image):
            depth_map = np.array(raw_depth).astype(np.float32)
        else:
            depth_map = np.array(raw_depth, dtype=np.float32)
        
        # 0〜1に正規化
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)

    with col2:
        st.subheader("深度マップ")
        st.image(depth_map, use_container_width=True)

    # --- 3D描画処理の爆速化 (Plotly Scatter3d を使用) ---
    st.subheader("🎮 2.5D 空間シーン")

    with st.spinner("3Dシーンを構築中..."):
        # 間引き処理
        h, w = depth_map.shape
        y_indices = np.arange(0, h, downsample)
        x_indices = np.arange(0, w, downsample)
        
        x_grid, y_grid = np.meshgrid(x_indices, y_indices)
        
        # 配列を1次元にフラット化して計算を高速化
        x_flat = x_grid.flatten()
        y_flat = (h - y_grid).flatten() # Y軸を上下反転
        z_flat = depth_map[y_grid, x_grid].flatten() * depth_scale * max_size
        
        # ピクセルカラーの抽出
        colors_flat = img_np[y_grid, x_grid].reshape(-1, 3)
        color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors_flat]

        # Scatter3d を使ってポイントクラウドとして描画（Mesh3dより圧倒的に描画が早い）
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                mode='markers',
                marker=dict(
                    size=downsample + 1,
                    color=color_strings,
                    opacity=1.0
                )
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectratio=dict(x=1, y=h/w, z=depth_scale),
                camera=dict(
                    eye=dict(x=0, y=-0.2, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success("爆速完了！")
