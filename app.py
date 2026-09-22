import os
# --- config.toml を作らずにPython側でアップロード制限と保護を強制解除する設定 ---
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"

import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="2.5D Scene Generator", layout="wide")
st.title("⚡ 2.5D 空間シーンジェネレーター")

# --- クラウド用に最軽量MiDaSモデルを使用 ---
@st.cache_resource
def load_midas():
    # クラウドメモリ対策：MiDaS_small (約40MB) を使用
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    model.eval()
    
    transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transforms

try:
    with st.spinner("AIモデルを読み込み中..."):
        model, transform = load_midas()
except Exception as e:
    st.error(f"モデルの読み込みエラー: {e}")

# --- サイドバー ---
st.sidebar.header("設定")
depth_scale = st.sidebar.slider("奥行きの強さ", 0.1, 2.0, 0.5, step=0.1)
downsample = st.sidebar.slider("3D点群の間引き", 1, 5, 2)

# --- 画像アップロード ---
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # 処理用にリサイズ
        img_resized = image.resize((256, 256))
        img_np = np.array(img_resized)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("元画像")
            st.image(img_resized, use_container_width=True)

        # 深度推論
        with st.spinner("奥行きを推定中..."):
            input_tensor = transform(img_resized).unsqueeze(0)
            with torch.no_grad():
                prediction = model(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(256, 256),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        with col2:
            st.subheader("深度マップ")
            st.image(depth_map, use_container_width=True)

        # 3D描画
        st.subheader("🎮 2.5D 空間")

        h, w = depth_map.shape
        y_indices = np.arange(0, h, downsample)
        x_indices = np.arange(0, w, downsample)
        x_grid, y_grid = np.meshgrid(x_indices, y_indices)

        x_flat = x_grid.flatten()
        y_flat = (h - y_grid).flatten()
        z_flat = depth_map[y_grid, x_grid].flatten() * depth_scale * 256

        colors_flat = img_np[y_grid, x_grid].reshape(-1, 3)
        color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors_flat]

        fig = go.Figure(data=[
            go.Scatter3d(
                x=x_flat,
                y=y_flat,
                z=z_flat,
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
        st.success("作成成功！")

    except Exception as e:
        st.error(f"画像処理中にエラーが発生しました: {e}")
