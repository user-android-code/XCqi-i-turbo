import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

st.set_page_config(page_title="2D to 3D Spatial Scene", layout="wide")
st.title("🖼️️ Hugging Faceモデルで作る3D空間シーン")

# 1. Hugging Faceから軽量・高品質モデル（Depth Anything V2 Small: 約98MB）をロード
@st.cache_resource
def load_hf_depth_model():
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    return processor, model

with st.spinner("Hugging Faceからモデル（約98MB）を読み込み中..."):
    processor, model = load_hf_depth_model()

# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてね", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 画像読み込み ＆ メモリ節約用に最大512pxにリサイズ
    input_image = Image.open(uploaded_file).convert("RGB")
    input_image.thumbnail((512, 512))
    img_np = np.array(input_image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="元画像", use_container_width=True)

    # サイドバーで伸縮・膨らみ調整
    st.sidebar.header("3D変形・伸縮パラメータ")
    depth_scale = st.sidebar.slider("膨らみ具合（Z軸の深さ）", 0.0, 2.0, 0.5, 0.05)
    stretch_x = st.sidebar.slider("横方向の伸縮（X軸）", 0.5, 3.0, 1.0, 0.1)
    stretch_y = st.sidebar.slider("縦方向の伸縮（Y軸）", 0.5, 3.0, 1.0, 0.1)
    
    # 2. 深度推論
    with st.spinner("深度（Depth）を推論中..."):
        inputs = processor(images=input_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 元画像サイズに補間
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        # 0 ~ 1 に正規化（手前を1、奥を0にする）
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    with col2:
        st.image(depth_map, caption="Hugging Faceモデルで推論した深度マップ", use_container_width=True)

    # 3. 3D空間への配置と膨らまし（メッシュ化）
    h, w, _ = img_np.shape
    x = np.linspace(-1 * stretch_x, 1 * stretch_x, w)
    y = np.linspace(1 * stretch_y, -1 * stretch_y, h) # Y軸反転
    grid_x, grid_y = np.meshgrid(x, y)
    
    # 深度マップでZ軸（奥行き）を膨らませる
    grid_z = depth_map * depth_scale

    # 描画軽量化のためにサンプリング（2ピクセルごと）
    step = 2
    
    # 4. Plotly 3Dで可視化（回転・伸縮・自由な視点変更が可能）
    fig = go.Figure(data=[
        go.Surface(
            x=grid_x[::step, ::step],
            y=grid_y[::step, ::step],
            z=grid_z[::step, ::step],
            surfacecolor=np.mean(img_np[::step, ::step], axis=2),
            colorscale='Viridis',
            showscale=False
        )
    ])

    fig.update_layout(
        title="3D空間シーン（ドラッグで横から見て膨らみを確認してね）",
        autosize=True,
        scene=dict(
            xaxis=dict(title="X (横)"),
            yaxis=dict(title="Y (縦)"),
            zaxis=dict(title="Z (膨らみ/奥行き)"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)
