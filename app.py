import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# タブ名を「XCqi」に設定
st.set_page_config(page_title="XCqi", layout="wide")

# タイトルを「XCqi-i-turbo」に設定
st.title("XCqi-i-turbo")

# 1. Hugging Faceから軽量モデル（Depth Anything V2 Small: 約98MB）をロード
@st.cache_resource
def load_hf_depth_model():
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    return processor, model

processor, model = load_hf_depth_model()

# メイン画面上に直接ファイル選択枠を配置（ラベルは完全非表示）
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    # 画像読み込み ＆ 処理用にリサイズ
    input_image = Image.open(uploaded_file).convert("RGB")
    input_image.thumbnail((512, 512))
    img_np = np.array(input_image)

    # 深度推論
    inputs = processor(images=input_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img_np.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    # 3D空間のメッシュ構築（デフォルトの膨らみ・伸縮率で固定）
    h, w, _ = img_np.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(1, -1, h)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = depth_map * 0.5  # 膨らみの固定値

    step = 2  # 描画高速化用の間引き

    # 3D空間シーンのみを描画
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
        autosize=True,
        height=750,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    st.plotly_chart(fig, use_container_width=True)
