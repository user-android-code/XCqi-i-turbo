import streamlit as st
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# --- ページ設定 ---
st.set_page_config(page_title="Image to 2.5D Scene", layout="wide")
st.title("🖼️ 一枚の画像から2.5D空間を作るAI")
st.write("画像をアップロードすると、深度（奥行き）を推定して3D空間を生成するよ！")

# --- 深度推定モデル（MiDaS Small: 超軽量）のロード ---
@st.cache_resource
def load_midas():
    # 1GB制限を考慮して一番軽量な MiDaS_small を使用
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    midas.eval()

    # 専用の前処理トランスフォームを取得
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return midas, transform

with st.spinner("AIモデルを読み込み中..."):
    midas, transform = load_midas()

# --- サイドバー設定 ---
st.sidebar.header("設定パラメーター")
max_size = st.sidebar.slider("処理解像度（小さくすると高速化）", 128, 512, 256, step=64)
depth_scale = st.sidebar.slider("奥行きの強さ（Z軸スケール）", 0.1, 2.0, 0.5, step=0.1)
downsample = st.sidebar.slider("3D表示の間引き（大きくすると滑らか動作）", 1, 5, 2)

# --- 画像アップロード ---
uploaded_file = st.file_uploader("画像をドラッグ＆ドロップまたは選択してね", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込み
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2列レイアウトで元画像と処理結果を表示
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像")
        st.image(image, use_container_width=True)

    # 処理用のサイズにリサイズ
    img_resized = image.resize((max_size, max_size))
    img_np = np.array(img_resized)

    # 深度推定の実行
    with st.spinner("深度（奥行き）を計算中..."):
        input_batch = transform(img_np)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        # 0〜1に正規化
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    with col2:
        st.subheader("推測された深度マップ（ヒートマップ）")
        st.image(depth_map, clamp=True, use_container_width=True)

    # --- 2.5D 3Dメッシュの作成 ---
    st.subheader("🎮 2.5D 空間シーン (ドラッグして3D操作してね)")

    # 描画用のメッシュグリッド作成 (間引き処理)
    h, w = depth_map.shape
    x = np.arange(0, w, downsample)
    y = np.arange(0, h, downsample)
    x_grid, y_grid = np.meshgrid(x, y)

    # 深度データの抽出（Y軸は画像の上下と反転させる）
    z_data = depth_map[::downsample, ::downsample] * depth_scale * max_size
    y_grid_flipped = h - y_grid 

    # 画像の各ピクセル色をPlotly用に変換
    colors = img_np[::downsample, ::downsample]
    # RGBの16進数カラーコード文字列に変換
    color_strings = [
        [f"rgb({r},{g},{b})" for r, g, b in row]
        for row in colors
    ]

    # Plotlyで3Dサーフェスを描画
    fig = go.Figure(data=[
        go.Surface(
            x=x_grid,
            y=y_grid_flipped,
            z=z_data,
            surfacecolor=np.zeros_like(z_data), # ダミー
            customdata=color_strings,
            colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']], # カラー処理用
            showscale=False,
            # テクスチャっぽく見せるために表面色をRGBでマッピング
            lighting=dict(ambient=0.9, diffuse=0.8, roughness=0.5),
        )
    ])

    # 表面のグラデーションではなく画像そのものの色を反映させるカスタム記述
    # 簡易的に表面テクスチャっぽく見せるため視界調整
    fig.update_traces(
        surfacecolor=np.mean(colors, axis=2),
        colorscale='Viridis', # フォールバック用
        showscale=False
    )

    # レイアウトの調整
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=h/w, z=depth_scale),
            camera=dict(
                eye=dict(x=0, y=-0.1, z=1.5) # 初期視点（正面やや上から）
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success("作成完了！マウスでドラッグして好きな角度から見てみてね。")
