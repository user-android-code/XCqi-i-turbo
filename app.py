import gc
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

st.set_page_config(
    page_title="2.5D Spatial Scene Generator",
    layout="centered"
)

st.title("2.5D Spatial Scene Generator (Depth V2)")
st.caption("1枚の2D画像から深度を推定し、2.5D空間視点を合成します")

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

@st.cache_resource
def load_depth_model():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.eval()
    return image_processor, model

try:
    with st.spinner("軽量深度モデルをロード中..."):
        image_processor, model = load_depth_model()
    st.sidebar.success("モデルのロード完了！")
except Exception as e:
    st.sidebar.error(f"モデルのロードに失敗しました: {e}")
    st.stop()

uploaded_file = st.file_uploader("2D画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_256 = ImageOps.fit(raw_img, (256, 256), Image.Resampling.LANCZOS)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("入力画像 (256x256)")
        st.image(img_256, use_container_width=True)

    st.markdown("---")
    st.subheader("カメラアングル調整")
    
    col_x, col_y = st.columns(2)
    with col_x:
        shift_x = st.slider("左右アングル (Yaw)", -15, 15, 0, step=1)
    with col_y:
        shift_y = st.slider("上下アングル (Pitch)", -15, 15, 0, step=1)

    if st.button("2.5D視点を生成！", type="primary"):
        with st.spinner("視点合成中..."):
            try:
                # 1. 深度推定
                inputs = image_processor(images=img_256, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth

                # 2. 深度マップのサイズ調整
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(256, 256),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

                # 深度の正規化 (0.0 ~ 1.0)
                depth_norm = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
                
                # 3. 視差マップに基づくメッシュ再投影 (2.5D ワープ処理)
                img_np = np.array(img_256)
                h, w, c = img_np.shape
                
                grid_y, grid_x = np.mgrid[0:h, 0:w]
                
                # 深度に応じたピクセル移動量の計算
                offset_x = (grid_x + shift_x * depth_norm).astype(np.float32)
                offset_y = (grid_y + shift_y * depth_norm).astype(np.float32)
                
                offset_x = np.clip(offset_x, 0, w - 1).astype(np.int32)
                offset_y = np.clip(offset_y, 0, h - 1).astype(np.int32)
                
                # ワープ画像の生成
                warped_img = img_np[offset_y, offset_x]
                generated_img = Image.fromarray(warped_img)

                with col2:
                    st.subheader("生成された2.5D視点")
                    st.image(generated_img, use_container_width=True)
                
                st.success("生成完了！")

            except Exception as e:
                st.error(f"推論中にエラーが発生しました: {e}")

            finally:
                gc.collect()
