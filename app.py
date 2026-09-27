import gc
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from diffusers import DiffusionPipeline

st.set_page_config(
    page_title="2.5D Spatial Scene Generator",
    layout="centered"
)

st.title("2.5D Spatial Scene Generator (OVIE)")
st.caption("1枚の2D画像(256x256)から新しい視点の空間シーンを生成するよ")

MODEL_ID = "kyutai/ovie"

@st.cache_resource
def load_ovie_pipeline():
    # Diffusersのパイプラインとしてモデル全体をロード
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    return pipe

try:
    with st.spinner("モデルをロード中..."):
        pipe = load_ovie_pipeline()
    st.sidebar.success("モデルのロード完了！")
except Exception as e:
    st.sidebar.error(f"モデルのロードに失敗したよ: {e}")
    st.stop()

uploaded_file = st.file_uploader("2D画像をアップロードしてね", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_256 = ImageOps.fit(raw_img, (256, 256), Image.Resampling.LANCZOS)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("入力画像 (256x256)")
        st.image(img_256, use_container_width=True)

    st.markdown("---")
    st.subheader("カメラアングル調整")
    
    col_yaw, col_pitch = st.columns(2)
    with col_yaw:
        yaw = st.slider("左右アングル (Yaw)", -20.0, 20.0, 0.0, step=1.0)
    with col_pitch:
        pitch = st.slider("上下アングル (Pitch)", -10.0, 10.0, 0.0, step=1.0)

    if st.button("2.5D視点を生成！", type="primary"):
        with st.spinner("新しい視点を生成中..."):
            try:
                camera_pose = [yaw, pitch]
                
                with torch.no_grad():
                    # Diffusers パイプライン経由での推論
                    # (モデルの仕様に応じて image / pose / camera_pose 等の引数を渡す)
                    result = pipe(
                        image=img_256,
                        pose=camera_pose
                    )
                    
                    # 出力画像の取り出し
                    if hasattr(result, "images"):
                        generated_img = result.images[0]
                    elif isinstance(result, list):
                        generated_img = result[0]
                    else:
                        generated_img = result

                with col2:
                    st.subheader("生成された2.5D視点")
                    st.image(generated_img, use_container_width=True)
                
                st.success("生成完了！")

            except Exception as e:
                st.error(f"推論中にエラーが発生したよ: {e}")

            finally:
                gc.collect()
