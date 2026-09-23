import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("🖼️ Depth Anything V2 深度推定")

# 1. モデルをロード（たったの99MB！）
@st.cache_resource
def load_model():
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

pipe = load_model()

# 2. 画像アップロード
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像")
        st.image(image, use_container_width=True)

    # 3. AIに渡して、返ってきた画像を表示するだけ！
    with st.spinner("解析中..."):
        result = pipe(image)
        depth_image = result["depth"] # 返ってきた奥行き画像

    with col2:
        st.subheader("返ってきた深度画像")
        st.image(depth_image, use_container_width=True)
