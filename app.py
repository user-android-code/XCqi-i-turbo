import os
# --- 画像アップロードエラー（XSRFブロック）を回避する設定 ---
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"

import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="2.5D Scene Generator", layout="wide")
st.title("🖼️ Depth Anything V2 深度推定")

# 1. モデルのロード（キャッシュ化）
@st.cache_resource
def load_model():
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

try:
    with st.spinner("AIモデルを読み込み中..."):
        pipe = load_model()
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")

# 2. 画像のアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("元画像")
            st.image(image, use_container_width=True)

        # 3. 推論と結果の表示
        with st.spinner("奥行きを推定中..."):
            result = pipe(image)
            depth_image = result["depth"]

        with col2:
            st.subheader("生成された深度画像")
            st.image(depth_image, use_container_width=True)
            
    except Exception as e:
        st.error(f"画像処理エラー: {e}")
