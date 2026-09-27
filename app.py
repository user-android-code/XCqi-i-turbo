import streamlit as st
import torch
from PIL import Image
from PIL import ImageOps
import gc

# 1. モデル読み込み（軽量化なし、そのままの精度でロード）
@st.cache_resource
def load_model():
    # Kyutai/OVIEのモデルロード処理（PyTorch / Transformers）
    # ※ 公式の指定クラスに合わせてインポート・呼び出し
    model = torch.hub.load(...)  # または AutoModel.from_pretrained("kyutai/ovie")
    model.eval()
    return model

st.title("2.5D Spatial Scene Generator (OVIE)")

model = load_model()

uploaded_file = st.file_uploader("画像をアップロード（自動で256x256に調整されるよ）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # 256x256 にアスペクト比を保ちつつ中央クロップ＆リサイズ
    img_256 = ImageOps.fit(raw_img, (256, 256), Image.Resampling.LANCZOS)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_256, caption="入力画像 (256x256)", use_column_width=True)
    
    # カメラパラメータの操作
    st.subheader("空間アングル設定")
    yaw = st.slider("左右アングル (Yaw)", -20.0, 20.0, 0.0, step=1.0)
    pitch = st.slider("上下アングル (Pitch)", -10.0, 10.0, 0.0, step=1.0)
    
    if st.button("2.5D視点を生成"):
        with st.spinner("OVIEで推論中..."):
            # OVIEに 256x256 画像とカメラポーズを入力
            # output_tensor = model(img_256, yaw=yaw, pitch=pitch)
            # output_img = tensor_to_pil(output_tensor)
            
            with col2:
                # st.image(output_img, caption="生成された空間視点", use_column_width=True)
                st.success("できた！")
