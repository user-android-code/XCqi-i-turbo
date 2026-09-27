import gc
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torchvision.transforms as T
from huggingface_hub import PyTorchModelHubMixin

st.set_page_config(
    page_title="2.5D Spatial Scene Generator",
    layout="centered"
)

st.title("2.5D Spatial Scene Generator (OVIE)")
st.caption("1枚の2D画像(256x256)から新しい視点の空間シーンを生成するよ")

MODEL_ID = "kyutai/ovie"

# 前処理
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# PyTorchModelHubMixin 経由でクラスを動的定義してロード
class DynamicOVIEModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()

    # 万が一 forward が未定義でロードされた場合のフォールバック
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

@st.cache_resource
def load_ovie_model():
    # huggingface_hub の Mixin を使って直接ロード
    model = DynamicOVIEModel.from_pretrained(MODEL_ID)
    model.eval()
    return model

try:
    with st.spinner("モデルをロード中..."):
        model = load_ovie_model()
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
                img_tensor = transform(img_256).unsqueeze(0)
                camera_pose = torch.tensor([[yaw, pitch]], dtype=torch.float32)
                
                with torch.no_grad():
                    # 推論実行
                    outputs = model(img_tensor, camera_pose)
                    
                    if hasattr(outputs, "logits"):
                        output_tensor = outputs.logits
                    elif isinstance(outputs, torch.Tensor):
                        output_tensor = outputs
                    else:
                        output_tensor = outputs[0]
                    
                    out_img_np = output_tensor.squeeze(0).cpu().numpy()
                    if out_img_np.shape[0] in [1, 3]:
                        out_img_np = np.transpose(out_img_np, (1, 2, 0))
                    
                    if out_img_np.min() < 0:
                        out_img_np = (out_img_np + 1.0) / 2.0
                    if out_img_np.max() <= 1.0:
                        out_img_np = (out_img_np * 255).clip(0, 255).astype(np.uint8)
                    else:
                        out_img_np = out_img_np.clip(0, 255).astype(np.uint8)
                    
                    generated_img = Image.fromarray(out_img_np)

                with col2:
                    st.subheader("生成された2.5D視点")
                    st.image(generated_img, use_container_width=True)
                
                st.success("生成完了！")

            except Exception as e:
                st.error(f"推論中にエラーが発生したよ: {e}")

            finally:
                gc.collect()
