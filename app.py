import gc
import json
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

st.set_page_config(
    page_title="2.5D Spatial Scene Generator",
    layout="centered"
)

st.title("2.5D Spatial Scene Generator (OVIE)")
st.caption("1枚の2D画像(256x256)から新しい視点の空間シーンを生成するよ")

REPO_ID = "kyutai/ovie"

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@st.cache_resource
def load_ovie_weights():
    # 1. config.json と model.safetensors をダウンロード
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # 2. safetensorsから重みテンソルを直接読み込み
    state_dict = load_file(weights_path)
    return config, state_dict

try:
    with st.spinner("モデルと重みをロード中..."):
        config, state_dict = load_ovie_weights()
    st.sidebar.success("ロード成功！")
except Exception as e:
    st.sidebar.error(f"ロードに失敗したよ: {e}")
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
                
                # 簡易的な重み行列の乗算演算（ダミー推論・テンソル合成のテスト処理）
                # ※ 実際のレイヤーが組み上がっていなくてもテンソル計算を通して描画確認するロジック
                with torch.no_grad():
                    # 入力テンソルの次元合わせ
                    out_tensor = img_tensor.clone()
                    
                    # カメラポーズに応じた簡単な視点ずらし効果（テスト用アルゴリズム）
                    shift_x = int(yaw * 0.5)
                    shift_y = int(pitch * 0.5)
                    out_tensor = torch.roll(out_tensor, shifts=(shift_y, shift_x), dims=(2, 3))

                    # 後処理
                    out_img_np = out_tensor.squeeze(0).cpu().numpy()
                    out_img_np = np.transpose(out_img_np, (1, 2, 0))
                    out_img_np = ((out_img_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                    
                    generated_img = Image.fromarray(out_img_np)

                with col2:
                    st.subheader("生成された2.5D視点")
                    st.image(generated_img, use_container_width=True)
                
                st.success("生成完了！")

            except Exception as e:
                st.error(f"推論中にエラーが発生したよ: {e}")

            finally:
                gc.collect()
