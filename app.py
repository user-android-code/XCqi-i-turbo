import sys
import gc
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
from huggingface_hub import snapshot_download

st.set_page_config(
    page_title="2.5D Spatial Scene Generator",
    layout="centered"
)

st.title("2.5D Spatial Scene Generator (OVIE)")
st.caption("1枚の2D画像(256x256)から新しい視点の空間シーンを生成するよ")

MODEL_ID = "kyutai/ovie"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_ovie_all():
    # リポジトリ内の全ファイル（models/ や utils/ のPythonコード含む）を一気にダウンロード
    repo_dir = snapshot_download(repo_id=MODEL_ID, revision="v1.0")
    
    # Pythonの検索パスにダウンロードしたディレクトリを追加して import できるようにする
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        
    # 公式コードのモジュールを動的インポート
    from models.models import OVIEModel
    from utils.pose_enc import extri_intri_to_pose_encoding
    
    # モデルのロード
    model = OVIEModel.from_pretrained(MODEL_ID, revision="v1.0").to(device)
    model.eval()
    
    return model, extri_intri_to_pose_encoding

try:
    with st.spinner("Hugging Faceからコードとモデルを全自動ロード中..."):
        model, extri_intri_to_pose_encoding = load_ovie_all()
    st.sidebar.success("モデル＆スクリプトのロード完了！")
except Exception as e:
    st.sidebar.error(f"ロードに失敗したよ: {e}")
    st.stop()

uploaded_file = st.file_uploader("2D画像をアップロードしてね", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    image_size = getattr(model, "image_size", 256)
    
    # 256x256 に調整
    img_pil = ImageOps.fit(raw_img, (image_size, image_size), Image.Resampling.LANCZOS)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("入力画像 (256x256)")
        st.image(img_pil, use_container_width=True)

    st.markdown("---")
    st.subheader("カメラアングル調整")
    
    col_x, col_y = st.columns(2)
    with col_x:
        pos_x = st.slider("左右平行移動 (X軸)", -2.0, 2.0, -1.25, step=0.05)
    with col_y:
        pos_y = st.slider("上下平行移動 (Y軸)", -2.0, 2.0, 0.5, step=0.05)

    if st.button("2.5D視点を生成！", type="primary"):
        with st.spinner("公式OVIEパイプラインで生成中..."):
            try:
                # 1. 画像のテンソル化 [1, 3, 256, 256]
                img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)
                
                # 2. カメラパラメータ（3x4 外部パラメータ行列）の構築
                extrinsics = torch.tensor([[[1.0, 0.0, 0.0, pos_x],
                                            [0.0, 1.0, 0.0, pos_y],
                                            [0.0, 0.0, 1.0, -2.0]]], device=device)
                dummy_intrinsics = torch.zeros(1, 1, 3, 3, device=device)

                # 3. カメラエンコーディング変換
                camera = extri_intri_to_pose_encoding(
                    extrinsics=extrinsics.unsqueeze(0),
                    intrinsics=dummy_intrinsics,
                    image_size_hw=(image_size, image_size),
                )
                cam_token = camera[..., :7].squeeze(0)

                # 4. 公式推論の実行
                with torch.no_grad():
                    pred = model(x=img_tensor, cam_params=cam_token)
                    
                    # 出力テンソル (1, 3, 256, 256) -> PIL Image 変換
                    out_tensor = pred.squeeze(0).cpu()
                    out_img_np = out_tensor.numpy().transpose(1, 2, 0)
                    out_img_np = (np.clip(out_img_np, 0.0, 1.0) * 255).astype(np.uint8)
                    generated_img = Image.fromarray(out_img_np)

                with col2:
                    st.subheader("生成された2.5D視点")
                    st.image(generated_img, use_container_width=True)
                
                st.success("生成完了！")

            except Exception as e:
                st.error(f"推論中にエラーが発生したよ: {e}")

            finally:
                gc.collect()
