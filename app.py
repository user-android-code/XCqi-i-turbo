import sys
import os
import gc
import base64
import io
import urllib.request
import zipfile
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
import streamlit.components.v1 as components

# ---------------------------------------------------------
# GitHubコードの自動ダウンロード（2ファイル完結用）
# ---------------------------------------------------------
OVIE_CODE_DIR = os.path.abspath("./ovie_repo")

if not os.path.exists(OVIE_CODE_DIR):
    os.makedirs(OVIE_CODE_DIR, exist_ok=True)
    zip_path = os.path.join(OVIE_CODE_DIR, "ovie.zip")
    url = "https://github.com/kyutai-labs/ovie/archive/refs/heads/main.zip"
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OVIE_CODE_DIR)
    
    extracted_folder = os.path.join(OVIE_CODE_DIR, "ovie-main")
    if extracted_folder not in sys.path:
        sys.path.insert(0, extracted_folder)
else:
    extracted_folder = os.path.join(OVIE_CODE_DIR, "ovie-main")
    if extracted_folder not in sys.path:
        sys.path.insert(0, extracted_folder)

# ---------------------------------------------------------
# Streamlit ページ構成
# ---------------------------------------------------------
st.set_page_config(
    page_title="Xcqi i-air",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# サイドバー・不要UIの非表示
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        .block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("Xcqi i-air")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# kyutai/ovie ロード
# ---------------------------------------------------------
@st.cache_resource
def load_ovie_model():
    from models.models import OVIEModel
    from utils.pose_enc import extri_intri_to_pose_encoding
    
    model = OVIEModel.from_pretrained("kyutai/ovie", revision="v1.0").to(device)
    model.eval()
    return model, extri_intri_to_pose_encoding

try:
    with st.spinner("OVIEモデルをロード中..."):
        model, extri_intri_to_pose_encoding = load_ovie_model()
except Exception as e:
    st.error(f"OVIEのロードに失敗しました: {e}")
    st.stop()

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    image_size = getattr(model, "image_size", 256)
    img_pil = ImageOps.fit(raw_img, (image_size, image_size), Image.Resampling.LANCZOS)

    file_id = uploaded_file.name + str(uploaded_file.size)
    if "ovie_cache" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("OVIEで8方向の空間視点を生成中..."):
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)
            dummy_intrinsics = torch.zeros(1, 1, 3, 3, device=device)

            angles = {
                "center": (0.0, 0.0),
                "left": (-1.25, 0.0),
                "right": (1.25, 0.0),
                "up": (0.0, 0.5),
                "down": (0.0, -0.5),
                "top_left": (-1.0, 0.4),
                "top_right": (1.0, 0.4),
                "bottom_left": (-1.0, -0.4),
                "bottom_right": (1.0, -0.4),
            }

            rendered_images = {"center": img_pil}

            for key, (pos_x, pos_y) in angles.items():
                if key == "center":
                    continue
                
                extrinsics = torch.tensor([[[1.0, 0.0, 0.0, pos_x],
                                            [0.0, 1.0, 0.0, pos_y],
                                            [0.0, 0.0, 1.0, -2.0]]], device=device)

                camera = extri_intri_to_pose_encoding(
                    extrinsics=extrinsics.unsqueeze(0),
                    intrinsics=dummy_intrinsics,
                    image_size_hw=(image_size, image_size),
                )
                cam_token = camera[..., :7].squeeze(0)

                with torch.no_grad():
                    pred = model(x=img_tensor, cam_params=cam_token)
                    out_tensor = pred.squeeze(0).cpu()
                    out_img_np = out_tensor.numpy().transpose(1, 2, 0)
                    out_img_np = (np.clip(out_img_np, 0.0, 1.0) * 255).astype(np.uint8)
                    rendered_images[key] = Image.fromarray(out_img_np)

            # Base64化
            b64_dict = {}
            for k, img in rendered_images.items():
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_dict[k] = base64.b64encode(buf.getvalue()).decode()

            st.session_state.ovie_cache = b64_dict
            st.session_state.file_id = file_id
            gc.collect()

    b64_data = st.session_state.ovie_cache

    # ---------------------------------------------------------
    # マウスポインター追従（iOS 26 空間シーン演出）
    # ---------------------------------------------------------
    html_code = f"""
    <div style="display: flex; justify-content: center; align-items: center; padding: 10px;">
        <div id="spatial-card" style="
            position: relative;
            width: 320px;
            height: 320px;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 15px 35px rgba(0,0,0,0.25);
            cursor: pointer;
            transform-style: preserve-3d;
            transition: transform 0.1s ease-out;
        ">
            <img id="scene-img" src="data:image/png;base64,{b64_data['center']}" style="width: 100%; height: 100%; object-fit: cover; display: block;" />
        </div>
    </div>

    <script>
        const card = document.getElementById('spatial-card');
        const img = document.getElementById('scene-img');

        const images = {{
            center: "data:image/png;base64,{b64_data['center']}",
            left: "data:image/png;base64,{b64_data['left']}",
            right: "data:image/png;base64,{b64_data['right']}",
            up: "data:image/png;base64,{b64_data['up']}",
            down: "data:image/png;base64,{b64_data['down']}",
            top_left: "data:image/png;base64,{b64_data['top_left']}",
            top_right: "data:image/png;base64,{b64_data['top_right']}",
            bottom_left: "data:image/png;base64,{b64_data['bottom_left']}",
            bottom_right: "data:image/png;base64,{b64_data['bottom_right']}"
        }};

        window.addEventListener('mousemove', (e) => {{
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {{
                const rotX = (y - 0.5) * -20;
                const rotY = (x - 0.5) * 20;
                card.style.transform = `perspective(1000px) rotateX(${{rotX}}deg) rotateY(${{rotY}}deg)`;

                let key = "center";
                if (x < 0.35 && y < 0.35) key = "top_left";
                else if (x > 0.65 && y < 0.35) key = "top_right";
                else if (x < 0.35 && y > 0.65) key = "bottom_left";
                else if (x > 0.65 && y > 0.65) key = "bottom_right";
                else if (x < 0.35) key = "left";
                else if (x > 0.65) key = "right";
                else if (y < 0.35) key = "up";
                else if (y > 0.65) key = "down";

                if (img.src !== images[key]) {{
                    img.src = images[key];
                }}
            }} else {{
                card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg)`;
                img.src = images["center"];
            }}
        }});
    </script>
    """

    components.html(html_code, height=360)
