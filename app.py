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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ---------------------------------------------------------
# 1. GitHubコードの自動取得（OVIE用）
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

# ---------------------------------------------------------
# 2. ページ設定 & UI非表示
# ---------------------------------------------------------
st.set_page_config(
    page_title="Xcqi i-air",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
# 3. W AIモデル（OVIE & Depth-Anything-V2）のロード
# ---------------------------------------------------------
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

@st.cache_resource
def load_all_models():
    # 1. OVIE ロード
    from models.models import OVIEModel
    from utils.pose_enc import extri_intri_to_pose_encoding
    ovie_model = OVIEModel.from_pretrained("kyutai/ovie", revision="v1.0").to(device)
    ovie_model.eval()

    # 2. Depth-Anything-V2 ロード
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
    depth_model.eval()

    return ovie_model, extri_intri_to_pose_encoding, depth_processor, depth_model

try:
    with st.spinner("AIモデル（OVIE + Depth-V2）を並列ロード中..."):
        ovie_model, extri_intri_to_pose_encoding, depth_processor, depth_model = load_all_models()
except Exception as e:
    st.error(f"モデルのロードに失敗しました: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. メインパイプライン処理
# ---------------------------------------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    image_size = getattr(ovie_model, "image_size", 256)
    img_pil = ImageOps.fit(raw_img, (image_size, image_size), Image.Resampling.LANCZOS)

    file_id = uploaded_file.name + str(uploaded_file.size)

    # OVIEで視点別画像を生成し、Depth-Anythingでそれぞれに3D立体化効果を付与
    if "hybrid_cache" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("OVIEで視点生成 ➔ Depth-V2で空間3D化処理中..."):
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)
            dummy_intrinsics = torch.zeros(1, 1, 3, 3, device=device)

            angles = {
                "center": (0.0, 0.0),
                "left": (-1.25, 0.0),
                "right": (1.25, 0.0),
                "up": (0.0, 0.5),
                "down": (0.0, -0.5),
            }

            rendered_images = {"center": img_pil}

            # 1. OVIEによる多視点生成
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
                    pred = ovie_model(x=img_tensor, cam_params=cam_token)
                    out_tensor = pred.squeeze(0).cpu()
                    out_img_np = out_tensor.numpy().transpose(1, 2, 0)
                    out_img_np = (np.clip(out_img_np, 0.0, 1.0) * 255).astype(np.uint8)
                    rendered_images[key] = Image.fromarray(out_img_np)

            # 2. Depth-Anything-V2 で各画像から高度な深度マップを推定 ➔ Base64化
            hybrid_cache = {}
            for k, img in rendered_images.items():
                # 深度マップ計算
                inputs = depth_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = depth_model(**inputs)
                    predicted_depth = outputs.predicted_depth

                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(image_size, image_size),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

                depth_norm = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
                depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))

                # RGB画像Base64化
                buf_rgb = io.BytesIO()
                img.save(buf_rgb, format="PNG")
                b64_rgb = base64.b64encode(buf_rgb.getvalue()).decode()

                # Depth画像Base64化
                buf_depth = io.BytesIO()
                depth_img.save(buf_depth, format="PNG")
                b64_depth = base64.b64encode(buf_depth.getvalue()).decode()

                hybrid_cache[k] = {"rgb": b64_rgb, "depth": b64_depth}

            st.session_state.hybrid_cache = hybrid_cache
            st.session_state.current_key = "center"
            st.session_state.file_id = file_id
            gc.collect()

    # 十字キーUI
    col_u1, col_u2, col_u3 = st.columns([1, 1, 1])
    with col_u2:
        if st.button("▲ 上アングル (OVIE)", use_container_width=True):
            st.session_state.current_key = "up"

    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    with col_m1:
        if st.button("◀ 左アングル", use_container_width=True):
            st.session_state.current_key = "left"
    with col_m2:
        if st.button("正面 (0)", use_container_width=True):
            st.session_state.current_key = "center"
    with col_m3:
        if st.button("右アングル ▶", use_container_width=True):
            st.session_state.current_key = "right"

    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    with col_d2:
        if st.button("▼ 下アングル", use_container_width=True):
            st.session_state.current_key = "down"

    st.markdown("---")

    # 現在選択されている視点のデータ
    curr_key = st.session_state.get("current_key", "center")
    data = st.session_state.hybrid_cache[curr_key]

    # 3Dシェーダー（マウスポインターでぬるぬる立体的に傾くカード演出）
    html_code = f"""
    <div style="display: flex; justify-content: center; align-items: center; padding: 10px;">
        <div id="spatial-card" style="
            position: relative;
            width: 320px;
            height: 320px;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
            cursor: pointer;
            transform-style: preserve-3d;
            transition: transform 0.1s ease-out;
        ">
            <canvas id="spatial-canvas" width="320" height="320" style="width: 100%; height: 100%; display: block;"></canvas>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const card = document.getElementById('spatial-card');
        const canvas = document.getElementById('spatial-canvas');
        
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
        camera.position.z = 2.5;

        const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: true }});
        renderer.setSize(320, 320);

        const loader = new THREE.TextureLoader();
        const imgTex = loader.load('data:image/png;base64,{data["rgb"]}');
        const depthTex = loader.load('data:image/png;base64,{data["depth"]}');

        const geometry = new THREE.PlaneGeometry(2, 2);
        const material = new THREE.ShaderMaterial({{
            uniforms: {{
                uTexture: {{ value: imgTex }},
                uDepth: {{ value: depthTex }},
                uOffset: {{ value: new THREE.Vector2(0, 0) }}
            }},
            vertexShader: `
                varying vec2 vUv;
                void main() {{
                    vUv = uv;
                    gl_Position = vec4(position, 1.0);
                }}
            `,
            fragmentShader: `
                uniform sampler2D uTexture;
                uniform sampler2D uDepth;
                uniform vec2 uOffset;
                varying vec2 vUv;
                
                void main() {{
                    float d = texture2D(uDepth, vUv).r;
                    vec2 displacedUv = vUv + uOffset * (d - 0.5) * 0.15;
                    gl_FragColor = texture2D(uTexture, displacedUv);
                }}
            `
        }});

        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        let mouseX = 0, mouseY = 0;
        let targetX = 0, targetY = 0;

        window.addEventListener('mousemove', (e) => {{
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {{
                targetX = (x / rect.width) * 2 - 1;
                targetY = -(y / rect.height) * 2 + 1;
            }} else {{
                targetX = 0;
                targetY = 0;
            }}
        }});

        function render() {{
            requestAnimationFrame(render);
            mouseX += (targetX - mouseX) * 0.1;
            mouseY += (targetY - mouseY) * 0.1;

            material.uniforms.uOffset.value.set(-mouseX, -mouseY);
            card.style.transform = `perspective(1000px) rotateY(${{mouseX * 15}}deg) rotateX(${{-mouseY * 15}}deg)`;

            renderer.render(scene, camera);
        }}
        render();
    </script>
    """

    components.html(html_code, height=360)
