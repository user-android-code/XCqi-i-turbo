import gc
import base64
import io
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import streamlit.components.v1 as components
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

st.set_page_config(
    page_title="Xcqi i-air",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# サイドバー・不要なUIの完全非表示
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        .block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("Xcqi i-air")

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

@st.cache_resource
def load_depth_model():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.eval()
    return image_processor, model

try:
    image_processor, model = load_depth_model()
except Exception as e:
    st.error(f"モデルロード失敗: {e}")
    st.stop()

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_256 = ImageOps.fit(raw_img, (256, 256), Image.Resampling.LANCZOS)

    with st.spinner("iOS空間シーンを生成中..."):
        # 1. 深度推定
        inputs = image_processor(images=img_256, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        depth_norm = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)

        # 2. 画像と深度マップをBase64データ化
        buffered = io.BytesIO()
        img_256.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))
        buffered_depth = io.BytesIO()
        depth_img.save(buffered_depth, format="PNG")
        depth_b64 = base64.b64encode(buffered_depth.getvalue()).decode()

    # 3. iOS Spatial Photo風 リアルタイム・パララックス（視差）キャンバス
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
        const imgTex = loader.load('data:image/png;base64,{img_b64}');
        const depthTex = loader.load('data:image/png;base64,{depth_b64}');

        // 空間シーン用カスタムシェーダー（視差・奥行き歪み）
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
                    vec2 displacedUv = vUv + uOffset * (d - 0.5) * 0.12;
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
                // -1.0 ~ 1.0 の範囲に正規化
                targetX = (x / rect.width) * 2 - 1;
                targetY = -(y / rect.height) * 2 + 1;
            }} else {{
                targetX = 0;
                targetY = 0;
            }}
        }});

        function render() {{
            requestAnimationFrame(render);

            // iOS風のなめらかなイージング（慣性移動）
            mouseX += (targetX - mouseX) * 0.1;
            mouseY += (targetY - mouseY) * 0.1;

            // 1. シェーダーの視差オフセット更新
            material.uniforms.uOffset.value.set(-mouseX, -mouseY);

            // 2. カード自体の立体的な傾き（iOSのSpatial Photoカード効果）
            card.style.transform = `perspective(1000px) rotateY(${{mouseX * 15}}deg) rotateX(${{-mouseY * 15}}deg)`;

            renderer.render(scene, camera);
        }}
        render();
    </script>
    """

    components.html(html_code, height=360)
