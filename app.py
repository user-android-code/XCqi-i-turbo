import os
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from transformers import pipeline
import base64
from io import BytesIO

st.set_page_config(page_title="2.5D 視差効果（パララックス）AI", layout="wide")
st.title("🎥 2.5D 視差効果（ゆらゆら動く画像）ジェネレーター")

@st.cache_resource
def load_model():
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

try:
    with st.spinner("AIモデルを読み込み中..."):
        pipe = load_model()
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")

# 画像をBase64文字列に変換するヘルパー関数
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像")
        st.image(image, use_container_width=True)

    with st.spinner("奥行き（深度）を推定中..."):
        result = pipe(image)
        depth_image = result["depth"].convert("L")

    with col2:
        st.subheader("生成された深度画像")
        st.image(depth_image, use_container_width=True)

    st.subheader("🎮 2.5D パララックス空間 (マウスを画像の上で動かしてみてね！)")

    # 画像と深度マップをHTML/JSに埋め込む
    img_b64 = image_to_base64(image)
    depth_b64 = image_to_base64(depth_image)

    # 2.5Dパララックス効果を生むHTML/WebGLコード
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #0e1117; display: flex; justify-content: center; align-items: center; }}
            canvas {{ border-radius: 10px; cursor: pointer; max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <canvas id="glcanvas"></canvas>
        <script>
            const imgSrc = "data:image/png;base64,{img_b64}";
            const depthSrc = "data:image/png;base64,{depth_b64}";

            const canvas = document.getElementById("glcanvas");
            const gl = canvas.getContext("webgl");

            if (!gl) {{
                alert("WebGLがサポートされていません");
            }}

            // バーテックスシェーダー
            const vsSource = `
                attribute vec2 a_position;
                varying vec2 v_texCoord;
                void main() {{
                    gl_Position = vec4(a_position, 0.0, 1.0);
                    v_texCoord = (a_position + 1.0) / 2.0;
                    v_texCoord.y = 1.0 - v_texCoord.y;
                }}
            `;

            // フラグメントシェーダー (深度に基づく視差ずれ効果)
            const fsSource = `
                precision mediump float;
                uniform sampler2D u_image;
                uniform sampler2D u_depth;
                uniform vec2 u_mouse;
                varying vec2 v_texCoord;

                void main() {{
                    float depth = texture2D(u_depth, v_texCoord).r;
                    vec2 offset = u_mouse * (depth - 0.5) * 0.05;
                    gl_FragColor = texture2D(u_image, v_texCoord + offset);
                }}
            `;

            function createShader(gl, type, source) {{
                const shader = gl.createShader(type);
                gl.shaderSource(shader, source);
                gl.compileShader(shader);
                return shader;
            }}

            const program = gl.createProgram();
            gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, vsSource));
            gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, fsSource));
            gl.linkProgram(program);
            gl.useProgram(program);

            const positionBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                -1, -1,  1, -1, -1,  1,
                -1,  1,  1, -1,  1,  1,
            ]), gl.STATIC_DRAW);

            const positionLocation = gl.getAttribLocation(program, "a_position");
            gl.enableVertexAttribArray(positionLocation);
            gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

            const mouseLoc = gl.getUniformLocation(program, "u_mouse");
            let mouseX = 0, mouseY = 0;
            let targetX = 0, targetY = 0;

            window.addEventListener("mousemove", (e) => {{
                const rect = canvas.getBoundingClientRect();
                targetX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
                targetY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
            }});

            function loadTexture(url, index) {{
                const texture = gl.createTexture();
                gl.activeTexture(gl.TEXTURE0 + index);
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));

                const img = new Image();
                img.onload = () => {{
                    canvas.width = img.width;
                    canvas.height = img.height;
                    gl.viewport(0, 0, img.width, img.height);
                    gl.activeTexture(gl.TEXTURE0 + index);
                    gl.bindTexture(gl.TEXTURE_2D, texture);
                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                }};
                img.src = url;
                return texture;
            }}

            loadTexture(imgSrc, 0);
            loadTexture(depthSrc, 1);

            gl.uniform1i(gl.getUniformLocation(program, "u_image"), 0);
            gl.uniform1i(gl.getUniformLocation(program, "u_depth"), 1);

            function render() {{
                mouseX += (targetX - mouseX) * 0.1;
                mouseY += (targetY - mouseY) * 0.1;
                gl.uniform2f(mouseLoc, mouseX, -mouseY);

                gl.drawArrays(gl.TRIANGLES, 0, 6);
                requestAnimationFrame(render);
            }}
            render();
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=600)
