import os
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from transformers import pipeline
import base64
from io import BytesIO

st.set_page_config(page_title="2.5D パララックス画像", layout="wide")
st.title("📱 2.5D カラー立体画像")

@st.cache_resource
def load_model():
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

try:
    with st.spinner("AIモデル準備中..."):
        pipe = load_model()
except Exception as e:
    st.error(f"エラー: {e}")

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

uploaded_file = st.file_uploader("画像をアップロードしてね", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    aspect_ratio = height / width

    with st.spinner("奥行きを計算中..."):
        result = pipe(image)
        depth_image = result["depth"].convert("L")

    st.subheader("👇 指でなぞって動かしてみてね！")

    img_b64 = image_to_base64(image)
    depth_b64 = image_to_base64(depth_image)

    display_height = int(600 * aspect_ratio)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background-color: transparent; display: flex; justify-content: center; align-items: center; overflow: hidden; }}
            canvas {{ width: 100%; height: auto; border-radius: 12px; touch-action: none; }}
        </style>
    </head>
    <body>
        <canvas id="glcanvas"></canvas>
        <script>
            const imgSrc = "data:image/png;base64,{img_b64}";
            const depthSrc = "data:image/png;base64,{depth_b64}";

            const canvas = document.getElementById("glcanvas");
            const gl = canvas.getContext("webgl");

            const vsSource = `
                attribute vec2 a_position;
                varying vec2 v_texCoord;
                void main() {{
                    gl_Position = vec4(a_position, 0.0, 1.0);
                    v_texCoord = (a_position + 1.0) / 2.0;
                    v_texCoord.y = 1.0 - v_texCoord.y;
                }}
            `;

            // 動く幅（立体感）を戻しつつ、急激な輪郭バグだけを滑らかに緩和するシェーダー
            const fsSource = `
                precision mediump float;
                uniform sampler2D u_image;
                uniform sampler2D u_depth;
                uniform vec2 u_mouse;
                varying vec2 v_texCoord;

                void main() {{
                    float depth = texture2D(u_depth, v_texCoord).r;

                    // 変位量（動きの大きさ）を 0.045 に調整（しっかり立体的に動く！）
                    vec2 offset = u_mouse * (depth - 0.5) * 0.045;
                    
                    vec2 targetUV = v_texCoord + offset;
                    float targetDepth = texture2D(u_depth, targetUV).r;

                    // 境目の強烈なガタツキだけをやんわり和らげる（完全ストップはしない）
                    float depthDiff = abs(depth - targetDepth);
                    if (depthDiff > 0.35) {{
                        offset *= (1.0 - (depthDiff - 0.35) * 1.2);
                    }}

                    vec2 finalUV = clamp(v_texCoord + offset, 0.001, 0.999);
                    gl_FragColor = texture2D(u_image, finalUV);
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

            function updatePos(clientX, clientY) {{
                const rect = canvas.getBoundingClientRect();
                targetX = ((clientX - rect.left) / rect.width - 0.5) * 2;
                targetY = ((clientY - rect.top) / rect.height - 0.5) * 2;
            }}

            window.addEventListener("mousemove", (e) => updatePos(e.clientX, e.clientY));
            window.addEventListener("touchmove", (e) => {{
                if(e.touches.length > 0) {{
                    updatePos(e.touches[0].clientX, e.touches[0].clientY);
                }}
            }}, {{passive: true}});

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

    components.html(html_code, height=display_height + 20)
