import os
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
import cv2
import torch
import gc
from transformers import pipeline
import base64
from io import BytesIO

st.set_page_config(page_title="XCqi", layout="wide")
st.title("XCqi i-turbo")

@st.cache_resource
def load_models():
    # 1. 深度推定モデル (Depth-Anything-V2)
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    # 2. 軽量背景削除モデル (RMBG-1.4: 約170MB)
    rmbg_pipe = pipeline(task="image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    return depth_pipe, rmbg_pipe

depth_pipe, rmbg_pipe = load_models()

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_background_inpaint(original_img, mask_img):
    """前景領域（マスク）をCV2のInpaintingで簡易穴埋めして背景を作る"""
    img_np = np.array(original_img)
    mask_np = np.array(mask_img.convert("L"))
    
    # マスク領域を少し膨張させて境界のゴミを削る
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
    
    # Teleaアルゴリズムでインペイント（超高速＆メモリ消費ほぼゼロ）
    inpainted_np = cv2.inpaint(img_np, dilated_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted_np)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    aspect_ratio = height / width

    with st.spinner("XCqi is calculating (Separating Layers)..."):
        with torch.no_grad():
            # 深度マップ生成
            depth_result = depth_pipe(image)
            depth_image = depth_result["depth"].convert("L")
            
            # 前景切り抜き (RGBA)
            fg_image = rmbg_pipe(image)
            
            # アルファチャンネルからマスクを抽出して背景を補完
            alpha_mask = fg_image.split()[-1]
            bg_image = create_background_inpaint(image, alpha_mask)
            
            # ガベージコレクションでメモリ即時解放
            gc.collect()

    fg_b64 = image_to_base64(fg_image)
    bg_b64 = image_to_base64(bg_image)
    depth_b64 = image_to_base64(depth_image)

    display_height = int(750 * aspect_ratio) if aspect_ratio < 1.2 else 650

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                background-color: transparent;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
                width: 100vw;
                height: 100vh;
            }}
            canvas {{
                max-width: 100%;
                max-height: 100vh;
                width: auto;
                height: auto;
                object-fit: contain;
                border-radius: 12px;
                touch-action: none;
                will-change: transform;
            }}
        </style>
    </head>
    <body>
        <canvas id="glcanvas"></canvas>
        <script>
            const fgSrc = "data:image/png;base64,{fg_b64}";
            const bgSrc = "data:image/png;base64,{bg_b64}";
            const depthSrc = "data:image/png;base64,{depth_b64}";

            const canvas = document.getElementById("glcanvas");
            
            const gl = canvas.getContext("webgl", {{
                preserveDrawingBuffer: false,
                powerPreference: "high-performance",
                alpha: true,
                desynchronized: true
            }});

            const vsSource = `
                attribute vec2 a_position;
                varying vec2 v_texCoord;
                void main() {{
                    gl_Position = vec4(a_position, 0.0, 1.0);
                    v_texCoord = (a_position + 1.0) * 0.5;
                    v_texCoord.y = 1.0 - v_texCoord.y;
                }}
            `;

            // 手前（FG）と背景（BG）を合成し、背景のゴーストを防ぐシェーダー
            const fsSource = `
                precision mediump float;
                uniform sampler2D u_fg;
                uniform sampler2D u_bg;
                uniform sampler2D u_depth;
                uniform vec2 u_mouse;
                varying vec2 v_texCoord;

                void main() {{
                    float depth = texture2D(u_depth, v_texCoord).r;

                    // 背景は極小のオフセットで安定させる
                    vec2 bgOffset = u_mouse * -0.008;
                    vec2 bgUV = clamp(v_texCoord + bgOffset, 0.001, 0.999);
                    vec4 bgColor = texture2D(u_bg, bgUV);

                    // 前景は深度に連動して大きく移動
                    vec2 fgOffset = u_mouse * (depth - 0.3) * 0.05;
                    vec2 fgUV = clamp(v_texCoord + fgOffset, 0.001, 0.999);
                    vec4 fgColor = texture2D(u_fg, fgUV);

                    // アルファブレンディング（前景の手前に背景を透過合成）
                    vec3 finalColor = mix(bgColor.rgb, fgColor.rgb, fgColor.a);
                    gl_FragColor = vec4(finalColor, 1.0);
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
                targetX = ((clientX - rect.left) / rect.width - 0.5) * 2.0;
                targetY = ((clientY - rect.top) / rect.height - 0.5) * 2.0;
            }}

            window.addEventListener("pointermove", (e) => updatePos(e.clientX, e.clientY), {{passive: true}});

            function loadTexture(url, index) {{
                const texture = gl.createTexture();
                gl.activeTexture(gl.TEXTURE0 + index);
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,0]));

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

            loadTexture(fgSrc, 0);
            loadTexture(bgSrc, 1);
            loadTexture(depthSrc, 2);

            gl.uniform1i(gl.getUniformLocation(program, "u_fg"), 0);
            gl.uniform1i(gl.getUniformLocation(program, "u_bg"), 1);
            gl.uniform1i(gl.getUniformLocation(program, "u_depth"), 2);

            function render() {{
                mouseX += (targetX - mouseX) * 0.15;
                mouseY += (targetY - mouseY) * 0.15;
                gl.uniform2f(mouseLoc, mouseX, -mouseY);

                gl.drawArrays(gl.TRIANGLES, 0, 6);
                requestAnimationFrame(render);
            }}
            render();
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=display_height)
