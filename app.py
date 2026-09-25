import os
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageFilter
import numpy as np
from transformers import pipeline
import base64
from io import BytesIO

st.set_page_config(page_title="XCqi", layout="wide")
st.title("XCqi i-turbo")

@st.cache_resource
def load_depth_model():
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

pipe = load_depth_model()

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_inpainted_background_and_fg(image, depth_image):
    depth_array = np.array(depth_image)
    threshold = np.percentile(depth_array, 50)
    mask_array = (depth_array > threshold).astype(np.uint8) * 255
    
    mask = Image.fromarray(mask_array).resize(image.size, Image.BILINEAR)
    
    fg_array = np.array(image)
    alpha = np.array(mask)
    fg_rgba = np.dstack((fg_array, alpha))
    fg_image = Image.fromarray(fg_rgba, mode="RGBA")
    
    bg_array = np.array(image).astype(np.float32)
    inv_mask = (alpha < 128).astype(np.float32)[:, :, None]
    
    bg_base = bg_array * inv_mask
    blurred_bg = Image.fromarray(bg_array.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=25))
    blurred_bg_array = np.array(blurred_bg).astype(np.float32)
    
    bg_final_array = bg_base + blurred_bg_array * (1.0 - inv_mask)
    bg_image = Image.fromarray(np.clip(bg_final_array, 0, 255).astype(np.uint8))
    
    return fg_image, bg_image

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    aspect_ratio = height / width

    with st.spinner("XCqi is calculating."):
        result = pipe(image)
        depth_image = result["depth"].convert("L")
        
        fg_image, bg_image = create_inpainted_background_and_fg(image, depth_image)

    fg_b64 = image_to_base64(fg_image)
    bg_b64 = image_to_base64(bg_image)

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

            const fsSource = `
                precision lowp float;
                uniform sampler2D u_fg;
                uniform sampler2D u_bg;
                uniform vec2 u_mouse;
                varying vec2 v_texCoord;

                void main() {{
                    vec2 bgUV = clamp(v_texCoord - u_mouse * 0.01, 0.0, 1.0);
                    vec4 bgColor = texture2D(u_bg, bgUV);

                    vec2 fgUV = clamp(v_texCoord + u_mouse * 0.035, 0.0, 1.0);
                    vec4 fgColor = texture2D(u_fg, fgUV);

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

            loadTexture(fgSrc, 0);
            loadTexture(bgSrc, 1);

            gl.uniform1i(gl.getUniformLocation(program, "u_fg"), 0);
            gl.uniform1i(gl.getUniformLocation(program, "u_bg"), 1);

            function render() {{
                gl.uniform2f(mouseLoc, targetX, -targetY);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
                requestAnimationFrame(render);
            }}
            render();
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=display_height)
