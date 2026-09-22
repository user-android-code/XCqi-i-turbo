import streamlit as st
import torch
import os
import urllib.request

st.set_page_config(page_title="LaMa Download Test")
st.title("LaMa Model Download Test")

model_dir = os.path.expanduser("~/.cache/simple_lama")
model_path = os.path.join(model_dir, "big-lama.pt")

url = "https://github.com/advimman/lama/raw/main/big-lama.pt"

if not os.path.exists(model_path):
    st.info("GitHubから big-lama.pt を直接ダウンロード中...")
    os.makedirs(model_dir, exist_ok=True)
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
        out_file.write(response.read())
    st.success("ダウンロード完了！")
else:
    st.info("すでにダウンロード済みのモデルを使用するよ。")

st.write("モデルをCPU上にロード中...")

try:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    st.success("モデルのロードに成功したよ！")
except Exception as e:
    st.error(f"ロード失敗: {e}")
