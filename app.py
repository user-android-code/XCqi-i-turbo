import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import time

st.set_page_config(page_title="Xmugi / Cpu Demo")

st.title("Xmugi / Cpu Demo")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    pipe.enable_attention_slicing()
    return pipe

st.write("Please enter your prompt in English.")
user_input = st.text_input("Prompt (English only)", "")

if st.button("execution"):
    if user_input.strip():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = 20
        last_percent = [0]

        def callback_fn(step, timestep, latents):
            current_step = min(step + 1, total_steps)
            target_percent = int((current_step / total_steps) * 100)
            for p in range(last_percent[0] + 1, target_percent + 1):
                progress_bar.progress(p)
                status_text.text(f"{p}/100")
                time.sleep(0.01)
            last_percent[0] = target_percent

        pipe = load_pipeline()
        
        image = pipe(
            user_input,
            height=256,
            width=256,
            num_inference_steps=total_steps,
            callback=callback_fn,
            callback_steps=1
        ).images[0]

        progress_bar.progress(100)
        status_text.text("100/100")
        st.image(image)
