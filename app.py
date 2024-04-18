# InternLM2-chat-1.8B-LoRA
# -- coding: utf-8 --
# -------------------------------
# @Author : Ning Zhang
# @Email : zhang_n@zju.edu.cn
# -------------------------------
# @File : app.py.py
# @Time : 2024/4/18 14:25
# -------------------------------
import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig
import os

# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-1.8b-4bit'
os.system(f'git clone -b master https://code.openxlab.org.cn/kino/internlm2-chat-1_8b-4bit.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline(base_path, backend_config=backend_config)

def model(image, text):
    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        response = pipe((text, image)).text
        return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo.launch()


