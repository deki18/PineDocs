import os
import sys

for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(proxy_var, None)

os.environ["no_proxy"] = "127.0.0.1,localhost"

import uvicorn
import gradio as gr
from fastapi import FastAPI
from app_gradio import create_ui

demo = create_ui()

app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
