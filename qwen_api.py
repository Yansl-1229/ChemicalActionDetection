# -*- coding: utf-8 -*-
from openai import OpenAI
import base64
import os
import mimetypes
import time

# Set OpenAI's API key and API base to use vLLM's API server.
OPENAI_API_KEY = "EMPTY"
# 内网： "http://localhost:8000/v1"
# 外网调试： "http://36.7.148.20:9100/api/vllm/v1"
OPENAI_API_BASE = "http://localhost:8000/v1"

STEP_DESCRIPTIONS = {
    "step01": "step01：配置硫酸铜溶液：使用药匙取适量硫酸铜颗粒加入烧杯中，并加入适量蒸馏水，搅拌均匀后倒入广口瓶中。刷洗烧杯和玻璃棒，并将废液倒入废液缸中。",
    "step02": "step02：配置氢氧化钠溶液：使用药匙取适量氢氧化钠颗粒加入烧杯中，并加入适量蒸馏水，搅拌均匀后倒入氢氧化钠溶液广口瓶中。刷洗烧杯和玻璃棒，并将废液倒入废液缸中。",
    "step03": "step03：配置柠檬酸溶液：使用药匙取适量柠檬酸颗粒加入烧杯中，并加入适量蒸馏水，搅拌均匀后倒入柠檬酸溶液广口瓶中。刷洗烧杯和玻璃棒，并将废液倒入废液缸中。",
    "step04": "step04：使用滴管在点滴板上分别滴加3-4滴氢氧化钠溶液，再分别滴加2-3滴紫色石蕊溶液和无色酚酞溶液；使用滴管在点滴板上分别滴加3-4滴柠檬酸溶液，再分别滴加2-3滴紫色石蕊溶液和无色酚酞溶液；观察点滴板上的实验现象后，将点滴板上的溶液倒入废液缸；",
    "step05": "step05：在试管中加入适量的硫酸铜溶液，然后再实验滴管加入适量氢氧化钠溶液，最后将试管放回试管架。"
}

def encode_file(file_path):
    """
    Encodes a local file to a data URI (Base64).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            mime_type = 'video/mp4'
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'application/octet-stream'

    with open(file_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{encoded_string}"

def get_media_content(path_or_url):
    """
    Returns the appropriate content string for the message.
    """
    if path_or_url.startswith(('http://', 'https://')):
        return path_or_url
    else:
        return encode_file(path_or_url)

class QwenVLClient:
    def __init__(self, api_base=OPENAI_API_BASE, api_key=OPENAI_API_KEY):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.model_name = self._detect_model()

    def _detect_model(self):
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
        except Exception as e:
            print(f"Model detection failed: {e}")
            pass
        return "/home/sanlian/Qwen3-VL-30B-A3B-Instruct-FP8" # Fallback name

    def analyze_step(self, video_path, step_key=None, description=None, mock=False):
        """
        Analyzes a specific step video using Qwen-VL.
        """
        if mock:
            return f"Mock Analysis for {step_key}: The actions in the video appear to match the standard operating procedure. No significant deviations detected."

        if description is None:
            if step_key in STEP_DESCRIPTIONS:
                description = STEP_DESCRIPTIONS[step_key]
            else:
                return f"Error: Unknown step {step_key} and no description provided."

        prompt = f"""请你对照下面的化学实验流程，检查当前完整视频中的实验过程是否和文本一致（必须确保文本中的每个流程都被视频中的动作所对应）。
        请务必输出合法的JSON格式，不要包含markdown代码块标记（如```json），包含以下字段：
        {{
            "pass_check": true,  // 布尔值，如果一致则为true，否则为false
            "qwen_analysis": "说明理由。若有不一致的地方，请指出。请将回复控制在50个字以内。"
        }}

        实验步骤：
        {description}
        """

        try:
            video_content = get_media_content(video_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": video_content}},
                    ],
                }
            ]

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            return completion.choices[0].message.content

        except Exception as e:
            print(f"API Call Failed: {e}")
            return f"Simulation: API connection failed. Assuming error in action for {step_key}. Please check the video manually. (Error: {str(e)})"

# Global instance
_client_instance = None

def analyze_with_qwen(video_path, step_key=None, description=None):
    global _client_instance
    if _client_instance is None:
        _client_instance = QwenVLClient()
    
    # Try to connect, fall back to mock if server is down (typical for demo)
    return _client_instance.analyze_step(video_path, step_key, description=description)
