import json
import time
import base64
import urllib3
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict
import re  # 添加正则表达式模块
from ..utils import create_ssl_compatible_session

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Banana2_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "draw a cat"}),
                "header_value": ("STRING", {"default": "Bearer sk-xxx", "multiline": False}),
                "model": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview", "multiline": False}),
                "aspect_ratio": (["auto","1:1", "9:16", "16:9", "4:3", "3:4","3:2","2:3"], {"default": "9:16"}),
                "image_size": (["1K","2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_request"
    CATEGORY = "MohuaAI/工具"
    
    def __init__(self):
        self.session = create_ssl_compatible_session()
        self.api_url = "https://mohuaai.cn/v1beta/models/{model}:generateContent"
        self.header_key = "Authorization"
        # 初始化缺失的配置变量
        self.retry_count = 3  # 最大重试次数
        self.timeout = 180  # 请求超时时间（秒）

    def extract_url_from_markdown(self, text):
        """从Markdown格式的文本中提取图片URL"""
        if not text:
            return None
        
        # 匹配Markdown图片格式: ![alt](url)
        pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(pattern, text)
        
        if matches:
            return matches[0]  # 返回第一个匹配的URL
        return None

    def process_request(self, header_value, model, text, aspect_ratio, image_size,
                       image_1=None, image_2=None, image_3=None, image_4=None,
                       image_5=None, image_6=None, image_7=None, image_8=None):
        
        # Collect all images
        input_images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
        valid_images_base64 = []

        # Process each image
        for idx, img in enumerate(input_images):
            if img is not None:
                try:
                    # Convert tensor to numpy
                    # Handle batch dimension if present (take first image)
                    if len(img.shape) == 4:
                        img_np = img[0].cpu().numpy()
                    else:
                        img_np = img.cpu().numpy()
                        
                    # Convert numpy array (0-1) to PIL Image (0-255)
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                    # 统一转换为RGB格式，处理灰度图/透明图问题
                    if img_pil.mode != "RGB":
                        img_pil = img_pil.convert("RGB")
                    
                    # Convert PIL Image to base64
                    buffered = BytesIO()
                    img_pil.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    valid_images_base64.append(img_base64)
                except Exception as e:
                    print(f"Error processing image_{idx+1}: {e}")
                    continue

        # headers
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Connection": "close"  # 强制关闭连接，防止长连接导致的等待
        }
        if self.header_key and header_value:
            headers[self.header_key] = header_value

        # payload construction
        parts = [{"text": text}]
        
        # Add images to parts
        for img_b64 in valid_images_base64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64
                }
            })

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size
                }
            }
        }

        print(f"[Banana2_API] Requesting: {self.api_url.format(model=model)}")
        start_time = time.time()
        response = self.session.post(
            self.api_url.format(model=model),
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        try:
            resp_json = response.json()
        except json.JSONDecodeError:
            resp_json = None

        if response.status_code != 200:
            err_msg = f"HTTP Error {response.status_code}"
            if resp_json:
                err_msg += f": {json.dumps(resp_json, ensure_ascii=False)}"
            raise Exception(err_msg)

        if not resp_json:
            raise Exception("Empty response from API")

        # Parse response to find image data (base64 or url)
        candidates = resp_json.get("candidates", [])
        
        # Check for prompt feedback block (Gemini specific)
        prompt_feedback = resp_json.get("promptFeedback")
        if prompt_feedback and prompt_feedback.get("blockReason"):
            raise Exception(f"Prompt blocked. Reason: {prompt_feedback.get('blockReason')}")

        # Check for candidates finish reason
        if candidates:
            first_candidate = candidates[0]
            finish_reason = first_candidate.get("finishReason")
            if finish_reason and finish_reason != "STOP":
                # If stopped for safety or other reasons and no content
                if not first_candidate.get("content"):
                    safety_ratings = first_candidate.get("safetyRatings", [])
                    raise Exception(f"Generation stopped. Reason: {finish_reason}. Ratings: {json.dumps(safety_ratings, ensure_ascii=False)}")

        found_data = None
        found_type = None  # 'base64' or 'url'

        def is_base64(s):
            return isinstance(s, str) and len(s) > 100 and not s.startswith("http")

        def is_url(s):
            return isinstance(s, str) and s.startswith(("http://", "https://"))

        # Recursive search for image data
        def find_image_recursive(obj):
            if isinstance(obj, dict):
                # Check specific keys first
                for key in ["data", "b64_json", "base64", "url", "image_url", "image"]:
                    if key in obj:
                        val = obj[key]
                        if is_base64(val):
                            return val, 'base64'
                        if is_url(val):
                            return val, 'url'
                        if isinstance(val, dict) and "url" in val:
                            if is_url(val["url"]):
                                return val["url"], 'url'
                
                # Case-insensitive check for keys
                for k, v in obj.items():
                    k_lower = k.lower()
                    if k_lower in ["data", "b64_json", "base64", "url", "image_url", "image"]:
                        if is_base64(v):
                            return v, 'base64'
                        if is_url(v):
                            return v, 'url'
                
                # Recurse
                for k, v in obj.items():
                    res = find_image_recursive(v)
                    if res:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_image_recursive(item)
                    if res:
                        return res
            return None

        # 首先尝试从文本内容中提取Markdown格式的图片链接
        if candidates:
            try:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    # 检查是否有文本内容
                    if "text" in part:
                        text_content = part["text"]
                        # 从Markdown中提取图片URL
                        image_url = self.extract_url_from_markdown(text_content)
                        if image_url and is_url(image_url):
                            found_data = image_url
                            found_type = 'url'
                            print(f"[Banana_API] Found image URL in Markdown text: {image_url}")
                            break
            except Exception as e:
                print(f"[Banana_API] Error extracting URL from Markdown: {e}")

        # 如果没找到，尝试标准的Gemini格式
        if not found_data and candidates:
            try:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    inline_data = part.get("inlineData") or part.get("inline_data")
                    if inline_data and isinstance(inline_data, dict) and "data" in inline_data:
                        found_data = inline_data["data"]
                        found_type = 'base64'
                        print("[Banana_API] Found base64 data in inlineData")
                        break
            except Exception as e:
                print(f"[Banana_API] Failed to parse standard Gemini response: {e}")

        # Fallback to recursive search
        if not found_data:
            res = find_image_recursive(resp_json)
            if res:
                found_data, found_type = res
                print(f"[Banana_API] Found image via recursive search: {found_type}")

        if found_data:
            img = None
            if found_type == 'url':
                print(f"[Banana_API] Downloading image from URL: {found_data}")
                try:
                    # 设置下载图片的超时时间
                    img_resp = self.session.get(found_data, timeout=self.timeout)
                    img_resp.raise_for_status()
                    img = Image.open(BytesIO(img_resp.content))
                    print(f"[Banana_API] Successfully downloaded image")
                except Exception as e:
                    print(f"[Banana_API] Error downloading image from URL: {e}")
                    # 尝试重新下载一次
                    try:
                        img_resp = self.session.get(found_data, timeout=self.timeout)
                        img_resp.raise_for_status()
                        img = Image.open(BytesIO(img_resp.content))
                        print(f"[Banana_API] Successfully downloaded image on retry")
                    except Exception as e2:
                        print(f"[Banana_API] Error downloading image on retry: {e2}")
                        raise Exception(f"Error downloading image from URL: {e2}")
            else:
                # Base64
                try:
                    img_data = base64.b64decode(found_data)
                    img = Image.open(BytesIO(img_data))
                    print(f"[Banana_API] Successfully decoded base64 image")
                except Exception as e:
                    print(f"[Banana_API] Error decoding base64: {e}")
                    raise Exception(f"Error decoding base64: {e}")

            if img:
                # 统一转换为RGB格式，保证通道数一致
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Convert PIL to Tensor (符合ComfyUI IMAGE格式要求: [batch, H, W, C])
                img_np = np.array(img).astype(np.float32) / 255.0
                # 增加batch维度
                img_tensor = torch.from_numpy(img_np)[None, :]
                print(f"[Banana_API] Request completed in {time.time() - start_time:.2f} seconds")
                return (img_tensor,)
        
        # If we reached here, we failed to find image data
        # Check for error message in response
        error = resp_json.get("error")
        if error:
            raise Exception(f"API Error: {json.dumps(error, ensure_ascii=False)}")
        
        print(f"[Banana_API] Response structure: {json.dumps(resp_json, ensure_ascii=False)}")
        raise Exception("Could not find 'data' field or valid image URL in response. Check console logs for structure.")