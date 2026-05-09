import json
import time
import base64
import urllib3
import requests
import comfy.utils
import torch
import numpy as np
import io
from urllib.parse import quote
from PIL import Image
from io import BytesIO
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re  # 添加正则表达式模块
from ..utils import create_ssl_compatible_session
from comfy.comfy_types import IO
# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Banana2_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "draw a cat"}),
                "header_value": ("STRING", {"default": "Bearer sk-xxx", "multiline": False}),
                "url":("STRING",{"dafault":"https://mohuaai.cn/v1beta/models/gemini-3-pro-image-preview:generateContent"}),
                "model": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview", "multiline": False}),
                "aspect_ratio": ("STRING", {"default": "9:16"}),
                "image_size": ("STRING", {"default": "2K"}),
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

    def process_request(self, header_value,url, model, text, aspect_ratio, image_size,
                       image_1=None, image_2=None, image_3=None, image_4=None,
                       image_5=None, image_6=None, image_7=None, image_8=None):
        
        if not url==None:
            self.api_url=url
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
    

#veo
class ComflyVideoAdapter:
    def __init__(self, video_path_or_url, width=1280, height=720):
        self.video_path_or_url = video_path_or_url
        self.is_url = isinstance(video_path_or_url, str) and video_path_or_url.startswith(("http://", "https://"))
        self.width = width
        self.height = height

    def get_dimensions(self):
        return (self.width, self.height)

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if not self.is_url:
            raise ValueError("Unsupported video source")

        response = requests.get(self.video_path_or_url, stream=True, timeout=300, verify=False)
        response.raise_for_status()
        with open(output_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_handle.write(chunk)
        return True


class Comfly_Googel_Veo3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "baseurl": ("STRING", {"default": "https://work.poloapi.com", "multiline": False}),
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "veo3.1"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "seconds": ("INT", {"default": 8, "min": 8, "max": 8, "step": 1}),
                "apikey": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "reference_url": ("STRING", {"default": "", "multiline": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Google"

    def __init__(self):
        self.api_key = None
        self.timeout = 900
        self.session = create_ssl_compatible_session()

    def get_headers(self):
        auth_value = self.api_key
        if auth_value and auth_value.lower().startswith("bearer "):
            auth_value = auth_value.split(" ", 1)[1].strip()

        return {
            "Accept": "application/json",
            "Authorization": auth_value
        }
    
    def image_to_file(self, image_tensor):
        if image_tensor is None:
            return None

        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]

        image_np = image_tensor.detach().cpu().numpy()
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        buffered.seek(0)
        return buffered

    @staticmethod
    def aspect_ratio_to_size(aspect_ratio):
        return "1280x720" if aspect_ratio == "16:9" else "720x1280"

    @staticmethod
    def size_to_dimensions(size):
        width, height = size.split("x", 1)
        return (int(width), int(height))

    @staticmethod
    def _find_first_reference_image(*images):
        for image in images:
            if image is not None:
                return image
        return None

    @staticmethod
    def _extract_task_id(payload):
        if not isinstance(payload, dict):
            return None

        data = payload.get("data")
        return (
            payload.get("id")
            or payload.get("task_id")
            or payload.get("taskId")
            or (data.get("id") if isinstance(data, dict) else None)
            or (data.get("task_id") if isinstance(data, dict) else None)
            or (data.get("taskId") if isinstance(data, dict) else None)
        )

    @staticmethod
    def _extract_status(payload):
        if not isinstance(payload, dict):
            return ""

        data = payload.get("data")
        status = payload.get("status")
        if status:
            return str(status).lower()

        if isinstance(data, dict):
            nested_status = data.get("status")
            if nested_status:
                return str(nested_status).lower()

        return ""

    def _extract_video_url(self, payload):
        def visit(value):
            if isinstance(value, dict):
                for key in ("video_url", "url", "output", "download_url"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
                        return candidate
                for nested in value.values():
                    found = visit(nested)
                    if found:
                        return found
            elif isinstance(value, list):
                for nested in value:
                    found = visit(nested)
                    if found:
                        return found
            elif isinstance(value, str) and value.startswith(("http://", "https://")) and ".mp4" in value:
                return value
            return None

        return visit(payload)
    
    def generate_video(self, baseurl, prompt, model="veo3.1", aspect_ratio="16:9", seconds=8, apikey="",
                      reference_url="", image1=None, image2=None, image3=None):

        baseurl = (baseurl or "").strip().rstrip("/")
        self.api_key = (apikey or "").strip()

        if not baseurl:
            raise ValueError("baseurl is required")

        if not self.api_key:
            raise ValueError("apikey is required")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            size = self.aspect_ratio_to_size(aspect_ratio)
            width, height = self.size_to_dimensions(size)

            form_data = {
                "prompt": prompt,
                "model": model,
                "seconds": str(seconds),
                "size": size,
            }

            files = None
            clean_reference_url = (reference_url or "").strip()
            reference_image = self._find_first_reference_image(image1, image2, image3)
            if clean_reference_url:
                form_data["input_reference"] = clean_reference_url
            elif reference_image is not None:
                image_file = self.image_to_file(reference_image)
                files = {
                    "input_reference": ("reference.png", image_file, "image/png"),
                }

            response = self.session.post(
                f"{baseurl}/v1/videos",
                headers=self.get_headers(),
                data=form_data,
                files=files,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"API Error: {response.status_code} - {response.text}")
                
            result = response.json()
            task_id = self._extract_task_id(result)
            if not task_id:
                raise RuntimeError("No task ID returned from API")
            
            print(f"[Comfly_Googel_Veo3] Task submitted successfully. Task ID: {task_id}")
            pbar.update_absolute(30)

            max_attempts = 150 
            attempts = 0
            video_url = None
            last_status_payload = result
            status_url = f"{baseurl}/v1/videos/{quote(str(task_id), safe='')}"
            
            while attempts < max_attempts:
                time.sleep(2) 
                attempts += 1
                
                status_response = self.session.get(
                    status_url,
                    headers=self.get_headers(),
                    timeout=self.timeout
                )

                if status_response.status_code != 200:
                    progress_value = min(85, 30 + (attempts * 55 // max_attempts))
                    pbar.update_absolute(progress_value)
                    continue

                status_result = status_response.json()

                last_status_payload = status_result
                status = self._extract_status(status_result)
                video_url = self._extract_video_url(status_result)

                raw_progress = status_result.get("progress")
                if isinstance(raw_progress, int):
                    pbar.update_absolute(min(90, 30 + int(raw_progress * 60 / 100)))
                elif isinstance(raw_progress, str) and raw_progress.endswith("%"):
                    try:
                        progress_num = int(raw_progress.rstrip("%"))
                        pbar.update_absolute(min(90, 30 + int(progress_num * 60 / 100)))
                    except ValueError:
                        progress_value = min(85, 30 + (attempts * 55 // max_attempts))
                        pbar.update_absolute(progress_value)
                else:
                    progress_value = min(85, 30 + (attempts * 55 // max_attempts))
                    pbar.update_absolute(progress_value)

                if video_url and status in {"completed", "succeeded", "success", "finished"}:
                    print(f"[Comfly_Googel_Veo3] Video URL: {video_url}")
                    break

                if status in {"failed", "failure", "error", "cancelled", "canceled"}:
                    raise RuntimeError(json.dumps(status_result, ensure_ascii=False))
            
            if not video_url:
                raise RuntimeError("Failed to retrieve video URL after multiple attempts")

            pbar.update_absolute(95)
            
            response_data = {
                "code": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "seconds": seconds,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "video_url": video_url,
                "has_reference_image": reference_image is not None or bool(clean_reference_url),
                "raw_response": last_status_payload,
            }
            
            pbar.update_absolute(100)
            
            video_adapter = ComflyVideoAdapter(video_url, width=width, height=height)
            return (video_adapter, video_url, json.dumps(response_data, ensure_ascii=False))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(f"[Comfly_Googel_Veo3] {error_message}")
            raise RuntimeError(error_message)



class GeminiTextOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "baseurl":("STRING",{"default":"https://ai.comfly.chat"}),
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-3.1-flash-lite-preview"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "mohua/Google"

    def __init__(self):
        self.api_key = None
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def tensor_to_base64(self, tensor):
        if tensor is None:
            return None
        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).clamp(0, 255).byte()
        tensor = tensor.cpu()
        if tensor.shape[-1] == 3:
            img = Image.fromarray(tensor.numpy(), 'RGB')
        else:
            img = Image.fromarray(tensor.numpy(), 'RGBA')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_text(self,baseurl, prompt, model, temperature, top_p, max_tokens, seed, image=None, video=None, api_key=""):

        self.api_key=api_key
        if not self.api_key:
            return ("API key not found in Comflyapi.json",)

        try:
            content = [{"type": "text", "text": prompt}]

            if video is not None:
                video_url = getattr(video, 'video_url', None)
                if video_url:
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": video_url}
                    })
            elif image is not None:
                if len(image.shape) == 4:
                    image = image[0]
                img_b64 = self.tensor_to_base64(image)
                if img_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })

            messages = [{"role": "user", "content": content}]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed if seed > 0 else None
            }

            response = requests.post(
                f"{baseurl}/v1/chat/completions",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            return (text,)

        except Exception as e:
            return (f"Error: {str(e)}",)
        

# 将Banana2_API进行多任务异步处理，输入批次提示词同时执行。
class Banana2_API_aysn(Banana2_API):
    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        input_types["optional"] = dict(input_types["optional"])
        input_types["optional"].update({
            "prompt_delimiter": ("STRING", {"default": "\n", "multiline": False}),
            "trim_prompt": ("BOOLEAN", {"default": True}),
            "drop_empty_prompt": ("BOOLEAN", {"default": True}),
            "max_workers": ("INT", {"default": 4, "min": 1, "max": 16}),
        })
        return input_types

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "process_request_batch"
    CATEGORY = "MohuaAI/宸ュ叿"

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return float("NaN")

    def _split_batch_prompts(self, text, prompt_delimiter, trim_prompt, drop_empty_prompt):
        if isinstance(text, (list, tuple)):
            prompts = []
            for item in text:
                prompts.extend(
                    self._split_batch_prompts(
                        item,
                        prompt_delimiter,
                        trim_prompt,
                        drop_empty_prompt,
                    )
                )
            return prompts

        prompt_text = "" if text is None else str(text)
        if prompt_delimiter == "":
            prompts = [prompt_text]
        elif prompt_delimiter == "\\n":
            prompts = prompt_text.splitlines()
        else:
            prompts = prompt_text.split(prompt_delimiter)

        if trim_prompt:
            prompts = [prompt.strip() for prompt in prompts]
        if drop_empty_prompt:
            prompts = [prompt for prompt in prompts if prompt]
        return prompts

    def _process_one_prompt(self, index, task_prompt, kwargs):
        worker = Banana2_API()
        image_tensor = worker.process_request(text=task_prompt, **kwargs)[0]
        return index, task_prompt, image_tensor

    def _blank_image(self):
        img = Image.new("RGB", (1024, 1024), color="white")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)[None, :]

    def process_request_batch(
        self, header_value, url, model, text, aspect_ratio, image_size,
        image_1=None, image_2=None, image_3=None, image_4=None,
        image_5=None, image_6=None, image_7=None, image_8=None,
        prompt_delimiter="\n", trim_prompt=True, drop_empty_prompt=True, max_workers=4,
    ):
        prompts = self._split_batch_prompts(
            text,
            prompt_delimiter,
            trim_prompt,
            drop_empty_prompt,
        )
        if not prompts:
            msg = "No valid prompts for Banana2_API_aysn"
            print(msg)
            return (self._blank_image(), msg)

        common_kwargs = {
            "header_value": header_value,
            "url": url,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "image_1": image_1,
            "image_2": image_2,
            "image_3": image_3,
            "image_4": image_4,
            "image_5": image_5,
            "image_6": image_6,
            "image_7": image_7,
            "image_8": image_8,
        }

        workers = min(max_workers, len(prompts))
        pbar = comfy.utils.ProgressBar(len(prompts))
        results = [None] * len(prompts)

        try:
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._process_one_prompt, index, task_prompt, common_kwargs)
                    for index, task_prompt in enumerate(prompts)
                ]

                completed = 0
                for future in as_completed(futures):
                    index, task_prompt, image_tensor = future.result()
                    results[index] = (task_prompt, image_tensor)
                    completed += 1
                    pbar.update_absolute(completed)

            tensors = []
            response_parts = [
                "**Banana2_API batch async**",
                f"Prompt Count: {len(prompts)}",
                f"Workers: {workers}",
                f"Elapsed: {time.time() - start_time:.2f}s",
                "",
            ]

            for index, item in enumerate(results, start=1):
                task_prompt, image_tensor = item
                if image_tensor is not None:
                    tensors.append(image_tensor)
                response_parts.append(f"--- Prompt {index} ---")
                response_parts.append(f"Prompt: {task_prompt}")

            if not tensors:
                msg = "No images decoded from Banana2_API_aysn response"
                print(msg)
                return (self._blank_image(), msg)

            return (torch.cat(tensors, dim=0), "\n".join(response_parts))

        except Exception as e:
            error_message = f"Banana2_API_aysn error: {str(e)}"
            import traceback
            print(traceback.format_exc())
            print(error_message)
            return (self._blank_image(), error_message)
