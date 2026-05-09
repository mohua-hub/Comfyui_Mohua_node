import re
import requests
import time
from comfy.comfy_types import IO
from PIL import Image
import io
from io import BytesIO
from comfy.utils import common_upscale, ProgressBar
import torch
from ..utils import pil2tensor, tensor2pil, ComflyVideoAdapter
import numpy as np
import base64
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


def downscale_input(image):
    samples = image.movedim(-1,1)

    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s



class gpt_image_2:

    _SIZE_MAP = {
        # 1:1
        ("1:1", "1K"): "1024x1024",
        ("1:1", "2K"): "2048x2048",
        ("1:1", "4K"): "2880x2880",

        # 16:9
        ("16:9", "1K"): "1280x720",
        ("16:9", "2K"): "2560x1440",
        ("16:9", "4K"): "3840x2160",

        # 9:16
        ("9:16", "1K"): "720x1280",
        ("9:16", "2K"): "1440x2560",
        ("9:16", "4K"): "2160x3840",

        # 4:3
        ("4:3", "1K"): "1152x864",
        ("4:3", "2K"): "2304x1728",
        ("4:3", "4K"): "3264x2448",

        # 3:4
        ("3:4", "1K"): "864x1152",
        ("3:4", "2K"): "1728x2304",
        ("3:4", "4K"): "2448x3264",

        # 3:2
        ("3:2", "1K"): "1248x832",
        ("3:2", "2K"): "2496x1664",
        ("3:2", "4K"): "3504x2336",

        # 2:3
        ("2:3", "1K"): "832x1248",
        ("2:3", "2K"): "1664x2496",
        ("2:3", "4K"): "2336x3504",

        # 5:4
        ("5:4", "1K"): "1120x896",
        ("5:4", "2K"): "2240x1792",
        ("5:4", "4K"): "3200x2560",

        # 4:5
        ("4:5", "1K"): "896x1120",
        ("4:5", "2K"): "1792x2240",
        ("4:5", "4K"): "2560x3200",

        # 21:9
        ("21:9", "1K"): "1456x624",
        ("21:9", "2K"): "3024x1296",
        ("21:9", "4K"): "3696x1584",

        # 9:21
        ("9:21", "1K"): "624x1456",
        ("9:21", "2K"): "1296x3024",
        ("9:21", "4K"): "1584x3696",

        # 2:1
        ("2:1", "1K"): "2048x1024",
        ("2:1", "2K"): "2688x1344",
        ("2:1", "4K"): "3840x1920",

        # 1:2
        ("1:2", "1K"): "1024x2048",
        ("1:2", "2K"): "1344x2688",
        ("1:2", "4K"): "1920x3840",
    }

    @staticmethod
    def _parse_size_wh(size_str):
        m = re.match(r"^(\d+)x(\d+)$", size_str.strip())
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    @classmethod
    def _validate_gpt_image2_size(cls, size_str):
        """
        gpt-image-2: long edge <= 3840; aspect <= 3:1;
        total pixels in [655360, 8294400].
        """
        if size_str == "auto":
            return True, None
        w, h = cls._parse_size_wh(size_str)
        if w is None:
            return False, "size 格式须为 宽x高，例如 1024x1024"
        if max(w, h) > 3840:
            return False, "长边须 <= 3840px"
        lo, hi = min(w, h), max(w, h)
        if hi / lo > 3.0 + 1e-9:
            return False, "长边:短边 不得超过 3:1"
        px = w * h
        if px < 655360 or px > 8294400:
            return False, "总像素须在 655,360～8,294,400 之间"
        return True, None

    @classmethod
    def _get_size_from_params(cls, aspect_ratio, resolution):
        """根据 aspect_ratio 和 resolution 获取实际的 size"""
        size = cls._SIZE_MAP.get((aspect_ratio, resolution))
        if size is None:
            return None, f"不支持的组合: {aspect_ratio} × {resolution}。"
        return size, None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "baseurl": ("STRING",{"default":"https://ai.comfly.chat"}),
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "resolution": ("STRING", {"default": "1K"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "mask": ("MASK",),
                "api_key": ("STRING", {"default": ""}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "background": (["auto", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "async_mode": ("BOOLEAN", {"default": True}),
                "webhook": ("STRING", {"default": ""}),
                "max_poll_attempts": ("INT", {"default": 300, "min": 10, "max": 1000}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 60}),
                "max_retries": ("INT", {"default": 5, "min": 1, "max": 10}),
                "initial_timeout": ("INT", {"default": 900, "min": 60, "max": 1200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "Comfly/Openai"

    def __init__(self):

        self.api_key = None
        self.timeout = 600
        self.session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _auth_headers_bearer(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _headers_json(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def make_request_with_retry(self, url, data=None, files=None, max_retries=5, initial_timeout=300):
        for attempt in range(1, max_retries + 1):
            current_timeout = min(initial_timeout * (1.5 ** (attempt - 1)), 1200)
            try:
                if files is not None:
                    response = self.session.post(
                        url,
                        headers=self._auth_headers_bearer(),
                        data=data,
                        files=files,
                        timeout=current_timeout
                    )
                else:
                    response = self.session.post(
                        url,
                        headers=self._headers_json(),
                        json=data,
                        timeout=current_timeout
                    )
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout:
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))
            except requests.exceptions.ConnectionError:
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (400, 401, 403):
                    raise
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))

    def get_headers_multipart(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _blank_input_file(self):
        buf = io.BytesIO()
        Image.new("RGB", (1024, 1024), color="white").save(buf, format="PNG")
        buf.seek(0)
        return ("blank.png", buf, "image/png")

    def _build_official_edits_multipart(
        self, prompt, image1, image2, image3, image4, image5, mask, n, quality, size, background,
        output_format, output_compression, moderation
    ):

        input_images = []
        for img in [image1, image2, image3, image4, image5]:
            if img is not None:
                input_images.append(img)
        
        if mask is not None and len(input_images) == 0:
            raise Exception("使用 mask 时必须提供至少一个 input image")

        files = {}

        if len(input_images) == 0:
            files["image"] = self._blank_input_file()
            total_images = 1
        else:
            image_list = []
            for img_tensor in input_images:
                batch_size = img_tensor.shape[0]
                for i in range(batch_size):
                    single_image = img_tensor[i : i + 1]
                    scaled_image = downscale_input(single_image).squeeze()
                    image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(image_np)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format="PNG")
                    img_byte_arr.seek(0)
                    image_list.append(("image_{}.png".format(len(image_list)), img_byte_arr, "image/png"))
            
            total_images = len(image_list)
            
            if total_images == 1:
                files["image"] = image_list[0]
            else:
                files["image[]"] = image_list

        if mask is not None:
            if total_images != 1:
                raise Exception("Mask requires exactly one input image")
            first_img = input_images[0]
            if mask.shape[1:] != first_img.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            _batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
            scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()
            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_byte_arr = io.BytesIO()
            mask_img.save(mask_byte_arr, format="PNG")
            mask_byte_arr.seek(0)
            files["mask"] = ("mask.png", mask_byte_arr, "image/png")

        data = {
            "prompt": prompt,
            "model": "gpt-image-2",
            "n": str(n),
            "quality": quality,
            "moderation": moderation,
            "size": size,  
        }
        if background != "auto":
            data["background"] = background
        if output_compression != 100:
            data["output_compression"] = str(output_compression)
        if output_format != "png":
            data["output_format"] = output_format

        if "image[]" in files:
            request_files = []
            for file_tuple in files["image[]"]:
                request_files.append(("image", file_tuple))
            if "mask" in files:
                request_files.append(("mask", files["mask"]))
        else:
            request_files = []
            if "image" in files:
                request_files.append(("image", files["image"]))
            if "mask" in files:
                request_files.append(("mask", files["mask"]))

        return data, request_files

    def _decode_b64_url_one(self, b64_json, image_url, max_retries, initial_timeout):
        """One image entry to tensor or None."""
        if b64_json:
            b64_data = b64_json
            if b64_data.startswith("data:image"):
                b64_data = b64_data.split(",", 1)[-1]
            elif b64_data.startswith("data:image/png;base64,"):
                b64_data = b64_data[len("data:image/png;base64,") :]
            image_data = base64.b64decode(b64_data)
            pil_img = Image.open(BytesIO(image_data))
            return pil2tensor(pil_img)
        if image_url:
            for download_attempt in range(1, max_retries + 1):
                try:
                    img_response = requests.get(
                        image_url,
                        timeout=min(initial_timeout * (1.5 ** (download_attempt - 1)), 900),
                    )
                    img_response.raise_for_status()
                    pil_img = Image.open(BytesIO(img_response.content))
                    return pil2tensor(pil_img)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                    if download_attempt == max_retries:
                        return None
                    time.sleep(min(2 ** (download_attempt - 1), 60))
        return None

    def _async_official(
        self,
        baseurl,
        prompt,
        image1,
        image2,
        image3,
        image4,
        image5,
        mask,
        pbar,
        max_poll_attempts,
        poll_interval,
        webhook,
        n,
        quality,
        size,
        background,
        output_format,
        output_compression,
        moderation,
        max_retries,
        initial_timeout,
    ):
        data, request_files = self._build_official_edits_multipart(
            prompt, image1, image2, image3, image4, image5, mask, n, quality, size, background,
            output_format, output_compression, moderation,
        )
        url = f"{baseurl}/v1/images/edits?async=true"
        if webhook.strip():
            url += f"&webhook={webhook.strip()}"

        pbar.update_absolute(10)
        response = requests.post(
            url,
            headers=self.get_headers_multipart(),
            data=data,
            files=request_files,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API Error: {response.status_code} - {response.text}")

        submit_result = response.json()
        task_id = submit_result.get("task_id") or submit_result.get("data")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {submit_result}")

        print(f"Task submitted. Task ID: {task_id}")
        pbar.update_absolute(20)

        query_url = f"{baseurl}/v1/images/tasks/{task_id}"
        final_result = None
        image_url_first = ""

        for attempts in range(1, max_poll_attempts + 1):
            time.sleep(poll_interval)
            try:
                status_response = requests.get(
                    query_url, headers=self.get_headers_multipart(), timeout=self.timeout
                )
                if status_response.status_code != 200:
                    print(f"Status check failed: {status_response.status_code}")
                    continue
                status_data = status_response.json()
                inner = status_data.get("data", {}) if isinstance(status_data, dict) else {}
                status = inner.get("status", "")
                progress_str = inner.get("progress", "0%")
                try:
                    if isinstance(progress_str, str) and progress_str.endswith("%"):
                        progress_value = int(progress_str[:-1])
                        pbar_value = min(95, 20 + int(progress_value * 0.75))
                        pbar.update_absolute(pbar_value)
                except (ValueError, AttributeError):
                    pass

                if status == "SUCCESS":
                    result_data = inner.get("data", {})
                    data_array = (
                        result_data.get("data", []) if isinstance(result_data, dict) else []
                    )
                    tensors = []
                    for item in data_array or []:
                        u = item.get("url", "") or ""
                        bj = item.get("b64_json", "") or ""
                        if u and not image_url_first:
                            image_url_first = u
                        t = self._decode_b64_url_one(
                            bj, u, max_retries, initial_timeout
                        )
                        if t is not None:
                            tensors.append(t)
                    if not tensors:
                        raise RuntimeError("Async task SUCCESS but no decodable image in data")
                    final_result = status_data
                    combined = torch.cat(tensors, dim=0)
                    pbar.update_absolute(100)
                    return (combined, image_url_first, task_id, final_result)
                if status == "FAILURE":
                    fail_reason = inner.get("fail_reason", "Unknown error")
                    raise RuntimeError(f"Task failed: {fail_reason}")
            except RuntimeError:
                raise
            except Exception as e:
                print(f"Error polling task status: {str(e)}")
        raise RuntimeError(f"Failed to get image after {max_poll_attempts} poll attempts")

    def _items_to_tensors(self, result, max_retries=5, initial_timeout=300):
        """Parse Images API data[] b64_json or url into a list of tensors."""
        out = []
        for item in result.get("data", []) or []:
            if "b64_json" in item and item["b64_json"]:
                b64_data = item["b64_json"]
                if b64_data.startswith("data:image"):
                    b64_data = b64_data.split(",", 1)[-1]
                elif b64_data.startswith("data:image/png;base64,"):
                    b64_data = b64_data[len("data:image/png;base64,"):]
                image_data = base64.b64decode(b64_data)
                pil_img = Image.open(BytesIO(image_data))
                out.append(pil2tensor(pil_img))
            elif "url" in item and item["url"]:
                for download_attempt in range(1, max_retries + 1):
                    try:
                        img_response = requests.get(
                            item["url"],
                            timeout=min(initial_timeout * (1.5 ** (download_attempt - 1)), 900)
                        )
                        img_response.raise_for_status()
                        pil_img = Image.open(BytesIO(img_response.content))
                        out.append(pil2tensor(pil_img))
                        break
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                        if download_attempt == max_retries:
                            break
                        time.sleep(min(2 ** (download_attempt - 1), 60))
        return out

    def _edits(
        self,baseurl, prompt, image1, image2, image3, image4, image5, mask, n, quality, size, background,
        output_format, output_compression, moderation, max_retries, initial_timeout, pbar
    ):
        data, request_files = self._build_official_edits_multipart(
            prompt, image1, image2, image3, image4, image5, mask, n, quality, size, background,
            output_format, output_compression, moderation,
        )
        pbar.update_absolute(20)
        response = self.make_request_with_retry(
            f"{baseurl}/v1/images/edits",
            data=data,
            files=request_files,
            max_retries=max_retries,
            initial_timeout=initial_timeout,
        )
        pbar.update_absolute(60)
        return response.json()

    def generate(
        self, prompt,baseurl, aspect_ratio="1:1", resolution="1k", image1=None, image2=None, 
        image3=None, image4=None, image5=None, mask=None, api_key="",
        n=1, quality="auto", background="auto",
        output_format="png", output_compression=100, moderation="auto",
        async_mode=True, webhook="", max_poll_attempts=300, poll_interval=5,
        max_retries=5, initial_timeout=900, seed=0
    ):

        blank = Image.new('RGB', (1024, 1024), color='white')
        blank_t = pil2tensor(blank)

        self.api_key= api_key
        if not self.api_key:
            msg = "API key not found in Comflyapi.json"
            print(msg)
            return (blank_t, "", msg)

        size, error_msg = self._get_size_from_params(aspect_ratio, resolution)
        if error_msg:
            print(error_msg)
            return (blank_t, "", error_msg)

        input_images = [img for img in [image1, image2, image3, image4, image5] if img is not None]
        num_input_images = len(input_images)

        pbar = pbar = ProgressBar(100)
        pbar.update_absolute(5)
        def _info_common(mode_line):
            s = f"**Comfly gpt-image-2 (official)** {mode_line}\n"
            s += f"Model: gpt-image-2\n"
            s += f"Prompt: {prompt}\n"
            s += f"Aspect Ratio: {aspect_ratio}\n"
            s += f"Resolution: {resolution}\n"
            s += f"Actual Size: {size}\n"
            s += f"Quality: {quality}\n"
            s += f"Input Images: {num_input_images}\n"
            w, h = self._parse_size_wh(size)
            if w is not None and h is not None and w * h > 2560 * 1440:
                s += "（总像素大于约 2560×1440，文档中视为实验性输出）\n"
            if background != "auto":
                s += f"Background: {background}\n"
            s += f"Output: {output_format}\n"
            return s

        try:
            ok, err_msg = self._validate_gpt_image2_size(size)
            if not ok:
                print(err_msg)
                return (blank_t, "", err_msg)

            if async_mode:
                combined, image_url, task_id, final_result = self._async_official(
                    baseurl,
                    prompt,
                    image1,
                    image2,
                    image3,
                    image4,
                    image5,
                    mask,
                    pbar,
                    max_poll_attempts,
                    poll_interval,
                    webhook,
                    n,
                    quality,
                    size,
                    background,
                    output_format,
                    output_compression,
                    moderation,
                    max_retries,
                    initial_timeout,
                )
                mode = "async: POST /v1/images/edits?async=true, GET /v1/images/tasks/{task_id}"
                info = _info_common(mode)
                info += f"Task ID: {task_id}\n"
                if image_url:
                    info += f"Image URL: {image_url}\n"
                if final_result:
                    inner = final_result.get("data", {})
                    inner_data = inner.get("data", {}) if isinstance(inner, dict) else {}
                    if (
                        isinstance(inner_data, dict)
                        and "usage" in inner_data
                    ):
                        usage = inner_data["usage"]
                        info += f"Total Tokens: {usage.get('total_tokens', 'N/A')}\n"
                return (combined, image_url or "", info)

            result = self._edits(
                baseurl,prompt, image1, image2, image3, image4, image5, mask, n, quality, size, background,
                output_format, output_compression, moderation,
                max_retries, initial_timeout, pbar
            )
            mode = "sync: /v1/images/edits (multipart" + (
                ", blank ref" if num_input_images == 0 else f", {num_input_images} images"
            ) + (", mask" if mask is not None else "") + ")"

            if "data" not in result or not result["data"]:
                msg = f"No image data in response: {result}"
                print(msg)
                return (blank_t, "", msg)

            tensors = self._items_to_tensors(result, max_retries, initial_timeout)
            pbar.update_absolute(95)

            if not tensors:
                msg = "No images decoded from response"
                print(msg)
                return (blank_t, "", msg)

            combined = torch.cat(tensors, dim=0)
            pbar.update_absolute(100)

            info = _info_common(mode)
            if "usage" in result:
                u = result["usage"]
                if isinstance(u, dict):
                    if "total_tokens" in u:
                        info += f"Total tokens: {u['total_tokens']}\n"
                    if "input_tokens" in u:
                        info += f"Input tokens: {u['input_tokens']}\n"
                    if "output_tokens" in u:
                        info += f"Output tokens: {u['output_tokens']}\n"

            return (combined, "", info)

        except Exception as e:
            error_message = f"Comfly_gpt_image_2_official error: {str(e)}"
            import traceback
            print(traceback.format_exc())
            print(error_message)
            return (blank_t, "", error_message)
        

# 将gpt_image2进行多任务异步处理，输入批次提示词同时执行。
class gpt_image2_Asyn(gpt_image_2):
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
        input_types["optional"]["async_mode"] = ("BOOLEAN", {"default": True})
        return input_types

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate_batch"
    CATEGORY = "Comfly/Openai"

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return float("NaN")

    def _split_batch_prompts(self, prompt, prompt_delimiter, trim_prompt, drop_empty_prompt):
        if isinstance(prompt, (list, tuple)):
            prompts = []
            for item in prompt:
                prompts.extend(
                    self._split_batch_prompts(
                        item,
                        prompt_delimiter,
                        trim_prompt,
                        drop_empty_prompt,
                    )
                )
            return prompts

        text = "" if prompt is None else str(prompt)
        if prompt_delimiter == "":
            prompts = [text]
        elif prompt_delimiter == "\\n":
            prompts = text.splitlines()
        else:
            prompts = text.split(prompt_delimiter)

        if trim_prompt:
            prompts = [p.strip() for p in prompts]
        if drop_empty_prompt:
            prompts = [p for p in prompts if p]
        return prompts

    def _generate_one_prompt(self, index, task_prompt, kwargs):
        worker = gpt_image_2()
        result = worker.generate(prompt=task_prompt, **kwargs)
        return index, task_prompt, result

    def generate_batch(
        self, prompt, baseurl, aspect_ratio="1:1", resolution="1k", image1=None, image2=None,
        image3=None, image4=None, image5=None, mask=None, api_key="",
        n=1, quality="auto", background="auto",
        output_format="png", output_compression=100, moderation="auto",
        async_mode=True, webhook="", max_poll_attempts=300, poll_interval=5,
        max_retries=5, initial_timeout=900, seed=0,
        prompt_delimiter="\n", trim_prompt=True, drop_empty_prompt=True, max_workers=4
    ):
        blank = Image.new("RGB", (1024, 1024), color="white")
        blank_t = pil2tensor(blank)

        prompts = self._split_batch_prompts(
            prompt,
            prompt_delimiter,
            trim_prompt,
            drop_empty_prompt,
        )
        if not prompts:
            msg = "No valid prompts for gpt_image2_Asyn"
            print(msg)
            return (blank_t, "", msg)

        common_kwargs = {
            "baseurl": baseurl,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "image1": image1,
            "image2": image2,
            "image3": image3,
            "image4": image4,
            "image5": image5,
            "mask": mask,
            "api_key": api_key,
            "n": n,
            "quality": quality,
            "background": background,
            "output_format": output_format,
            "output_compression": output_compression,
            "moderation": moderation,
            "async_mode": async_mode,
            "webhook": webhook,
            "max_poll_attempts": max_poll_attempts,
            "poll_interval": poll_interval,
            "max_retries": max_retries,
            "initial_timeout": initial_timeout,
            "seed": seed,
        }

        workers = min(max_workers, len(prompts))
        pbar = ProgressBar(len(prompts))
        results = [None] * len(prompts)

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._generate_one_prompt, index, task_prompt, common_kwargs)
                    for index, task_prompt in enumerate(prompts)
                ]
                completed = 0
                for future in as_completed(futures):
                    index, task_prompt, result = future.result()
                    results[index] = (task_prompt, result)
                    completed += 1
                    pbar.update_absolute(completed)

            tensors = []
            urls = []
            response_parts = [
                "**Comfly gpt-image-2 batch async**",
                f"Prompt Count: {len(prompts)}",
                f"Workers: {workers}",
                "",
            ]

            for index, item in enumerate(results, start=1):
                task_prompt, result = item
                image_tensor, image_url, response_text = result
                if image_tensor is not None:
                    tensors.append(image_tensor)
                if image_url:
                    urls.append(image_url)
                response_parts.append(f"--- Prompt {index} ---")
                response_parts.append(f"Prompt: {task_prompt}")
                response_parts.append(response_text or "")

            if not tensors:
                msg = "No images decoded from batch response"
                print(msg)
                return (blank_t, "\n".join(urls), msg)

            return (torch.cat(tensors, dim=0), "\n".join(urls), "\n".join(response_parts))

        except Exception as e:
            error_message = f"gpt_image2_Asyn error: {str(e)}"
            import traceback
            print(traceback.format_exc())
            print(error_message)
            return (blank_t, "", error_message)
