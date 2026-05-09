import os
import re
from io import BytesIO

import folder_paths
import node_helpers
import numpy as np
import requests
import torch
from PIL import Image, ImageOps, ImageSequence

class LoadImagesMulti:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return {
            "required": {
                "filenames": ("STRING", {
                    "default": "filename1.png\nfilename2.png",
                    "tooltip": "输入多个文件名，用逗号或者换行分隔，例如： 1.png, 2.jpg, dir/sub.png",
                    "multiline": True  # 多行文本域
                }),
            }
        }

    CATEGORY = "Mohua"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING",
                    "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "masks", "filepaths",
                    "image1", "image2", "image3", "image4", "image5", "image6")
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,True,False,
                      False,False,False,False,False,False)
    FUNCTION = "load_images"

    def load_images(self, filenames):
        # 解析用户输入的多个文件名

        filenames = re.split(r'[, \r\n]+', filenames)  # 按逗号、空格或任何换行符分割
        filenames = [f.strip() for f in filenames if f.strip()]  # 去掉首尾空格和空字符串

        output_images = []
        output_masks = []
        output_paths = []
        single_images = []
        if len(filenames) == 0:
            return (output_images, output_masks, "\n".join(output_paths),
                *single_images)
        excluded_formats = ["MPO"]

        for fname in filenames:
            # 支持子目录，如 "sub/my.png"
            img_path = folder_paths.get_annotated_filepath(fname)

            if not folder_paths.exists_annotated_filepath(fname):
                raise FileNotFoundError(f"文件不存在: {fname}")

            img = node_helpers.pillow(Image.open, img_path)

            # frames_img = []
            # frames_mask = []

            w, h = None, None

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == "I":
                    i = i.point(lambda x: x * (1 / 255))

                rgb = i.convert("RGB")

                # 统一尺寸
                if w is None:
                    w, h = rgb.size
                elif rgb.size != (w, h):
                    continue

                # 转 tensor
                rgb_tensor = torch.from_numpy(
                    np.array(rgb).astype(np.float32) / 255.0
                )[None,]

                # Mask
                if "A" in i.getbands():
                    alpha = i.getchannel("A")
                    mask_np = np.array(alpha).astype(np.float32) / 255.0
                    mask_tensor = 1. - torch.from_numpy(mask_np)
                elif i.mode == "P" and "transparency" in i.info:
                    alpha = i.convert("RGBA").getchannel("A")
                    mask_np = np.array(alpha).astype(np.float32) / 255.0
                    mask_tensor = 1. - torch.from_numpy(mask_np)
                else:
                    mask_tensor = torch.zeros((64, 64), dtype=torch.float32)

                output_images.append(rgb_tensor)
                output_masks.append(mask_tensor.unsqueeze(0))

            # if len(frames_img) > 1 and img.format not in excluded_formats:
            #     image_tensor = torch.cat(frames_img, dim=0)
            #     mask_tensor = torch.cat(frames_mask, dim=0)
            # else:
            #     image_tensor = frames_img[0]
            #     mask_tensor = frames_mask[0]

            # output_images.append(frames_img)
            # output_masks.append(frames_mask)
            output_paths.append(img_path)

        # 合并为 batch（N, H, W, C）
        # batch_images = output_images  # 保持 list，每个元素是不同尺寸的 tensor
        # batch_masks = output_masks
        # 前6张单图输出（如果不够就用None占位）
        single_images = [output_images[i] if i < len(output_images) else None for i in range(6)]

        return (output_images, output_masks, "\n".join(output_paths),
                *single_images)



class ProcessString:
    OPTIONS = [
        "不改变",
        "取数字",
        "取字母",
        "转大写",
        "转小写",
        "取中文",
        "去标点",
        "去换行",
        "去空行",
        "去空格",
        "去格式",
        "统计字数",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": True, "default": ""}),
                "option": (cls.OPTIONS, {"default": "不改变"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_string"
    CATEGORY = "Mohua_tools"
    DESCRIPTION = "字符串处理节点，支持提取数字、字母、中文以及常见清理操作。"

    @staticmethod
    def IS_CHANGED():
        return float("NaN")

    def process_string(self, input_string, option):
        if option == "取数字":
            result = "".join(re.findall(r"\d", input_string))
        elif option == "取字母":
            result = "".join(filter(lambda char: char.isalpha() and not self.is_chinese(char), input_string))
        elif option == "转大写":
            result = input_string.upper()
        elif option == "转小写":
            result = input_string.lower()
        elif option == "取中文":
            result = "".join(filter(self.is_chinese, input_string))
        elif option == "去标点":
            result = re.sub(r"[^\w\s\u4e00-\u9fff]", "", input_string)
        elif option == "去换行":
            result = input_string.replace("\n", "")
        elif option == "去空行":
            result = "\n".join(filter(lambda line: line.strip(), input_string.splitlines()))
        elif option == "去空格":
            result = input_string.replace(" ", "")
        elif option == "去格式":
            result = re.sub(r"\s+", "", input_string)
        elif option == "统计字数":
            result = str(len(input_string))
        else:
            result = input_string

        return (result,)

    @staticmethod
    def is_chinese(char):
        return "\u4e00" <= char <= "\u9fff"


class TextSplitBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": True, "default": ""}),
                "split_mode": (["delimiter", "fixed_length"], {"default": "delimiter"}),
                "delimiter": ("STRING", {"default": "\n", "multiline": False}),
                "delimiter_mode": (["plain", "regex"], {"default": "plain"}),
                "chunk_length": ("INT", {"default": 10, "min": 1, "step": 1}),
                "trim_result": ("BOOLEAN", {"default": True}),
                "drop_empty": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text_list", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split_text"
    CATEGORY = "Mohua_tools"
    DESCRIPTION = "将输入文本按分隔符或固定长度切分，并以 STRING 列表批次输出。"

    @staticmethod
    def IS_CHANGED():
        return float("NaN")

    def split_text(
        self,
        input_string,
        split_mode,
        delimiter,
        delimiter_mode,
        chunk_length,
        trim_result,
        drop_empty,
    ):
        if not input_string:
            return ([], 0)

        if split_mode == "delimiter":
            segments = self._split_by_delimiter(input_string, delimiter, delimiter_mode)
        elif split_mode == "fixed_length":
            segments = self._split_by_fixed_length(input_string, chunk_length)
        else:
            raise ValueError(f"Unsupported split_mode: {split_mode}")

        segments = self._normalize_segments(segments, trim_result, drop_empty)
        return (segments, len(segments))

    def _split_by_delimiter(self, input_string, delimiter, delimiter_mode):
        if delimiter == "":
            return [input_string]

        if delimiter_mode == "regex":
            return re.split(delimiter, input_string)

        return input_string.split(delimiter)

    def _split_by_fixed_length(self, input_string, chunk_length):
        if chunk_length <= 0:
            raise ValueError("chunk_length must be greater than 0")

        return [input_string[index:index + chunk_length] for index in range(0, len(input_string), chunk_length)]

    @staticmethod
    def _normalize_segments(segments, trim_result, drop_empty):
        if trim_result:
            segments = [segment.strip() for segment in segments]

        if drop_empty:
            segments = [segment for segment in segments if segment]

        return segments


# 通过url下载后返回
class LoadImagesMultibyURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {
                    "default": "https://example.com/image1.png\nhttps://example.com/image2.jpg",
                    "tooltip": "输入多个图片 URL，支持逗号、空格或换行分隔",
                    "multiline": True,
                }),
            },
            "optional": {
                "timeout": ("INT", {"default": 30, "min": 1, "max": 300}),
                "user_agent": ("STRING", {
                    "default": "Mozilla/5.0",
                    "multiline": False,
                }),
            },
        }

    CATEGORY = "Mohua"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING",
                    "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "masks", "urls",
                    "image1", "image2", "image3", "image4", "image5", "image6")
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True, True, False,
                      False, False, False, False, False, False)
    FUNCTION = "load_images_by_url"

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return float("NaN")

    @staticmethod
    def _split_urls(urls):
        if isinstance(urls, (list, tuple)):
            output = []
            for item in urls:
                output.extend(LoadImagesMultibyURL._split_urls(item))
            return output
        return [u.strip() for u in re.split(r'[, \r\n]+', str(urls or "")) if u.strip()]

    @staticmethod
    def _image_frame_to_tensors(frame):
        frame = ImageOps.exif_transpose(frame)

        if frame.mode == "I":
            frame = frame.point(lambda x: x * (1 / 255))

        rgb = frame.convert("RGB")
        rgb_tensor = torch.from_numpy(
            np.array(rgb).astype(np.float32) / 255.0
        )[None,]

        if "A" in frame.getbands():
            alpha = frame.getchannel("A")
            mask_np = np.array(alpha).astype(np.float32) / 255.0
            mask_tensor = 1. - torch.from_numpy(mask_np)
        elif frame.mode == "P" and "transparency" in frame.info:
            alpha = frame.convert("RGBA").getchannel("A")
            mask_np = np.array(alpha).astype(np.float32) / 255.0
            mask_tensor = 1. - torch.from_numpy(mask_np)
        else:
            mask_tensor = torch.zeros((rgb.height, rgb.width), dtype=torch.float32)

        return rgb_tensor, mask_tensor.unsqueeze(0)

    def load_images_by_url(self, urls, timeout=30, user_agent="Mozilla/5.0"):
        url_list = self._split_urls(urls)
        output_images = []
        output_masks = []
        output_urls = []

        if len(url_list) == 0:
            return (output_images, output_masks, "", None, None, None, None, None, None)

        headers = {}
        if user_agent:
            headers["User-Agent"] = user_agent

        session = requests.Session()

        for url in url_list:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            img = node_helpers.pillow(Image.open, BytesIO(response.content))
            w, h = None, None

            for frame in ImageSequence.Iterator(img):
                rgb_tensor, mask_tensor = self._image_frame_to_tensors(frame)
                frame_h, frame_w = rgb_tensor.shape[1], rgb_tensor.shape[2]

                if w is None:
                    w, h = frame_w, frame_h
                elif (frame_w, frame_h) != (w, h):
                    continue

                output_images.append(rgb_tensor)
                output_masks.append(mask_tensor)

            output_urls.append(url)

        single_images = [output_images[i] if i < len(output_images) else None for i in range(6)]

        return (output_images, output_masks, "\n".join(output_urls), *single_images)
