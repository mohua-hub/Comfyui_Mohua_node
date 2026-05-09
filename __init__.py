from .nodes.nodes_google import *
from .nodes.tools import *
from .nodes.nodes_openai import *


NODE_CLASS_MAPPINGS = {
    "Banana_pro_MohuaAI": Banana2_API,
    "Banana2_API_aysn_MohuaAI": Banana2_API_aysn,
    "Comfly_Googel_Veo3_MohuaAI": Comfly_Googel_Veo3,
    "LoadImagesMulti_MohuaAI": LoadImagesMulti,
    "LoadImagesMultibyURL_MohuaAI": LoadImagesMultibyURL,
    "ProcessString_MohuaAI": ProcessString,
    "TextSplitBatch_MohuaAI": TextSplitBatch,
    "gpt_image_2_MohuaAI": gpt_image_2,
    "gpt_image2_Asyn_MohuaAI": gpt_image2_Asyn,
    "GeminiTextOnly_MohuaAI": GeminiTextOnly
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Banana_pro_MohuaAI": "Banana_pro_MohuaAI",
    "Banana2_API_aysn_MohuaAI": "Banana2_API_aysn_MohuaAI",
    "Comfly_Googel_Veo3_MohuaAI": "Comfly_Googel_Veo3_MohuaAI",
    "ProcessString_MohuaAI": "ProcessString_MohuaAI",
    "LoadImagesMulti_MohuaAI": "LoadImagesMulti_MohuaAI",
    "LoadImagesMultibyURL_MohuaAI": "LoadImagesMultibyURL_MohuaAI",
    "TextSplitBatch_MohuaAI": "TextSplitBatch_MohuaAI",
    "gpt_image_2_MohuaAI":"gpt_image_2_MohuaAI",
    "gpt_image2_Asyn_MohuaAI":"gpt_image2_Asyn_MohuaAI",
    "GeminiTextOnly_MohuaAI":"GeminiTextOnly_MohuaAI"
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
