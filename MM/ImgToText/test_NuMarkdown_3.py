import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

model_path = r"G:/Models/MM/ImgToText/NuMarkdown-8B-Thinking-Q4_K_M.gguf"
mmproj_path = r"G:/Models/MM/ImgToText/mmproj-BF16.gguf"
image_path = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"

chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
llm = Llama(
    model_path=model_path,
    chat_handler=chat_handler,
    n_ctx=16384,
    n_gpu_layers=-1,  # 如果有 NVIDIA 显卡就填 -1，没有就填 0
    logits_all=True,
    verbose=True
)


def encode_image_to_data_uri(path):
    with open(path, "rb") as f:
        base_64_str = base64.b64encode(f.read()).decode('utf-8')  # 图片转 Base64
        return f"data:image/jpeg;base64,{base_64_str}"


data_uri = encode_image_to_data_uri(image_path)

prompt = "请用中文详细描述这幅动漫图像中的人物，包括外貌特征、衣着细节以及神态动作。"
prompt = "请用中文简要描述人物的神情"
response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }
    ],
    temperature=0.5,  # 描述性任务可以适当提高随机性，让词汇更丰富
    max_tokens=1024  # 给足空间让它描述
)

print(response["choices"][0]["message"]["content"])
# 这幅动漫图像中的主要人物是一位年轻的女性角色。她有着长长的银白色头发，头发自然垂落，有些微卷。她的发色非常纯净，几乎接近白色，给人一种非常清新和纯洁的感觉。她的眼睛大而明亮，颜色为淡蓝色，充满活力和好奇，仿佛在观察着周围的一切。她的皮肤白皙，脸型圆润，带有婴儿般的可爱特征。
#
# 她的面部表情非常生动，微微张开嘴巴，似乎在轻声地说些什么或在笑，脸颊微微红润，显得非常自然和真实。她的双手放在下巴上，手指微微弯曲，像是在思考或感到害羞。她的眼睛看向画面的左侧，显示出一种好奇和警觉的神态。
#
# 她穿着一件白色的连帽卫衣，卫衣的帽子部分被她戴在头上，帽子的边缘露出一些耳朵形状的装饰。卫衣的领口是高领设计，袖口处有松紧带，显得非常舒适和时尚。在她的头部两侧，有类似猫耳的装饰，颜色为浅粉色，边缘有蓝色的装饰，显得非常可爱。她的耳朵上还戴着一个简单的耳环，颜色为银色。
#
# 背景是粉蓝色的天空，云朵柔和，给人一种宁静和梦幻的感觉。整体色调非常柔和，主要以粉色、白色和淡蓝色为主，营造出一种温暖和美好的氛围。
