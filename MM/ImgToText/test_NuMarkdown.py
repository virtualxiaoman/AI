from llama_cpp import Llama
from PIL import Image

model_path = r"G:/Models/MM/ImgToText/NuMarkdown-8B-Thinking-Q4_K_M.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=4096,  # 可选，上下文最大 token
    # n_gpu_layers=20 # 可选，如果你想启用 GPU 部分层加速
)

path_1 = "F:/Picture/pixiv/BA/Shiroko/140349737_p0.png"
path_2 = "F:/Picture/洛天依实体卡百变洛天依/R-C.png"
path_3 = "G:/AAA重要文档、证书、照片、密钥/竞赛获奖证书 汇总备份处/中国大学生计算机设计大赛/中国大学生计算机设计大赛省级一等奖.png"
img = Image.open(path_3).convert("RGB")

# 将图片转换为 base64 URL scheme（部分 llama_cpp 需要图片 URL 或 base64）
import base64, io

buffer = io.BytesIO()
img.save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
image_data_url = f"data:image/png;base64,{b64}"

# 构造消息（image + text）
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_data_url},
            {"type": "text", "text": "图片中包含的所有文字，请尽可能完整准确地列出这些文字内容。"}
        ],
    }
]

# 执行模型推理
response = llm.create_chat_completion(messages=messages)

# 输出文本（含 markdown 结构）
print(response["choices"][0]["message"]["content"])
