# 方案A：llama-cpp-python multimodal 使用（按官方 docs 示例格式）
import base64, io
from PIL import Image
from llama_cpp import Llama


def image_to_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


model_path = r"G:/Models/MM/ImgToText/NuMarkdown-8B-Thinking-Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    # verbose=True,  # 可打开查看选用的 chat_format / 调试信息
)

data_uri = image_to_data_uri(
    r"G:/AAA重要文档、证书、照片、密钥/竞赛获奖证书 汇总备份处/中国大学生计算机设计大赛/中国大学生计算机设计大赛省级一等奖.png")

messages = [
    {"role": "system",
     "content": "You are an assistant that extracts all visible text from the image. Only output the raw text."},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": "请将图片中包含的所有文字尽可能完整准确地列出，保持原格式（不要添加其它评论）。"}
        ],
    }
]

# 设置 temperature=0 更确定性；注意增加 max_tokens 以免被截断
response = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=4096)

# 调试时先打印整个 response，查看实际返回结构
import json

print(json.dumps(response, ensure_ascii=False, indent=2))

# 常见的取法（根据你看到的 response 结构选择）
choice = response["choices"][0]
# 有的版本返回 .get("text")，有的返回 message.content
print("----可能的输出字段----")
if "text" in choice:
    print(choice["text"])
elif "message" in choice and "content" in choice["message"]:
    print(choice["message"]["content"])
else:
    # 保险打印
    print(choice)
