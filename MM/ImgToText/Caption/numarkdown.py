import base64
from typing import Optional, List, Dict, Any

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler


class ImageToTextLLM:
    """
    使用 GGUF 多模态模型完成：
        输入：图片路径
        输出：文本描述
    """

    def __init__(self, model_path: str, mmproj_path: str, n_ctx: int = 32768, n_gpu_layers: int = -1,
                 temperature: float = 0.3, max_tokens: int = 512, verbose: bool = False):
        self.temperature = temperature
        self.max_tokens = max_tokens
        chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
        self.llm = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=n_ctx,
                         n_gpu_layers=n_gpu_layers, logits_all=True, verbose=verbose)

    @staticmethod
    def _img_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            base_64_str = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base_64_str}"

    def infer(self, image_path: str, prompt: str, temperature: Optional[float] = None,
              max_tokens: Optional[int] = None) -> str:
        img_base64 = self._img_to_base64(image_path)

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_base64}},
                    ],
                }
            ],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

        return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    model_path = r"G:/Models/MM/ImgToText/NuMarkdown-8B-Thinking-Q4_K_M.gguf"
    mmproj_path = r"G:/Models/MM/ImgToText/mmproj-BF16.gguf"
    image_path = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
    image_path = "D:/Users/Administrator/Desktop/表情包/71FE2B9B7025D865169A3A38793591C9.jpg"

    model = ImageToTextLLM(
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_gpu_layers=-1,
    )

    prompt = "Please use the most precise English word to summarize the expression of this anime character."
    result = model.infer(image_path, prompt=prompt)
    print(result)  # The most precise English word to summarize the expression of this anime character is "confused".
    # 效果也一般
