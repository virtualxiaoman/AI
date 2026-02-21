import re
import os
import io
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List, Tuple, Union, Optional, Set


class QwenVLInferencer:
    IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}

    def __init__(self, model_path="Qwen/Qwen3-VL-4B-Thinking", device="auto", do_sample=False, temperature=0.7,
                 top_p=0.9, repetition_penalty=1.05, response_max_tokens=2048, max_retries=1):
        """
        初始化模型和处理器。
        """
        # print(f"正在加载模型: {model_path}")
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        # print("模型加载完成。")
        # 生成相关默认配置
        self.is_thinking_model = "Thinking" in model_path  # 判断模型是不是思考版本
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.response_max_tokens = response_max_tokens
        self.max_retries = max_retries

        self.found_closing = False
        self.retry_prompt = "请基于上面的思考直接给出最终答案。不要重复上面的思考内容，直接给出最终回复。"

    def _preprocess_image(self, img_input: Union[str, Image.Image], max_pixels: int = 1000000) -> Image.Image:
        """
        预处理图片：支持路径或PIL对象，若总像素超过阈值则按比例缩放。
        max_pixels: 默认 1,000,000 (约等于 1024x1024)
        """
        # 1. 加载图片
        if isinstance(img_input, str):
            if not any(img_input.lower().endswith(ext) for ext in self.IMAGE_EXTENSIONS):
                raise ValueError(f"不支持的文件格式: {os.path.splitext(img_input)[1]}")
            img = Image.open(img_input)
        else:
            img = img_input

        # 统一转为 RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 2. 基于像素总量进行动态缩放
        w, h = img.size
        current_pixels = w * h

        if current_pixels > max_pixels:
            # 计算缩放比例：我们要让 (w*s) * (h*s) = max_pixels
            # 所以 s^2 = max_pixels / (w*h) -> s = sqrt(max_pixels / current_pixels)
            scale = (max_pixels / current_pixels) ** 0.5

            new_w = int(w * scale)
            new_h = int(h * scale)

            # Qwen-VL 建议长宽最好是 28 的倍数以获得最佳对齐效果（可选优化）
            new_w = (new_w // 28) * 28
            new_h = (new_h // 28) * 28

            # 兜底：防止缩太小变成 0
            new_w = max(new_w, 28)
            new_h = max(new_h, 28)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"检测到高分辨率图片 ({w}x{h}), 已按比例缩放至 {new_w}x{new_h}")
        # else:
            # print(f"图片尺寸为 {w}x{h}, 无需缩放。")

        return img

    def _parse_output(self, raw_output):
        """
        默认是思考版本。
        根据用户观察到的格式解析：[Thinking内容]</think>[回复内容]
        """
        self.found_closing = "</think>" in raw_output
        # 使用 split 按照第一个 </think> 进行切割
        parts = raw_output.split("</think>", 1)

        if len(parts) == 2:
            # 成功分割：前半部分是思考，后半部分是回复
            thinking_content = parts[0].strip()
            response_content = parts[1].strip()
        else:
            thinking_content = raw_output.strip()
            response_content = ""

        # 移除可能存在的起始 <think> 标签（以防万一模型偶尔又带上了）
        thinking_content = thinking_content.replace("<think>", "").strip()

        # 清理常见的结束控制符
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|file_separator|>"]
        for token in stop_tokens:
            response_content = response_content.replace(token, "")

        return thinking_content, response_content.strip()

    def generate_output(self, img, prompt, max_new_tokens=None):
        """
        执行推理。
        可以改为：
        thinking, response = self.generate_batch([img_path], prompt, max_new_tokens=max_new_tokens)[0]
        return thinking, response

        Args:
            img (str): 图片的本地路径。
            prompt (str): 提示词。
            max_new_tokens (int): 最大生成长度。

        Returns:
            tuple: (thinking_content, response_content)
        """
        if max_new_tokens is not None and isinstance(max_new_tokens, int):
            self.response_max_tokens = max_new_tokens

        img = self._preprocess_image(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img},
                    {
                        "type": "text",
                        "text": prompt},
                ],
            }
        ]

        output_text = self._generate_output(messages)

        # 如果是非思考版本，直接返回整个文本作为回复，thinking 为空
        if not self.is_thinking_model:
            return "", output_text.strip()
        else:
            thinking, response = self._parse_output(output_text)

            # 常规成功路径：已经包含闭合标签且有回复
            if self.found_closing and response:
                return thinking, response
            # 如果未闭合但模型已经直接给出回复（理论上不应该出现）
            if (not self.found_closing) and response:
                return thinking, response

            # 剩下的情况——回退：当没有闭合标签且有 thinking 时，使用 thinking 作为 assistant 上下文再请求直接输出答案
            retries = 0
            while retries < self.max_retries and thinking:
                print(f"回退尝试 {retries + 1}/{self.max_retries}。")
                retries += 1
                retry_messages = [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": thinking}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": self.retry_prompt}]}
                ]
                output_text_2 = self._generate_output(retry_messages)
                thinking2, response_2 = self._parse_output(output_text_2)
                thinking = thinking + "\n" + thinking2 if thinking2 else thinking  # 累积thinking的内容
                if response_2:
                    return thinking, response_2
                else:
                    thinking, response = self._parse_output(output_text_2)

            # 所有回退失败：退而将最初解码的内容返回（确保函数总有返回）。优先返回 response，否则返回原始 output_text
            return thinking, (response or output_text).strip()

    def _generate_output(self, msg):
        inputs = self.processor.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}  # 确保在模型设备上

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.response_max_tokens,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]  # 裁剪掉输入长度部分
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text
    # def generate_batch(
    #         self,
    #         img_paths: List[str],
    #         prompt: Union[str, List[str]],
    #         max_new_tokens: int = 2048,
    #         repetition_penalty: float = 1.05,
    #         do_sample: bool = True,
    #         temperature: float = 0.7,
    #         top_p: float = 0.9
    # ) -> List[Tuple[str, str]]:
    #     """
    #     批量推理：一次处理多张图片并返回对应 (thinking, response) 列表。
    #     - img_paths: 本地图片路径列表
    #     - prompt: 如果是字符串，则对每张图片使用相同 prompt；如果是列表，则长度需等于 img_paths
    #     """
    #
    #     if isinstance(prompt, str):
    #         prompts = [prompt] * len(img_paths)
    #     else:
    #         prompts = list(prompt)
    #         if len(prompts) != len(img_paths):
    #             raise ValueError("当 prompt 为列表时，长度必须与 img_paths 相同。")
    #
    #     # 构造 messages（batch）
    #     messages = []
    #     for img_path, pr in zip(img_paths, prompts):
    #         messages.append({
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": img_path},
    #                 {"type": "text", "text": pr}
    #             ]
    #         })
    #
    #     # 将 batch 转为模型输入张量
    #     inputs = self.processor.apply_chat_template(
    #         messages,
    #         tokenize=True,
    #         add_generation_prompt=True,
    #         return_dict=True,
    #         return_tensors="pt"
    #     )
    #
    #     # 移动到模型设备
    #     inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    #
    #     # 记录每个样本的 input length（按 attention_mask 计）
    #     if "attention_mask" in inputs:
    #         input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    #     else:
    #         # 退化情况：使用 input_ids 的长度
    #         input_lens = [inputs["input_ids"].shape[1]] * len(img_paths)
    #
    #     # 推理
    #     with torch.no_grad():
    #         generated_ids = self.model.generate(
    #             **inputs,
    #             max_new_tokens=max_new_tokens,
    #             repetition_penalty=repetition_penalty,
    #             do_sample=do_sample,
    #             temperature=temperature,
    #             top_p=top_p,
    #         )
    #
    #     # generated_ids: tensor (batch, seq_len_out)
    #     # 将每个样本裁剪掉输入部分：用 input_lens
    #     results = []
    #     for i in range(generated_ids.shape[0]):
    #         out_ids = generated_ids[i].tolist()
    #         in_len = int(input_lens[i])
    #         # 保护性判断
    #         if len(out_ids) <= in_len:
    #             trimmed = out_ids[:]
    #         else:
    #             trimmed = out_ids[in_len:]
    #
    #         # 解码：注意保留特殊 token 以便解析 <think> 标签
    #         # 这里使用 processor.batch_decode 接口按单样本解码
    #         text = self.processor.batch_decode([torch.tensor(trimmed)], skip_special_tokens=False,
    #                                            clean_up_tokenization_spaces=False)[0]
    #         thinking, response = self._parse_output(text)
    #         results.append((thinking, response))
    #
    #     return results


# --- 使用示例 ---

if __name__ == "__main__":
    # 1. 初始化类 (建议放在全局或初始化位置，避免重复加载模型)
    inference_engine = QwenVLInferencer()

    # 2. 设置参数
    image_file = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
    image_file = "D:/Users/Administrator/Desktop/表情包/0B15F0FCBA713F33719AF4AAE371253F.gif"
    # image_file = "D:/Users/Administrator/Desktop/表情包/0eeb09bff8ab32e78c2a0abe90d17d70.mp4"
    # image_file = "D:/Users/Administrator/Desktop/表情包/LuoTianYi/下次一定.jpg"
    user_prompt = "请用一个词描述图中动漫人物的表情。"

    try:
        # 3. 调用生成方法
        thinking, response = inference_engine.generate_output(image_file, user_prompt)

        # 4. 打印结果
        print("-" * 30)
        print("【Thinking 过程】:")
        print(thinking)
        print("-" * 30)
        print("【最终回复】:")
        print(response)
        print("-" * 30)

    except Exception as e:
        print(f"发生错误: {e}")
