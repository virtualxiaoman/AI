import torch
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List, Tuple, Union, Optional


class QwenVLInferencer:
    def __init__(self, model_path="Qwen/Qwen3-VL-4B-Thinking", device="auto"):
        """
        初始化模型和处理器。
        """
        print(f"正在加载模型: {model_path}")
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("模型加载完成。")

    def _parse_output(self, raw_output):
        """
        根据用户观察到的格式解析：[Thinking内容]</think>[回复内容]
        """
        # 使用 split 按照第一个 </think> 进行切割
        parts = raw_output.split("</think>", 1)

        if len(parts) == 2:
            # 成功分割：前半部分是思考，后半部分是回复
            thinking_content = parts[0].strip()
            response_content = parts[1].strip()
        else:
            # 如果没找到 </think>，则认为全部是回复（或者模型跳过了思考过程）
            thinking_content = ""
            response_content = raw_output.strip()

        # 移除可能存在的起始 <think> 标签（以防万一模型偶尔又带上了）
        thinking_content = thinking_content.replace("<think>", "").strip()

        # 清理常见的结束控制符
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|file_separator|>"]
        for token in stop_tokens:
            response_content = response_content.replace(token, "")

        return thinking_content, response_content.strip()

    def generate(self, img_path, prompt, max_new_tokens=2048):
        """
        执行推理。
        可以改为：
        thinking, response = self.generate_batch([img_path], prompt, max_new_tokens=max_new_tokens)[0]
        return thinking, response

        Args:
            img_path (str): 图片的本地路径。
            prompt (str): 提示词。
            max_new_tokens (int): 最大生成长度。

        Returns:
            tuple: (thinking_content, response_content)
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path},
                    {
                        "type": "text",
                        "text": prompt},
                ],
            }
        ]

        # 准备输入
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # 推理 (使用 no_grad 节省显存)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
            )
            # 之前的设置：
            #                 do_sample=True,
            #                 temperature=0.7,
            #                 top_p=0.9,

        # 裁剪掉输入部分的 tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 解码，保留特殊 token 以便解析 <think> 标签
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]  # 取 batch 中的第一个结果

        # 解析并返回元组
        return self._parse_output(output_text)

    def generate_batch(
            self,
            img_paths: List[str],
            prompt: Union[str, List[str]],
            max_new_tokens: int = 2048,
            repetition_penalty: float = 1.05,
            do_sample: bool = True,
            temperature: float = 0.7,
            top_p: float = 0.9
    ) -> List[Tuple[str, str]]:
        """
        批量推理：一次处理多张图片并返回对应 (thinking, response) 列表。
        - img_paths: 本地图片路径列表
        - prompt: 如果是字符串，则对每张图片使用相同 prompt；如果是列表，则长度需等于 img_paths
        """

        if isinstance(prompt, str):
            prompts = [prompt] * len(img_paths)
        else:
            prompts = list(prompt)
            if len(prompts) != len(img_paths):
                raise ValueError("当 prompt 为列表时，长度必须与 img_paths 相同。")

        # 构造 messages（batch）
        messages = []
        for img_path, pr in zip(img_paths, prompts):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": pr}
                ]
            })

        # 将 batch 转为模型输入张量
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 移动到模型设备
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 记录每个样本的 input length（按 attention_mask 计）
        if "attention_mask" in inputs:
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        else:
            # 退化情况：使用 input_ids 的长度
            input_lens = [inputs["input_ids"].shape[1]] * len(img_paths)

        # 推理
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        # generated_ids: tensor (batch, seq_len_out)
        # 将每个样本裁剪掉输入部分：用 input_lens
        results = []
        for i in range(generated_ids.shape[0]):
            out_ids = generated_ids[i].tolist()
            in_len = int(input_lens[i])
            # 保护性判断
            if len(out_ids) <= in_len:
                trimmed = out_ids[:]
            else:
                trimmed = out_ids[in_len:]

            # 解码：注意保留特殊 token 以便解析 <think> 标签
            # 这里使用 processor.batch_decode 接口按单样本解码
            text = self.processor.batch_decode([torch.tensor(trimmed)], skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False)[0]
            thinking, response = self._parse_output(text)
            results.append((thinking, response))

        return results


# --- 使用示例 ---

if __name__ == "__main__":
    # 1. 初始化类 (建议放在全局或初始化位置，避免重复加载模型)
    inference_engine = QwenVLInferencer()

    # 2. 设置参数
    image_file = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
    user_prompt = "请用一个词描述图中动漫人物的表情。"

    try:
        # 3. 调用生成方法
        thinking, response = inference_engine.generate(image_file, user_prompt)

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
