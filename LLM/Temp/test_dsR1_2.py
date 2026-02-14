import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


class DeepSeekStreamInferencer:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        ).to(self.device if self.device != "cuda" else None)

    def chat(self, prompt, stream=True, max_new_tokens=2048):
        """
        对话主接口
        :param stream: 是否开启流式输出
        """
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # 基础生成参数
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if stream:
            return self._stream_generate(generation_kwargs)
        else:
            return self._normal_generate(generation_kwargs)

    def _normal_generate(self, kwargs):
        """非流式生成"""
        with torch.no_grad():
            output_ids = self.model.generate(**kwargs)
            # 过滤掉 input 部分
            new_tokens = output_ids[0][len(kwargs['input_ids'][0]):]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _stream_generate(self, kwargs):
        """流式生成器"""
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs['streamer'] = streamer

        # 开启新线程进行推理，否则会阻塞主线程导致无法读取 streamer
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        # 逐个产出文本
        for new_text in streamer:
            yield new_text


# --- 使用示例 ---
if __name__ == "__main__":
    bot = DeepSeekStreamInferencer()
    user_p = "介绍一下小满。"

    # --- 场景 1：使用流式输出 ---
    print("--- [流式模式] ---")
    print("思考中...\n", end="", flush=True)
    for chunk in bot.chat(user_p, stream=True):
        print(chunk, end="", flush=True)  # 实时打印每一个词
    print("\n\n--- [生成结束] ---\n")

    # --- 场景 2：使用普通输出 ---
    # print("--- [普通模式] ---")
    # full_text = bot.chat(user_p, stream=False)
    # print(full_text)
