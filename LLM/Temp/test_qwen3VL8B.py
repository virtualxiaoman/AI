import ollama
import os


class OllamaSession:
    def __init__(self, model='qwen3-vl:4b', persona_file=None, options=None):
        """
        初始化对话 Session
        :param model: 模型名称
        :param persona_file: 系统提示词，用于设定模型的人设或规则
        :param options: Ollama 运行参数 (temperature, top_p 等)
        """
        self.model = model
        self.options = options or {}

        # 初始化消息列表
        self.messages = []

        # 如果有系统提示词，首先加入
        if persona_file and os.path.exists(persona_file):
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona_content = f.read()
            # 将人设作为 system 消息，奠定整场对话的基调
            self.messages.append({'role': 'system', 'content': persona_content})
        else:
            print("警告：未找到人设文件，将以默认模式运行。")

    def chat(self, user_input, stream=False):
        """
        发送消息并获取回复
        :param user_input: 用户的对话内容
        :param stream: 是否开启流式输出
        """
        # 1. 将用户输入加入历史记录
        self.messages.append({'role': 'user', 'content': user_input})

        try:
            # 2. 调用 Ollama API
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                options=self.options,
                stream=stream
            )

            if stream:
                return self._handle_stream(response)
            else:
                # 3. 将模型回复加入历史记录，保持上下文同步
                assistant_content = response['message']['content']
                self.messages.append({'role': 'assistant', 'content': assistant_content})
                return assistant_content

        except Exception as e:
            return f"发生错误: {str(e)}"

    def _handle_stream(self, response_gen):
        """处理流式响应的内部私有方法"""
        full_response = ""
        for chunk in response_gen:
            content = chunk['message']['content']
            print(content, end='', flush=True)  # 实时打印到控制台
            full_response += content

        print()  # 换行
        # 流式结束后，也要存入历史记录
        self.messages.append({'role': 'assistant', 'content': full_response})
        return full_response

    def clear_history(self):
        """清空对话历史"""
        # 保留第一个 system prompt (如果有的话)
        if self.messages and self.messages[0]['role'] == 'system':
            self.messages = [self.messages[0]]
        else:
            self.messages = []


# --- 使用示例 ---
if __name__ == "__main__":
    # 定义你之前的那些配置
    my_options = options = {
        # 常用采样控制
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        # 生成长度 / 上下文
        "num_predict": 2048,  # 要求返回的最大 token 数（名称/行为视版本）
        "num_ctx": 65536,  # 如果模型支持，扩展上下文窗口
        # 减少重复
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
    }

    # 初始化
    session = OllamaSession(
        model='qwen3-vl:8b',
        persona_file='E:/Py-Project/MMT/resources/prompt/Shiroko.txt',
        options=my_options
    )

    print("--- 已进入对话模式（输入 'quit' 退出） ---")
    while True:
        user_msg = input("老师: ")
        if user_msg.lower() in ['quit', 'exit']:
            break

        print("白子: ", end="")
        session.chat(user_msg, stream=True)
