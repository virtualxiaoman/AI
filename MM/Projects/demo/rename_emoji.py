import re
import sys
from pathlib import Path
from typing import Union, Set
import unicodedata

from MM.ImgToText.Caption.qwen3VL4B import QwenVLInferencer


class ImageSemanticRenamer:
    IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}  # 定义支持的图片后缀

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-4B-Thinking"):
        """
        初始化重命名器，加载模型。
        """
        print(f"正在加载模型: {model_path}")
        self.inference_engine = QwenVLInferencer(model_path=model_path)
        print("模型加载完成。")

    @staticmethod
    def _sanitize_filename(text: str, max_length: int = 100) -> str:
        """
        清洗并格式化文件名
        """
        # 1. Unicode 归一化
        text = unicodedata.normalize('NFKC', text)

        # 2. 核心修改：统一分隔符
        # 将中文逗号 '，' 替换为英文逗号 ','
        text = text.replace('，', ',')

        # 3. 规范化空格：先按逗号切分，去除每个词前后的空格，再用 ', ' 重新连接
        parts = [p.strip() for p in text.split(',') if p.strip()]
        text = ', '.join(parts)

        # 4. 移除系统非法字符 (注意：这里不要移除逗号和空格，因为我们要保留它们)
        # Windows 非法字符: < > : " / \ | ? *
        text = re.sub(r'[<>:"/\\|?*]', '', text)

        # 5. 移除换行/制表符并二次修剪
        text = text.replace('\n', '').replace('\r', '').replace('\t', '').strip()

        # 6. 截断长度
        if len(text) > max_length:
            text = text[:max_length]

        return text

    def _get_unique_path(self, file_path: Path) -> Path:
        """
        如果目标路径已存在，则在文件名后添加序号 1, 2 等，直到找到唯一路径。
        """
        if not file_path.exists():
            return file_path

        stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent
        counter = 1

        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    def rename_folder(self, folder_path: Union[str, Path], max_new_tokens: int = 2048):
        """
        遍历文件夹，识别图片内容并重命名。
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            print(f"错误：路径 '{folder_path}' 不是有效的文件夹。")
            return

        print(f"开始处理文件夹: {folder_path}\n" + "=" * 50)

        # 设定 Prompt (保持你原有的设定)
        prompt = ("你是动漫人物表情识别助手。"
                  "请观察图片，输出1~5个中文短词描述图中的动漫人物。"
                  "先给出具体的情绪或神情词（必选，可以1~3个词语）；"
                  "然后给出人物的动作描述（最好有，可以1~2个词语）；"
                  "如果还有其他特别的内容需要描述也可以补充到后面（可以0~2个词语或短句）；"
                  "如果图片上有文字也可以通过OCR提取出来再补充到后面。"
                  "输出的几个描述词要尽量有区分度，不要使用过于相似的词语。"
                  "必须用英文逗号加一个空格 ', ' 分隔描述词，不要使用中文逗号。"
                  "输出的内容严格无多余文字。")

        # 获取所有待处理文件列表
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS]
        total_files = len(files)

        print(f"共 {total_files} 张图片。")

        for index, file_path in enumerate(files, 1):
            img_path_str = str(file_path.absolute())
            original_name = file_path.name

            print(f"[{index}/{total_files}] {original_name} ", end="")

            try:
                # 1. 调用推理
                thinking, response = self.inference_engine.generate_output(
                    img_path_str, prompt, max_new_tokens=max_new_tokens
                )

                # 2. 清洗输出内容作为新文件名
                clean_name = self._sanitize_filename(response)

                # 如果模型输出为空，则跳过重命名
                if not clean_name:
                    print("  警告：模型输出为空，跳过重命名。")
                    continue

                # 3. 构建新路径并处理重名
                new_filename = clean_name + file_path.suffix
                target_path = folder / new_filename

                # 如果新名字和旧名字完全一样，跳过
                if target_path == file_path:
                    print("  新文件名与原名一致，无需修改。")
                    continue

                # 获取唯一路径（处理重名）
                final_path = self._get_unique_path(target_path)

                # 4. 执行重命名
                file_path.rename(final_path)

                # print(f"  识别结果: {response}")
                print(f"  重命名为: {final_path.name}")

            except Exception as e:
                print(f"  处理图片 {original_name} 时发生错误: {e}")

            print("-" * 30)

        print("=" * 50 + "\n所有图片处理完毕。")


# 使用示例
if __name__ == "__main__":
    renamer = ImageSemanticRenamer(model_path="Qwen/Qwen3-VL-4B-Thinking")
    # target_folder = "../../../Datasets/Temp/Emoji"
    target_folder = "D:/Users/Administrator/Desktop/表情包/未分类/2"
    renamer.rename_folder(target_folder)
