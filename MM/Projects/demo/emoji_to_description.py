# import hashlib
# import os
# import re
# from pathlib import Path
# from typing import Union
#
# import unicodedata
#
# from MM.ImgToText.Caption.qwen3VL4B import QwenVLInferencer
#
# # 定义支持的图片后缀
# IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}  # todo: gif
#
#
# def batch_process_emotions(folder_path, model_path="Qwen/Qwen3-VL-4B-Thinking", max_new_tokens=1024):
#     """
#     遍历文件夹，识别图片中人物的表情。
#     """
#     # 1. 初始化推理类 (只加载一次模型)
#     inference_engine = QwenVLInferencer(model_path=model_path)
#     # 2. 检查文件夹路径是否存在
#     folder = Path(folder_path)
#     if not folder.exists() or not folder.is_dir():
#         print(f"错误：路径 '{folder_path}' 不是有效的文件夹。")
#         return
#     print(f"开始处理文件夹: {folder_path}\n" + "=" * 50)
#
#     # 3. 遍历文件夹
#     # 使用 sorted 保证处理顺序
#     for file_path in sorted(folder.iterdir()):
#         # 判断是否为文件且后缀在支持列表中
#         if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
#             img_path_str = str(file_path.absolute())
#
#             try:
#                 # 设定 prompt
#                 prompt = ("你是动漫人物表情识别助手。"
#                           "请观察图片，输出1~5个中文短词描述图中的动漫人物。"
#                           "先给出具体的情绪或神情词（必选，可以1~3个词语）；"
#                           "然后给出人物的动作描述（最好有，可以1~2个词语）；"
#                           "如果图片上有文字也可以通过OCR提取出来再补充到后面；"
#                           "如果还有其他特别的内容需要描述也可以补充到后面（可以0~2个词语或短句）。"
#                           "用英文逗号','分隔描述词。"
#                           "输出的内容严格无多余文字。")
#                 # "请观察图片，输出1~3个中文短词描述图中的动漫人物。"
#                 # "先给出具体的情绪或神情词（包括但不限于兴奋、委屈、傲娇、脸红、期待），"
#                 # "然后给出人物的动作描述（包括但不限于思考、盯、赌气、托腮）。"
#                 # "用英文逗号','分隔描述词。"
#                 # "输出的内容严格无多余文字。"
#                 # 4. 调用推理
#                 thinking, response = inference_engine.generate_output(img_path_str, prompt,
#                                                                       max_new_tokens=max_new_tokens)
#
#                 # 5. 按要求输出内容
#                 print(f"图片路径: {img_path_str}")
#                 print(f"思考内容: {thinking}")
#                 print(f"输出内容: {response}")
#                 print("-" * 30)
#
#             except Exception as e:
#                 print(f"处理图片 {file_path.name} 时发生错误: {e}")
#         else:
#             # 跳过非图片文件
#             continue
#
#     print("=" * 50 + "\n所有图片处理完毕。")
#
#
# # class ImageRenamer:
# #     def __init__(self, model_path: str = "Qwen/Qwen3-VL-4B-Thinking", device: str = "auto"):
# #         self.inferencer = QwenVLInferencer(model_path=model_path, device=device)
# #
# #     @staticmethod
# #     def _safe_filename(name: str, keep_whitespace: bool = False, max_len: int = 200) -> str:
# #         """
# #         将模型返回的描述转换为合法的文件名：
# #         - 去掉文件系统非法字符
# #         - 规范化 unicode
# #         - 截断过长名称
# #         """
# #         if not name:
# #             return ""
# #
# #         # 规范化 unicode
# #         name = unicodedata.normalize('NFKC', name)
# #
# #         # 删除控制字符
# #         name = "".join(ch for ch in name if ord(ch) >= 32)
# #
# #         # 替换常见非法字符
# #         invalid = r'<>:"/\\|?*\n\r\t'
# #         name = ''.join('_' if c in invalid else c for c in name)
# #
# #         # 选择是否保留空格
# #         if not keep_whitespace:
# #             name = re.sub(r'\s+', '_', name)
# #
# #         # 限制长度
# #         if len(name) > max_len:
# #             name = name[:max_len]
# #
# #         # 去掉首尾空格或下划线
# #         name = name.strip(' _-.')
# #
# #         return name or ""
# #
# #     @staticmethod
# #     def _file_hash(path: Union[str, Path], length=8) -> str:
# #         h = hashlib.sha1()
# #         with open(path, 'rb') as f:
# #             while True:
# #                 chunk = f.read(8192)
# #                 if not chunk:
# #                     break
# #                 h.update(chunk)
# #         return h.hexdigest()[:length]
# #
# #     def rename_folder(
# #             self,
# #             folder_path: Union[str, Path],
# #             prompt: Union[str, None] = None,
# #             batch_size: int = 4,
# #             dry_run: bool = False,
# #             overwrite: bool = False
# #     ):
# #         """
# #         将文件夹下的图片按模型返回的描述重命名。
# #         - prompt: 若为 None，则使用一个默认 prompt（可按需改）
# #         - batch_size: 每次送入模型的图片数量
# #         - dry_run: True 时仅打印计划改名，不执行实际改名
# #         - overwrite: True 时允许覆盖存在的同名文件（不推荐）
# #         """
# #         folder = Path(folder_path)
# #         if not folder.exists() or not folder.is_dir():
# #             raise ValueError(f"路径不是有效文件夹: {folder_path}")
# #
# #         if prompt is None:
# #             prompt = (
# #                 "你是动漫人物表情识别助手。"
# #                 "请观察图片，输出1~5个中文短词描述图中的动漫人物。"
# #                 "先给出具体的情绪或神情词（必选，可以1~3个词语）；"
# #                 "然后给出人物的动作描述（最好有，可以1~2个词语）；"
# #                 "如果还有其他特别的内容需要描述也可以补充到后面（可以0~2个词语）；"
# #                 "如果图片上有文字也可以通过OCR提取出来再补充到后面。"
# #                 "用英文逗号','分隔描述词。"
# #                 "输出的内容严格无多余文字。"
# #             )
# #
# #         # 收集图片
# #         files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
# #         if not files:
# #             print("未发现支持的图片文件。")
# #             return
# #
# #         total = len(files)
# #         print(f"发现 {total} 张图片，开始分批处理（batch_size={batch_size}）")
# #
# #         for i in range(0, total, batch_size):
# #             batch_files = files[i:i + batch_size]
# #             img_paths = [str(p) for p in batch_files]
# #
# #             try:
# #                 results = self.inferencer.generate_batch(img_paths, prompt)
# #             except RuntimeError as e:
# #                 # 常见情况：OOM 或者 GPU 错误；给出提示并尝试单张回退
# #                 print(f"批量推理时发生错误: {e}. 尝试逐张处理这一批以回避 OOM。")
# #                 results = []
# #                 for p in img_paths:
# #                     try:
# #                         res = self.inferencer.generate(p, prompt)
# #                         results.append(res)
# #                     except Exception as e2:
# #                         print(f"单张推理也失败: {p} -> {e2}")
# #                         results.append(("", ""))
# #
# #             # 对每个结果执行重命名
# #             for pf, (thinking, response) in zip(batch_files, results):
# #                 original_ext = pf.suffix.lower()
# #                 safe_name = self._safe_filename(response)
# #                 if not safe_name:
# #                     # 回退为基于文件 hash 的默认名
# #                     safe_name = f"img_{self._file_hash(pf)}"
# #
# #                 dest_name = f"{safe_name}{original_ext}"
# #                 dest_path = pf.with_name(dest_name)
# #
# #                 # 处理重名冲突
# #                 counter = 1
# #                 while dest_path.exists() and not overwrite:
# #                     # 如果目标文件就是当前文件，则跳过（同名同内容）
# #                     try:
# #                         if dest_path.samefile(pf):
# #                             # 完全相同文件，视为无需重命名
# #                             break
# #                     except Exception:
# #                         pass
# #
# #                     dest_path = pf.with_name(f"{safe_name}_{counter}{original_ext}")
# #                     counter += 1
# #
# #                 # 执行或打印
# #                 if dry_run:
# #                     print(f"[DRYRUN] {pf.name} -> {dest_path.name}")
# #                 else:
# #                     try:
# #                         if dest_path.exists() and overwrite:
# #                             os.replace(pf, dest_path)
# #                         else:
# #                             pf.rename(dest_path)
# #                         print(f"{pf.name} -> {dest_path.name}")
# #                     except Exception as e:
# #                         print(f"重命名失败: {pf.name} -> {dest_path.name} ; 错误: {e}")
# #
# #         print("处理完成。")
#
#
# # ---------------- 使用示例 ----------------
# if __name__ == "__main__":
#     emoji_folder = "../../../Datasets/Temp/Emoji"
#     batch_process_emotions(emoji_folder)
#
#     # # 示例：将 ./emojis 文件夹下的图片按模型描述重命名，batch_size=8，先 dry-run
#     # img_folder = "../../../Datasets/Temp/Emoji"
#     # renamer = ImageRenamer(model_path="Qwen/Qwen3-VL-4B-Thinking", device="auto")
#     # # 先做一次 dry run 查看结果
#     # renamer.rename_folder(img_folder, batch_size=8, dry_run=True)
