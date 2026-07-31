"""
OCR 引擎模块
使用 PaddleOCR VL 识别扫描件 PDF 中的文字
懒加载：仅在首次调用时初始化模型
"""

import gc
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from config import settings, BASE_DIR

# 模块级单例（懒加载）
_model = None
_processor = None
_device: Optional[str] = None


def _resolve_device(requested: str) -> str:
    """解析推理设备"""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _load_model():
    """加载 PaddleOCR VL 模型和处理器（仅执行一次）"""
    global _model, _processor, _device

    ocr_cfg = settings.ocr
    model_path = BASE_DIR / ocr_cfg.model_path

    if not model_path.exists():
        raise FileNotFoundError(f"OCR 模型目录不存在: {model_path}")

    _device = _resolve_device(ocr_cfg.device)
    logger.info(f"加载 PaddleOCR VL 模型: {model_path} → {_device}")

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        _processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=_device if _device == "cuda" else None,
        )

        if _device == "cpu":
            _model = _model.float()

        _model.eval()
        logger.info("PaddleOCR VL 模型加载完成")

    except Exception as e:
        logger.error(f"PaddleOCR VL 模型加载失败: {e}")
        _model = None
        _processor = None
        raise


def ocr_page(image, prompt: Optional[str] = None) -> str:
    """对单张图片执行 OCR 识别

    Args:
        image: PIL.Image.Image 或图片路径
        prompt: OCR 提示词（默认使用配置中的 prompt）

    Returns:
        识别出的文字内容
    """
    global _model, _processor

    # 懒加载
    if _model is None or _processor is None:
        _load_model()

    if prompt is None:
        prompt = settings.ocr.prompt

    # 支持传入路径字符串
    if isinstance(image, (str, Path)):
        from PIL import Image
        image = Image.open(str(image)).convert("RGB")

    try:
        # 构造输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # 移到设备
        if _device != "cpu":
            inputs = {k: v.to(_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        # 截取生成部分（去掉输入 token）
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        result = _processor.decode(generated_ids, skip_special_tokens=True).strip()

        logger.debug(f"OCR 识别完成，文本长度: {len(result)}")
        return result

    except Exception as e:
        logger.error(f"OCR 推理失败: {e}")
        return ""


def release():
    """释放模型显存（用于显式清理）"""
    global _model, _processor
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("OCR 模型已释放")
