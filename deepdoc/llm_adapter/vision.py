"""
Vision LLM utilities for DeepDoc

Enhanced vision_llm_chunk with better error handling and format support.
"""

import io
import logging
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

# Try to import markdown cleaning function
try:
    from .utils import clean_markdown_block
except ImportError:
    def clean_markdown_block(text: str) -> str:
        """Fallback markdown cleaning function"""
        import re
        text = re.sub(r'^\s*```markdown\s*\n?', '', text)
        text = re.sub(r'\n?\s*```\s*$', '', text)
        return text.strip()


def vision_llm_chunk(
    binary: Any,
    vision_model: Any,
    prompt: Optional[str] = None,
    callback: Optional[Callable] = None
) -> str:
    """
    Enhanced vision LLM chunk processing with better error handling and format support.

    This is an improved version that supports multiple image formats and provides
    better error handling compared to the basic depend/vision_llm_chunk.py.

    Args:
        binary: Image binary data (PIL Image, bytes, or BytesIO)
        vision_model: Vision model instance with describe_with_prompt method
        prompt: Optional prompt for image description
        callback: Optional callback function for progress reporting

    Returns:
        str: Processed markdown text from vision model
    """
    callback = callback or (lambda prog, msg: None)

    img = binary
    txt = ""

    try:
        # Convert image to bytes with format fallback
        img_binary = io.BytesIO()

        # Try different formats in order of preference
        formats_to_try = ['JPEG', 'PNG', 'WEBP', 'BMP']

        saved_successfully = False
        for fmt in formats_to_try:
            try:
                img.save(img_binary, format=fmt)
                saved_successfully = True
                break
            except Exception:
                img_binary.seek(0)
                img_binary.truncate()  # Clear buffer for next attempt
                continue

        if not saved_successfully:
            raise ValueError("Unable to save image in any supported format")

        img_binary.seek(0)

        # Call vision model
        ans = clean_markdown_block(
            vision_model.describe_with_prompt(img_binary.read(), prompt)
        )

        txt += "\n" + ans

        return txt

    except Exception as e:
        error_msg = f"Vision model processing failed: {str(e)}"
        logger.error(error_msg)
        callback(-1, error_msg)
        return ""


def vision_llm_chunk_with_fallback(
    binary: Any,
    vision_model: Any,
    prompt: Optional[str] = None,
    callback: Optional[Callable] = None,
    fallback_text: str = "Image processing failed"
) -> str:
    """
    Vision LLM chunk processing with fallback text on failure.

    Args:
        binary: Image binary data
        vision_model: Vision model instance
        prompt: Optional prompt
        callback: Optional callback
        fallback_text: Text to return on failure

    Returns:
        str: Processed text or fallback text
    """
    result = vision_llm_chunk(binary, vision_model, prompt, callback)
    return result if result.strip() else fallback_text


def batch_vision_llm_chunk(
    images: list,
    vision_model: Any,
    prompts: Optional[list] = None,
    callback: Optional[Callable] = None,
    max_workers: int = 3
) -> list:
    """
    Process multiple images in parallel.

    Args:
        images: List of image binary data
        vision_model: Vision model instance
        prompts: Optional list of prompts (same length as images)
        callback: Optional callback function
        max_workers: Maximum number of parallel workers

    Returns:
        list: List of processed text results
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not images:
        return []

    if prompts and len(prompts) != len(images):
        raise ValueError("prompts list must have same length as images list")

    results = [None] * len(images)
    prompts = prompts or [None] * len(images)

    def process_single(idx: int, img: Any, prompt: Optional[str]) -> tuple:
        try:
            result = vision_llm_chunk(img, vision_model, prompt, callback)
            return idx, result
        except Exception as e:
            logger.error(f"Failed to process image {idx}: {e}")
            return idx, ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single, i, img, prompt)
            for i, (img, prompt) in enumerate(zip(images, prompts))
        ]

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results
