"""
Utility functions for LLM adapter
"""

import re


def clean_markdown_block(text: str) -> str:
    """
    Clean markdown block formatting from text.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        str: Cleaned text without markdown formatting
    """
    if not text:
        return ""

    # Remove markdown code block markers
    text = re.sub(r'^\s*```markdown\s*\n?', '', text)
    text = re.sub(r'\n?\s*```\s*$', '', text)

    return text.strip()


def extract_image_description(text: str) -> str:
    """
    Extract the main description from vision model output.

    Args:
        text: Raw output from vision model

    Returns:
        str: Cleaned description
    """
    # Clean markdown formatting
    text = clean_markdown_block(text)

    # Remove common prefixes that vision models might add
    prefixes_to_remove = [
        "The image shows",
        "This image depicts",
        "The picture shows",
        "This is an image of",
        "The photo shows",
    ]

    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            # Only remove if it's followed by actual content
            remaining = text[len(prefix):].strip()
            if remaining and not remaining.startswith(("a", "an", "the")):
                continue
            text = remaining
            break

    return text.strip()


def validate_image_data(image_data: bytes, max_size: int = 10 * 1024 * 1024) -> bool:
    """
    Validate image data.

    Args:
        image_data: Raw image bytes
        max_size: Maximum allowed size in bytes

    Returns:
        bool: True if valid
    """
    if not image_data:
        return False

    if len(image_data) > max_size:
        return False

    # Check for common image signatures
    if len(image_data) < 8:
        return False

    # JPEG signature
    if image_data.startswith(b'\xff\xd8\xff'):
        return True

    # PNG signature
    if image_data.startswith(b'\x89PNG\r\n\x1a\n'):
        return True

    # GIF signature
    if image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
        return True

    # BMP signature
    if image_data.startswith(b'BM'):
        return True

    # WebP signature
    if image_data.startswith(b'RIFF') and len(image_data) >= 12:
        if image_data[8:12] == b'WEBP':
            return True

    return False
