"""
MetaMan - ComfyUI Custom Node Package
Universal metadata management for AI image generation platforms
"""

from .metaman import MetaManLoadImage, MetaManExtractComponents, MetaManEmbedAndSave

# Core three-node architecture
NODE_CLASS_MAPPINGS = {
    "MetaManLoadImage": MetaManLoadImage,
    "MetaManExtractComponents": MetaManExtractComponents,
    "MetaManEmbedAndSave": MetaManEmbedAndSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaManLoadImage": "MetaMan Load Image",
    "MetaManExtractComponents": "MetaMan Extract Components", 
    "MetaManEmbedAndSave": "MetaMan Embed & Save"
}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Package info
__version__ = "1.0.0"
__author__ = "MetaMan Development Team"
__description__ = "Universal metadata management for AI image generation platforms"
