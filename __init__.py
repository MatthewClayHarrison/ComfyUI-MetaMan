"""
MetaMan - ComfyUI Custom Node Package
Universal metadata management for AI image generation platforms
"""

from .metaman import MetaManUniversalNode
from .specialized_nodes import MetaManWorkflowSaver, MetaManDependencyResolver

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    "MetaManUniversalNode": MetaManUniversalNode,
    "MetaManWorkflowSaver": MetaManWorkflowSaver,
    "MetaManDependencyResolver": MetaManDependencyResolver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaManUniversalNode": "MetaMan Universal",
    "MetaManWorkflowSaver": "MetaMan Workflow Saver", 
    "MetaManDependencyResolver": "MetaMan Dependency Resolver"
}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Package info
__version__ = "1.0.0"
__author__ = "MetaMan Development Team"
__description__ = "Universal metadata management for AI image generation platforms"
