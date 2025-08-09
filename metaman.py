"""
MetaMan Node for ComfyUI
Universal metadata management for AI image generation platforms
Supports A1111, ComfyUI, Civitai, Tensor.ai, Forge, and other services
"""

import torch
import json
import yaml
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import piexif
import folder_paths
import hashlib
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class MetaManUniversalNode:
    """
    Universal metadata management node for ComfyUI
    Handles metadata conversion between all major AI image generation services
    """
    
    # Supported services
    SUPPORTED_SERVICES = [
        "automatic1111", "comfyui", "civitai", "forge", 
        "tensor.ai", "seaart.ai", "leonardo.ai", "midjourney", "generic"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": ([
                    "extract_universal", "convert_to_service", "save_workflow", 
                    "generate_dependencies", "export_metadata"
                ], {"default": "extract_universal"}),
                "target_service": (cls.SUPPORTED_SERVICES, {"default": "automatic1111"}),
            },
            "optional": {
                "output_format": (["png_chunk", "json_file", "txt_file", "embed_in_image"], {"default": "png_chunk"}),
                "template_override": ("STRING", {"default": ""}),
                "include_workflow": ("BOOLEAN", {"default": True}),
                "include_dependencies": ("BOOLEAN", {"default": True}),
                "dependency_sources": (["civitai", "huggingface", "all"], {"default": "all"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "universal_metadata", "service_metadata", "dependencies")
    FUNCTION = "process_universal_metadata"
    CATEGORY = "MetaMan"
    DESCRIPTION = "Universal metadata management across all AI image generation platforms"

    def __init__(self):
        """Initialize MetaMan with universal schema and service templates"""
        self.schema_path = os.path.join(os.path.dirname(__file__), "templates", "universal_schema.yaml")
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates", "services")
        self.universal_schema = self._load_universal_schema()
        self.service_templates = self._load_service_templates()
        
    def _load_universal_schema(self) -> Dict:
        """Load the universal metadata schema"""
        try:
            if os.path.exists(self.schema_path):
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                return self._create_default_schema()
        except Exception as e:
            print(f"MetaMan: Error loading schema: {e}")
            return self._create_default_schema()
    
    def _create_default_schema(self) -> Dict:
        """Create default universal schema"""
        return {
            "schema_version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "fields": {
                # Core generation parameters
                "prompt": {
                    "type": "string",
                    "description": "Positive prompt text",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai", "leonardo.ai"],
                    "required": True
                },
                "negative_prompt": {
                    "type": "string", 
                    "description": "Negative prompt text",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai"],
                    "required": False
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of inference steps",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai", "leonardo.ai"],
                    "range": [1, 150]
                },
                "cfg_scale": {
                    "type": "float",
                    "description": "Classifier-free guidance scale", 
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai"],
                    "range": [1.0, 30.0]
                },
                "sampler": {
                    "type": "string",
                    "description": "Sampling method",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai"],
                    "mappings": {
                        "automatic1111": {
                            "Euler": "Euler",
                            "Euler a": "Euler a", 
                            "DPM++ 2M Karras": "DPM++ 2M Karras",
                            "DPM++ SDE Karras": "DPM++ SDE Karras"
                        },
                        "comfyui": {
                            "Euler": "euler",
                            "Euler a": "euler_ancestral",
                            "DPM++ 2M Karras": "dpmpp_2m",
                            "DPM++ SDE Karras": "dpmpp_sde"
                        }
                    }
                },
                "scheduler": {
                    "type": "string",
                    "description": "Noise scheduler",
                    "supported_by": ["comfyui", "forge", "tensor.ai"],
                    "mappings": {
                        "comfyui": ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],
                        "forge": ["normal", "karras", "exponential"]
                    }
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai", "leonardo.ai"]
                },
                "width": {
                    "type": "integer", 
                    "description": "Image width",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai", "leonardo.ai"]
                },
                "height": {
                    "type": "integer",
                    "description": "Image height", 
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai", "leonardo.ai"]
                },
                
                # Model information
                "model_name": {
                    "type": "string",
                    "description": "Base model name",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai", "leonardo.ai"]
                },
                "model_hash": {
                    "type": "string",
                    "description": "Model file hash (SHA256 preferred)",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge"]
                },
                "model_download_url": {
                    "type": "string",
                    "description": "URL to download the model",
                    "supported_by": ["civitai", "huggingface"]
                },
                
                # LoRA information
                "loras": {
                    "type": "array",
                    "description": "LoRA models used",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai"],
                    "item_schema": {
                        "name": "string",
                        "weight": "float", 
                        "hash": "string",
                        "download_url": "string"
                    }
                },
                
                # Embeddings/Textual Inversions
                "embeddings": {
                    "type": "array", 
                    "description": "Textual inversions/embeddings used",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge"],
                    "item_schema": {
                        "name": "string",
                        "hash": "string", 
                        "download_url": "string"
                    }
                },
                
                # Advanced parameters
                "clip_skip": {
                    "type": "integer",
                    "description": "CLIP layers to skip",
                    "supported_by": ["automatic1111", "forge", "civitai"]
                },
                "eta": {
                    "type": "float",
                    "description": "Eta parameter for DDIM",
                    "supported_by": ["automatic1111", "forge"]
                },
                "denoising_strength": {
                    "type": "float", 
                    "description": "Denoising strength for img2img",
                    "supported_by": ["automatic1111", "comfyui", "forge", "tensor.ai"]
                },
                
                # Service-specific fields
                "comfyui_workflow": {
                    "type": "object",
                    "description": "Complete ComfyUI workflow JSON",
                    "supported_by": ["comfyui"]
                },
                "tensor_ai_style": {
                    "type": "string",
                    "description": "Tensor.AI style preset",
                    "supported_by": ["tensor.ai"]
                },
                "leonardo_preset": {
                    "type": "string", 
                    "description": "Leonardo.AI model preset",
                    "supported_by": ["leonardo.ai"]
                },
                "midjourney_version": {
                    "type": "string",
                    "description": "Midjourney model version",
                    "supported_by": ["midjourney"]
                },
                
                # Metadata
                "creation_time": {
                    "type": "string",
                    "description": "Image creation timestamp",
                    "supported_by": ["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "seaart.ai", "leonardo.ai"]
                },
                "source_service": {
                    "type": "string",
                    "description": "Original service that created the image", 
                    "supported_by": ["metaman"]
                },
                "metaman_version": {
                    "type": "string",
                    "description": "MetaMan version used",
                    "supported_by": ["metaman"]
                }
            }
        }

    def _load_service_templates(self) -> Dict:
        """Load service-specific output templates"""
        templates = {}
        if os.path.exists(self.templates_dir):
            for service_file in os.listdir(self.templates_dir):
                if service_file.endswith('.yaml'):
                    service_name = service_file.replace('.yaml', '')
                    try:
                        with open(os.path.join(self.templates_dir, service_file), 'r', encoding='utf-8') as f:
                            templates[service_name] = yaml.safe_load(f)
                    except Exception as e:
                        print(f"MetaMan: Error loading template for {service_name}: {e}")
        
        return templates

    def process_universal_metadata(self, image, operation, target_service, 
                                 output_format="png_chunk", template_override="", 
                                 include_workflow=True, include_dependencies=True,
                                 dependency_sources="all"):
        """
        Main processing function for universal metadata operations
        """
        try:
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                img_tensor = image[0]
                img_array = (img_tensor.cpu().numpy() * 255).astype('uint8')
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = image
            
            # Process based on operation
            if operation == "extract_universal":
                return self._extract_to_universal(pil_image, target_service, include_workflow, include_dependencies)
            elif operation == "convert_to_service":
                return self._convert_to_service(pil_image, target_service, output_format, include_workflow)
            elif operation == "save_workflow":
                return self._save_workflow(pil_image, output_format)
            elif operation == "generate_dependencies": 
                return self._generate_dependencies(pil_image, dependency_sources)
            elif operation == "export_metadata":
                return self._export_metadata(pil_image, target_service, output_format)
            else:
                return (image, "Unknown operation", "", "")
                
        except Exception as e:
            error_msg = f"MetaMan Error: {str(e)}"
            return (image, error_msg, "", "")
    
    def _extract_to_universal(self, image, target_service, include_workflow, include_dependencies):
        """Extract metadata from any source and convert to universal format"""
        # Detect source service and extract metadata
        source_metadata = self._extract_source_metadata(image)
        source_service = self._detect_source_service(source_metadata)
        
        # Convert to universal format
        universal_metadata = self._convert_to_universal(source_metadata, source_service)
        
        # Add workflow if requested and available
        if include_workflow:
            workflow = self._extract_workflow(image, source_service)
            if workflow:
                universal_metadata["workflow"] = workflow
        
        # Generate dependencies if requested
        dependencies = ""
        if include_dependencies:
            dependencies = self._generate_dependency_list(universal_metadata)
        
        # Convert to target service format
        service_metadata = self._convert_from_universal(universal_metadata, target_service)
        
        return (
            image,
            json.dumps(universal_metadata, indent=2),
            service_metadata,
            dependencies
        )
    
    def _extract_source_metadata(self, image) -> Dict:
        """Extract metadata from image regardless of source format"""
        metadata = {}
        
        # Check PNG text chunks
        if hasattr(image, 'text') and image.text:
            # A1111/Civitai parameters format
            if 'parameters' in image.text:
                metadata.update(self._parse_a1111_parameters(image.text['parameters']))
            
            # ComfyUI workflow format  
            if 'workflow' in image.text:
                try:
                    metadata['comfyui_workflow'] = json.loads(image.text['workflow'])
                except:
                    pass
            
            # ComfyUI prompt format
            if 'prompt' in image.text:
                try:
                    metadata['comfyui_prompt'] = json.loads(image.text['prompt'])
                except:
                    pass
            
            # Custom "meta" chunk (our universal format)
            if 'meta' in image.text:
                try:
                    metadata.update(json.loads(image.text['meta']))
                except:
                    pass
        
        # Check EXIF data
        if hasattr(image, '_getexif') and image._getexif():
            exif_data = image._getexif()
            # Look for AI generation data in EXIF
            metadata.update(self._parse_exif_metadata(exif_data))
        
        return metadata
    
    def _parse_a1111_parameters(self, params_text: str) -> Dict:
        """Parse A1111/Civitai parameters format"""
        metadata = {}
        lines = params_text.strip().split('\n')
        
        if lines:
            # First line is usually the positive prompt
            metadata['prompt'] = lines[0].strip()
            
            # Look for negative prompt
            for i, line in enumerate(lines[1:], 1):
                if line.startswith('Negative prompt:'):
                    metadata['negative_prompt'] = line.replace('Negative prompt:', '').strip()
                    continue
                
                # Parse parameter line (usually the last line)
                if ',' in line and ':' in line:
                    params = self._parse_parameter_line(line)
                    metadata.update(params)
        
        return metadata
    
    def _parse_parameter_line(self, line: str) -> Dict:
        """Parse A1111 parameter line"""
        params = {}
        
        # Split by comma, but be careful of commas in quoted values
        parts = []
        current = ""
        in_quotes = False
        
        for char in line:
            if char == '"' and (not current or current[-1] != '\\'):
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            parts.append(current.strip())
        
        # Parse each parameter
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Convert known numeric values
                if key in ['steps', 'width', 'height', 'seed', 'clip_skip']:
                    try:
                        params[key] = int(value)
                    except:
                        params[key] = value
                elif key in ['cfg_scale', 'denoising_strength', 'eta']:
                    try:
                        params[key] = float(value)
                    except:
                        params[key] = value
                else:
                    params[key] = value
        
        return params
    
    def _detect_source_service(self, metadata: Dict) -> str:
        """Detect which service generated the image based on metadata"""
        if 'comfyui_workflow' in metadata:
            return 'comfyui'
        elif 'model_hash' in metadata and 'sampler' in metadata:
            return 'automatic1111'
        elif 'tensor_ai_style' in metadata:
            return 'tensor.ai'
        elif 'leonardo_preset' in metadata:
            return 'leonardo.ai'
        elif 'midjourney_version' in metadata:
            return 'midjourney'
        else:
            return 'generic'
    
    def _convert_to_universal(self, source_metadata: Dict, source_service: str) -> Dict:
        """Convert source metadata to universal format"""
        universal = {
            "schema_version": self.universal_schema["schema_version"],
            "source_service": source_service,
            "creation_time": datetime.now().isoformat(),
            "metaman_version": "1.0.0",
            "metadata": {}
        }
        
        # Map source fields to universal schema
        for field_name, field_config in self.universal_schema["fields"].items():
            if source_service in field_config.get("supported_by", []):
                # Direct mapping
                if field_name in source_metadata:
                    universal["metadata"][field_name] = source_metadata[field_name]
                # Handle mappings
                elif "mappings" in field_config and source_service in field_config["mappings"]:
                    source_value = source_metadata.get(field_name)
                    if source_value:
                        mapping = field_config["mappings"][source_service]
                        if isinstance(mapping, dict):
                            # Reverse lookup for source->universal conversion
                            for universal_val, source_val in mapping.items():
                                if source_val == source_value:
                                    universal["metadata"][field_name] = universal_val
                                    break
                            else:
                                universal["metadata"][field_name] = source_value
                        else:
                            universal["metadata"][field_name] = source_value
        
        return universal
    
    def _convert_from_universal(self, universal_metadata: Dict, target_service: str) -> str:
        """Convert universal metadata to target service format"""
        if target_service not in self.SUPPORTED_SERVICES:
            return f"Unsupported target service: {target_service}"
        
        # Use service template if available
        if target_service in self.service_templates:
            return self._apply_service_template(universal_metadata, target_service)
        else:
            return self._generate_default_format(universal_metadata, target_service)
    
    def _apply_service_template(self, universal_metadata: Dict, target_service: str) -> str:
        """Apply service-specific template for output formatting"""
        template = self.service_templates[target_service]
        metadata = universal_metadata.get("metadata", {})
        
        if target_service == "automatic1111":
            return self._format_a1111_output(metadata, template)
        elif target_service == "comfyui":
            return self._format_comfyui_output(metadata, template)
        else:
            return json.dumps(metadata, indent=2)
    
    def _format_a1111_output(self, metadata: Dict, template: Dict) -> str:
        """Format metadata for A1111 compatibility"""
        lines = []
        
        # Positive prompt
        if 'prompt' in metadata:
            lines.append(metadata['prompt'])
        
        # Negative prompt
        if 'negative_prompt' in metadata:
            lines.append(f"Negative prompt: {metadata['negative_prompt']}")
        
        # Parameters line
        params = []
        param_order = template.get("parameter_order", [
            "steps", "sampler", "cfg_scale", "seed", "width", "height", 
            "model_hash", "clip_skip", "denoising_strength"
        ])
        
        for param in param_order:
            if param in metadata and metadata[param] is not None:
                if param == "cfg_scale":
                    params.append(f"CFG scale: {metadata[param]}")
                elif param == "model_hash":
                    params.append(f"Model hash: {metadata[param]}")
                elif param == "clip_skip":
                    params.append(f"Clip skip: {metadata[param]}")
                elif param == "denoising_strength":
                    params.append(f"Denoising strength: {metadata[param]}")
                elif param == "width" and "height" in metadata:
                    params.append(f"Size: {metadata['width']}x{metadata['height']}")
                elif param != "height":  # Skip height since it's handled with width
                    params.append(f"{param.title()}: {metadata[param]}")
        
        if params:
            lines.append(", ".join(params))
        
        return "\n".join(lines)
    
    def _format_comfyui_output(self, metadata: Dict, template: Dict) -> str:
        """Format metadata for ComfyUI compatibility"""
        # For ComfyUI, we primarily return the workflow if available
        if 'comfyui_workflow' in metadata:
            return json.dumps(metadata['comfyui_workflow'], indent=2)
        else:
            # Generate basic ComfyUI-compatible metadata
            comfyui_meta = {}
            for field, value in metadata.items():
                if field in self.universal_schema["fields"]:
                    field_config = self.universal_schema["fields"][field]
                    if "comfyui" in field_config.get("supported_by", []):
                        # Apply mappings if available
                        if "mappings" in field_config and "comfyui" in field_config["mappings"]:
                            mapping = field_config["mappings"]["comfyui"]
                            if isinstance(mapping, dict) and value in mapping:
                                comfyui_meta[field] = mapping[value]
                            else:
                                comfyui_meta[field] = value
                        else:
                            comfyui_meta[field] = value
            
            return json.dumps(comfyui_meta, indent=2)
    
    def _generate_default_format(self, universal_metadata: Dict, target_service: str) -> str:
        """Generate default format for services without specific templates"""
        metadata = universal_metadata.get("metadata", {})
        filtered = {}
        
        # Only include fields supported by target service
        for field, value in metadata.items():
            if field in self.universal_schema["fields"]:
                field_config = self.universal_schema["fields"][field]
                if target_service in field_config.get("supported_by", []):
                    filtered[field] = value
        
        return json.dumps(filtered, indent=2)
    
    def _generate_dependency_list(self, universal_metadata: Dict) -> str:
        """Generate list of model dependencies with download URLs"""
        dependencies = []
        metadata = universal_metadata.get("metadata", {})
        
        # Main model
        if 'model_name' in metadata:
            dep = {
                "type": "checkpoint",
                "name": metadata['model_name'],
                "hash": metadata.get('model_hash', ''),
                "download_url": metadata.get('model_download_url', ''),
                "sources": self._find_model_sources(metadata.get('model_name', ''), metadata.get('model_hash', ''))
            }
            dependencies.append(dep)
        
        # LoRAs
        if 'loras' in metadata and isinstance(metadata['loras'], list):
            for lora in metadata['loras']:
                dep = {
                    "type": "lora",
                    "name": lora.get('name', ''),
                    "weight": lora.get('weight', 1.0),
                    "hash": lora.get('hash', ''),
                    "download_url": lora.get('download_url', ''),
                    "sources": self._find_model_sources(lora.get('name', ''), lora.get('hash', ''))
                }
                dependencies.append(dep)
        
        # Embeddings
        if 'embeddings' in metadata and isinstance(metadata['embeddings'], list):
            for embedding in metadata['embeddings']:
                dep = {
                    "type": "embedding",
                    "name": embedding.get('name', ''),
                    "hash": embedding.get('hash', ''),
                    "download_url": embedding.get('download_url', ''),
                    "sources": self._find_model_sources(embedding.get('name', ''), embedding.get('hash', ''))
                }
                dependencies.append(dep)
        
        return json.dumps(dependencies, indent=2)
    
    def _find_model_sources(self, name: str, hash_value: str) -> List[Dict]:
        """Find download sources for a model (placeholder for API integration)"""
        sources = []
        
        # TODO: Implement actual API calls to Civitai, HuggingFace, etc.
        # This would search by name and/or hash to find download URLs
        
        # Civitai search (placeholder)
        if name:
            sources.append({
                "platform": "civitai",
                "search_url": f"https://civitai.com/api/v1/models?query={name}",
                "confidence": "medium"
            })
        
        # HuggingFace search (placeholder) 
        if name:
            sources.append({
                "platform": "huggingface",
                "search_url": f"https://huggingface.co/models?search={name}",
                "confidence": "medium"
            })
        
        return sources
    
    def _extract_workflow(self, image, source_service: str) -> Optional[Dict]:
        """Extract workflow information if available"""
        if hasattr(image, 'text') and image.text:
            if 'workflow' in image.text:
                try:
                    return json.loads(image.text['workflow'])
                except:
                    pass
        return None
    
    def _save_workflow(self, image, output_format: str):
        """Save workflow as file or embed in image"""
        # Implementation for workflow saving
        return (image, "Workflow save not yet implemented", "", "")
    
    def _generate_dependencies(self, image, dependency_sources: str):
        """Generate dependency information"""
        # Implementation for dependency generation
        return (image, "Dependency generation not yet implemented", "", "")
    
    def _export_metadata(self, image, target_service: str, output_format: str):
        """Export metadata in specified format"""
        # Implementation for metadata export
        return (image, "Metadata export not yet implemented", "", "")
    
    def _parse_exif_metadata(self, exif_data: Dict) -> Dict:
        """Parse EXIF data for AI generation information"""
        # Implementation for EXIF parsing
        return {}


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MetaManUniversalNode": MetaManUniversalNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaManUniversalNode": "MetaMan Universal"
}
