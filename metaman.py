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
from typing import Optional, Any, Union
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
                "target_service": (cls.SUPPORTED_SERVICES, {"default": "automatic1111"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata_json")
    FUNCTION = "process_metadata"
    CATEGORY = "MetaMan"
    DESCRIPTION = "Universal metadata conversion and embedding for AI image generation platforms"

    def __init__(self):
        """Initialize MetaMan with universal schema and service templates"""
        self.schema_path = os.path.join(os.path.dirname(__file__), "templates", "universal_schema.yaml")
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates", "services")
        self.universal_schema = self._load_universal_schema()
        self.service_templates = self._load_service_templates()
        
    def _load_universal_schema(self) -> dict:
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
    
    def _create_default_schema(self) -> dict:
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

    def _load_service_templates(self) -> dict:
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

    def process_metadata(self, image, target_service):
        """
        Main processing function: extract metadata, convert to target service, and embed in image
        """
        try:
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                img_tensor = image[0]
                img_array = (img_tensor.cpu().numpy() * 255).astype('uint8')
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = image
            
            # Extract source metadata and detect service
            source_metadata = self._extract_source_metadata(pil_image)
            source_service = self._detect_source_service(source_metadata)
            
            # Convert to universal format
            universal_metadata = self._convert_to_universal(source_metadata, source_service)
            
            # Add workflow extraction
            workflow = self._extract_workflow(pil_image, source_service)
            if workflow:
                universal_metadata["workflow"] = workflow
            
            # Generate dependencies using Civitai API
            dependencies = self._generate_civitai_dependencies(universal_metadata)
            if dependencies:
                universal_metadata["dependencies"] = dependencies
            
            # Convert to target service format
            target_metadata = self._convert_from_universal(universal_metadata, target_service)
            
            # Embed target service metadata in image
            result_image = self._embed_metadata_in_image(pil_image, target_metadata, target_service)
            
            # Prepare JSON output with all metadata
            json_output = {
                "source_service": source_service,
                "target_service": target_service,
                "universal_metadata": universal_metadata,
                "target_metadata": target_metadata,
                "conversion_timestamp": datetime.now().isoformat()
            }
            
            # Convert back to tensor if needed
            if isinstance(image, torch.Tensor):
                if isinstance(result_image, Image.Image):
                    import numpy as np
                    img_array = np.array(result_image).astype(np.float32) / 255.0
                    output_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    return (output_tensor, json.dumps(json_output, indent=2))
            
            return (result_image, json.dumps(json_output, indent=2))
            
        except Exception as e:
            error_msg = f"MetaMan Error: {str(e)}"
            return (image, json.dumps({"error": error_msg}, indent=2))
    
    def _embed_metadata_in_image(self, image: Image.Image, metadata: str, target_service: str) -> Image.Image:
        """
        Embed target service metadata into image PNG chunks
        """
        # Create new image copy
        result_image = image.copy() if hasattr(image, 'copy') else image
        
        # Prepare PNG info
        png_info = PngInfo()
        
        # Copy existing metadata
        if hasattr(image, 'text') and image.text:
            for key, value in image.text.items():
                png_info.add_text(key, value)
        
        # Add target service metadata
        if target_service == "automatic1111" or target_service == "civitai":
            # A1111/Civitai uses 'parameters' chunk
            png_info.add_text("parameters", metadata)
        elif target_service == "comfyui":
            # ComfyUI uses 'workflow' and 'prompt' chunks
            try:
                # If metadata is JSON, try to extract workflow/prompt
                if metadata.startswith('{'):
                    data = json.loads(metadata)
                    if 'comfyui_workflow' in data:
                        png_info.add_text("workflow", json.dumps(data['comfyui_workflow']))
                    if 'comfyui_prompt' in data:
                        png_info.add_text("prompt", json.dumps(data['comfyui_prompt']))
                else:
                    # Fallback: add as workflow
                    png_info.add_text("workflow", metadata)
            except:
                png_info.add_text("workflow", metadata)
        else:
            # For other services, use service name as chunk name
            chunk_name = target_service.replace('.', '_')
            png_info.add_text(chunk_name, metadata)
        
        # Add MetaMan universal chunk
        metaman_data = {
            'schema_version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'metaman_version': '1.0.0',
            'target_service': target_service,
            'original_metadata': metadata
        }
        png_info.add_text("meta", json.dumps(metaman_data, separators=(',', ':')))
        
        # Save to temporary location to embed metadata
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            result_image.save(tmp_file, "PNG", pnginfo=png_info)
            tmp_file.flush()
            
            # Reload image with embedded metadata
            result_image = Image.open(tmp_file.name)
            result_image.load()  # Ensure image is fully loaded
            
            # Clean up temp file
            os.unlink(tmp_file.name)
        
        return result_image
    
    def _extract_source_metadata(self, image) -> dict:
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
    
    def _parse_a1111_parameters(self, params_text: str) -> dict:
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
    
    def _parse_parameter_line(self, line: str) -> dict:
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
    
    def _detect_source_service(self, metadata: dict) -> str:
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
    
    def _convert_to_universal(self, source_metadata: dict, source_service: str) -> dict:
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
    
    def _convert_from_universal(self, universal_metadata: dict, target_service: str) -> str:
        """Convert universal metadata to target service format"""
        if target_service not in self.SUPPORTED_SERVICES:
            return f"Unsupported target service: {target_service}"
        
        # Use service template if available
        if target_service in self.service_templates:
            return self._apply_service_template(universal_metadata, target_service)
        else:
            return self._generate_default_format(universal_metadata, target_service)
    
    def _apply_service_template(self, universal_metadata: dict, target_service: str) -> str:
        """Apply service-specific template for output formatting"""
        template = self.service_templates[target_service]
        metadata = universal_metadata.get("metadata", {})
        
        if target_service == "automatic1111":
            return self._format_a1111_output(metadata, template)
        elif target_service == "comfyui":
            return self._format_comfyui_output(metadata, template)
        else:
            return json.dumps(metadata, indent=2)
    
    def _format_a1111_output(self, metadata: dict, template: dict) -> str:
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
    
    def _format_comfyui_output(self, metadata: dict, template: dict) -> str:
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
    
    def _generate_default_format(self, universal_metadata: dict, target_service: str) -> str:
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
    
    def _generate_civitai_dependencies(self, universal_metadata: dict) -> list[dict]:
        """
        Generate model dependencies using Civitai API
        """
        dependencies = []
        metadata = universal_metadata.get("metadata", {})
        
        # Main model
        if 'model_name' in metadata and metadata['model_name']:
            dep = self._search_civitai_model(metadata['model_name'], 'checkpoint', metadata.get('model_hash', ''))
            if dep:
                dependencies.append(dep)
        
        # LoRAs
        if 'loras' in metadata and isinstance(metadata['loras'], list):
            for lora in metadata['loras']:
                if isinstance(lora, dict) and lora.get('name'):
                    dep = self._search_civitai_model(lora['name'], 'lora', lora.get('hash', ''), lora.get('weight', 1.0))
                    if dep:
                        dependencies.append(dep)
        
        # Embeddings
        if 'embeddings' in metadata and isinstance(metadata['embeddings'], list):
            for embedding in metadata['embeddings']:
                if isinstance(embedding, dict) and embedding.get('name'):
                    dep = self._search_civitai_model(embedding['name'], 'embedding', embedding.get('hash', ''))
                    if dep:
                        dependencies.append(dep)
        
        return dependencies
    
    def _search_civitai_model(self, name: str, model_type: str, hash_value: str = '', weight: float = None) -> Optional[dict]:
        """
        Search for a model on Civitai API
        """
        try:
            import requests
            import time
            
            # Rate limiting
            time.sleep(0.5)
            
            # Search by name first
            search_url = f"https://civitai.com/api/v1/models?query={name}&limit=5"
            
            response = requests.get(search_url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            items = data.get('items', [])
            
            if not items:
                return None
            
            # Find best match
            best_match = None
            for item in items:
                # Check if name matches (case insensitive)
                if name.lower() in item.get('name', '').lower():
                    # If we have a hash, try to match it
                    if hash_value:
                        for version in item.get('modelVersions', []):
                            for file in version.get('files', []):
                                if file.get('hashes', {}).get('SHA256', '').startswith(hash_value[:10]):
                                    best_match = item
                                    break
                            if best_match:
                                break
                    else:
                        best_match = item
                        break
            
            if not best_match:
                best_match = items[0]  # Fallback to first result
            
            # Extract information
            model_info = {
                'type': model_type,
                'name': name,
                'civitai_id': best_match.get('id'),
                'civitai_name': best_match.get('name'),
                'creator': best_match.get('creator', {}).get('username', ''),
                'download_url': '',
                'hash': hash_value,
                'confidence': 0.8 if hash_value else 0.6
            }
            
            if weight is not None:
                model_info['weight'] = weight
            
            # Get download URL from latest version
            versions = best_match.get('modelVersions', [])
            if versions:
                latest_version = versions[0]
                files = latest_version.get('files', [])
                if files:
                    model_info['download_url'] = files[0].get('downloadUrl', '')
                    model_info['civitai_version_id'] = latest_version.get('id')
            
            return model_info
            
        except Exception as e:
            print(f"MetaMan: Error searching Civitai for {name}: {e}")
            return None
    
    def _extract_workflow(self, image, source_service: str) -> Optional[dict]:
        """Extract workflow information if available"""
        if hasattr(image, 'text') and image.text:
            if 'workflow' in image.text:
                try:
                    return json.loads(image.text['workflow'])
                except:
                    pass
        return None
    

    
    def _parse_exif_metadata(self, exif_data: dict) -> dict:
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
