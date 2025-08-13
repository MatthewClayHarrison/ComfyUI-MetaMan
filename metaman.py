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


class MetaManLoadAndConvert:
    """
    Load image file, extract metadata, and convert to target service format
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image_file": (sorted(files), {"image_upload": True}),
                "target_service": (["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "leonardo.ai"], {"default": "automatic1111"})
            }
        }
    
    CATEGORY = "MetaMan"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "converted_metadata")
    FUNCTION = "load_and_convert"
    DESCRIPTION = "Load image with metadata and convert to target service format"
    
    def load_and_convert(self, image_file, target_service):
        """
        Load image, extract metadata, and convert to target format
        """
        try:
            input_dir = folder_paths.get_input_directory()
            image_path = folder_paths.get_annotated_filepath(image_file, input_dir)
            
            print(f"MetaMan Load & Convert: Loading {image_path}")
            
            # Load image with PIL to preserve metadata
            pil_image = Image.open(image_path)
            
            # Extract metadata from the original file
            source_metadata = self._extract_metadata_from_image(pil_image, image_path)
            
            # Detect source service
            source_service = self._detect_source_service(source_metadata)
            print(f"MetaMan Load & Convert: Detected source service: {source_service}")
            
            # Convert to target service format
            converted_metadata = self._convert_metadata(source_metadata, source_service, target_service)
            
            # Convert PIL image to tensor for ComfyUI
            import numpy as np
            img_array = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            print(f"MetaMan Load & Convert: Successfully converted from {source_service} to {target_service}")
            
            return (img_tensor, converted_metadata)
            
        except Exception as e:
            print(f"MetaMan Load & Convert Error: {e}")
            # Return empty tensor and error message
            empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            error_msg = f"Error: {str(e)}"
            return (empty_tensor, error_msg)
    
    def _extract_metadata_from_image(self, pil_image, file_path):
        """Extract all available metadata from PIL image"""
        metadata = {}
        
        try:
            # PNG text chunks
            if hasattr(pil_image, 'text') and pil_image.text:
                print(f"MetaMan Load & Convert: Found PNG chunks: {list(pil_image.text.keys())}")
                
                for key, value in pil_image.text.items():
                    # Store raw chunk data
                    metadata[f"png_chunk_{key}"] = value
                    
                    # Parse specific formats
                    if key == 'parameters':
                        # A1111/Civitai parameters
                        try:
                            parsed_params = self._parse_a1111_parameters(value)
                            metadata.update(parsed_params)
                        except Exception as e:
                            print(f"MetaMan Load & Convert: Error parsing A1111 parameters: {e}")
                    
                    elif key in ['workflow', 'prompt']:
                        # ComfyUI workflow/prompt data
                        try:
                            json_data = json.loads(value)
                            metadata[f"comfyui_{key}"] = json_data
                            
                            # If this is prompt data, extract parameters immediately
                            if key == 'prompt':
                                extracted_params = self._extract_params_from_comfyui_prompt(json_data)
                                metadata.update(extracted_params)
                                
                        except Exception as e:
                            print(f"MetaMan Load & Convert: Error parsing ComfyUI {key}: {e}")
            
        except Exception as e:
            print(f"MetaMan Load & Convert: Error extracting metadata: {e}")
        
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
        """Parse A1111 parameter line with improved handling of complex prompts"""
        params = {}
        
        # Enhanced splitting that handles quotes, parentheses, and brackets
        parts = []
        current = ""
        in_quotes = False
        paren_depth = 0
        bracket_depth = 0
        
        for char in line:
            if char == '"' and (not current or current[-1] != '\\'):
                in_quotes = not in_quotes
            elif char == '(' and not in_quotes:
                paren_depth += 1
            elif char == ')' and not in_quotes:
                paren_depth -= 1
            elif char == '[' and not in_quotes:
                bracket_depth += 1
            elif char == ']' and not in_quotes:
                bracket_depth -= 1
            elif char == ',' and not in_quotes and paren_depth == 0 and bracket_depth == 0:
                # Only split on commas that are not inside quotes, parentheses, or brackets
                part = current.strip()
                if part:  # Only add non-empty parts
                    parts.append(part)
                current = ""
                continue
            current += char
        
        # Don't forget the last part
        if current.strip():
            parts.append(current.strip())
        
        print(f"MetaMan Debug: Parsed {len(parts)} parameter parts from line")
        
        # Parse each parameter
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if ':' in part:
                # Find the first colon that's not inside parentheses or brackets
                colon_pos = -1
                paren_depth = 0
                bracket_depth = 0
                in_quotes = False
                
                for i, char in enumerate(part):
                    if char == '"' and (i == 0 or part[i-1] != '\\'):
                        in_quotes = not in_quotes
                    elif char == '(' and not in_quotes:
                        paren_depth += 1
                    elif char == ')' and not in_quotes:
                        paren_depth -= 1
                    elif char == '[' and not in_quotes:
                        bracket_depth += 1
                    elif char == ']' and not in_quotes:
                        bracket_depth -= 1
                    elif char == ':' and not in_quotes and paren_depth == 0 and bracket_depth == 0:
                        colon_pos = i
                        break
                
                if colon_pos > 0:
                    key = part[:colon_pos].strip().lower().replace(' ', '_')
                    value = part[colon_pos + 1:].strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    
                    print(f"MetaMan Debug: Parsed parameter '{key}': '{value}' (length: {len(value)})")
                    
                    # Convert known numeric values
                    if key in ['steps', 'width', 'height', 'seed', 'clip_skip']:
                        try:
                            params[key] = int(value)
                        except ValueError:
                            params[key] = value
                    elif key in ['cfg_scale', 'denoising_strength', 'eta']:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            params[key] = value
                    else:
                        params[key] = value
                else:
                    print(f"MetaMan Debug: Skipping malformed parameter part: '{part}'")
            else:
                print(f"MetaMan Debug: Skipping non-parameter part: '{part}'")
        
        print(f"MetaMan Debug: Final parsed parameters: {list(params.keys())}")
        return params
    
    def _extract_params_from_comfyui_prompt(self, prompt_data: dict) -> dict:
        """Extract generation parameters from ComfyUI prompt data"""
        params = {}
        
        try:
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue
                    
                class_type = node_data.get('class_type', '')
                inputs = node_data.get('inputs', {})
                
                # Extract from KSampler nodes
                if class_type == 'KSampler':
                    if 'steps' in inputs:
                        params['steps'] = inputs['steps']
                    if 'cfg' in inputs:
                        params['cfg_scale'] = inputs['cfg']
                    if 'sampler_name' in inputs:
                        params['sampler'] = inputs['sampler_name']
                    if 'scheduler' in inputs:
                        params['scheduler'] = inputs['scheduler']
                    if 'seed' in inputs:
                        params['seed'] = inputs['seed']
                    if 'denoise' in inputs:
                        params['denoising_strength'] = inputs['denoise']
                
                # Extract from CheckpointLoaderSimple
                elif class_type == 'CheckpointLoaderSimple':
                    if 'ckpt_name' in inputs:
                        params['model_name'] = inputs['ckpt_name']
                
                # Extract from CLIPTextEncode (prompts)
                elif class_type == 'CLIPTextEncode':
                    if 'text' in inputs:
                        # First positive prompt we find
                        if 'prompt' not in params and inputs['text'].strip():
                            params['prompt'] = inputs['text']
                        # If we already have a prompt, this might be negative
                        elif 'negative_prompt' not in params and inputs['text'].strip():
                            # Simple heuristic: if it contains common negative words
                            negative_indicators = ['worst', 'low quality', 'blurry', 'bad', 'ugly', 'deformed']
                            if any(indicator in inputs['text'].lower() for indicator in negative_indicators):
                                params['negative_prompt'] = inputs['text']
                
                # Extract from EmptyLatentImage (dimensions)
                elif class_type == 'EmptyLatentImage':
                    if 'width' in inputs:
                        params['width'] = inputs['width']
                    if 'height' in inputs:
                        params['height'] = inputs['height']
                
                # Extract from LoraLoader nodes
                elif class_type == 'LoraLoader':
                    if 'lora_name' in inputs and 'strength_model' in inputs:
                        if 'loras' not in params:
                            params['loras'] = []
                        lora_info = {
                            'name': inputs['lora_name'],
                            'weight': inputs['strength_model']
                        }
                        if 'strength_clip' in inputs:
                            lora_info['clip_weight'] = inputs['strength_clip']
                        params['loras'].append(lora_info)
        
        except Exception as e:
            print(f"MetaMan Load & Convert: Error extracting ComfyUI params: {e}")
        
        return params
    
    def _detect_source_service(self, metadata: dict) -> str:
        """Detect which service generated the image based on metadata"""
        # ComfyUI indicators
        comfyui_indicators = [
            'comfyui_workflow' in metadata,
            'comfyui_prompt' in metadata,
            any(key.startswith('comfyui_') for key in metadata.keys())
        ]
        
        if any(comfyui_indicators):
            return 'comfyui'
        elif 'model_hash' in metadata and 'sampler' in metadata:
            return 'automatic1111'
        elif 'prompt' in metadata and isinstance(metadata['prompt'], str):
            if any(key in metadata for key in ['steps', 'cfg_scale', 'seed']):
                return 'automatic1111'
        
        return 'generic'
    
    def _convert_metadata(self, source_metadata: dict, source_service: str, target_service: str) -> str:
        """Convert metadata to target service format"""
        try:
            if target_service == "automatic1111":
                return self._format_a1111_output(source_metadata)
            elif target_service == "comfyui":
                return self._format_comfyui_output(source_metadata)
            else:
                # Generic JSON format
                filtered = {k: v for k, v in source_metadata.items() if not k.startswith('png_chunk_')}
                return json.dumps(filtered, indent=2)
        except Exception as e:
            return f"Conversion error: {str(e)}"
    
    def _format_a1111_output(self, metadata: dict) -> str:
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
        param_order = ["steps", "sampler", "cfg_scale", "seed", "width", "height", "model_name"]
        
        for param in param_order:
            if param in metadata and metadata[param] is not None:
                if param == "cfg_scale":
                    params.append(f"CFG scale: {metadata[param]}")
                elif param == "model_name":
                    params.append(f"Model: {metadata[param]}")
                elif param == "width" and "height" in metadata:
                    params.append(f"Size: {metadata['width']}x{metadata['height']}")
                elif param != "height":  # Skip height since it's handled with width
                    params.append(f"{param.title()}: {metadata[param]}")
        
        if params:
            lines.append(", ".join(params))
        
        return "\n".join(lines)
    
    def _format_comfyui_output(self, metadata: dict) -> str:
        """Format metadata for ComfyUI compatibility"""
        # For ComfyUI, return the workflow if available
        if 'comfyui_workflow' in metadata:
            return json.dumps(metadata['comfyui_workflow'], indent=2)
        elif 'comfyui_prompt' in metadata:
            return json.dumps(metadata['comfyui_prompt'], indent=2)
        else:
            # Generate basic ComfyUI-compatible metadata
            comfyui_meta = {k: v for k, v in metadata.items() if not k.startswith('png_chunk_')}
            return json.dumps(comfyui_meta, indent=2)


class MetaManExtractComponents:
    """
    Extract individual workflow components from metadata JSON
    Perfect for stealing specific elements from images for reuse
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata_json": ("STRING", {"forceInput": True})
            }
        }
    
    CATEGORY = "MetaMan"
    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "STRING", "STRING", "INT", "INT", "INT", "STRING", "STRING", "*", "*", "FLOAT")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "steps", "cfg_scale", "sampler", "scheduler", "seed", "width", "height", "model_name", "vae_name", "loras", "embeddings", "denoising_strength")
    FUNCTION = "extract_components"
    DESCRIPTION = "Extract individual workflow components for reuse in current workflow"
    
    def extract_components(self, metadata_json):
        """
        Parse metadata JSON and extract individual components
        """
        try:
            if not metadata_json or not metadata_json.strip():
                return self._return_empty_components()
            
            # Parse the JSON from MetaManLoadAndConvert
            metadata_data = json.loads(metadata_json)
            
            # Check if this is from our Load & Convert node or raw metadata
            if 'metadata' in metadata_data:
                metadata = metadata_data['metadata']
            else:
                metadata = metadata_data
            
            print(f"MetaMan Extract Components: Processing metadata with keys: {list(metadata.keys())}")
            
            # Extract each component with fallbacks
            positive_prompt = str(metadata.get('prompt', ''))
            negative_prompt = str(metadata.get('negative_prompt', ''))
            steps = int(metadata.get('steps', 20))
            cfg_scale = float(metadata.get('cfg_scale', 7.0))
            sampler = str(metadata.get('sampler', 'euler'))
            scheduler = str(metadata.get('scheduler', 'normal'))
            seed = int(metadata.get('seed', -1))
            width = int(metadata.get('width', 512))
            height = int(metadata.get('height', 512))
            
            # Model name with de-obfuscation support
            model_name = str(metadata.get('model_name_real', metadata.get('model_name', '')))
            
            # VAE name
            vae_name = str(metadata.get('vae_name', ''))
            
            denoising_strength = float(metadata.get('denoising_strength', 1.0))
            
            # Enhanced LoRAs - prefer de-obfuscated version if available (simplified output)
            loras_list = []
            if 'loras_enhanced' in metadata and isinstance(metadata['loras_enhanced'], list):
                # Simplify enhanced LoRAs to only include real_name and weight
                raw_loras = metadata['loras_enhanced']
                for lora in raw_loras:
                    if isinstance(lora, dict):
                        simplified_lora = {
                            'real_name': lora.get('real_name', lora.get('name', '')),
                            'weight': lora.get('weight', 1.0)
                        }
                        loras_list.append(simplified_lora)
                    else:
                        # Handle non-dict LoRAs (fallback)
                        loras_list.append({'real_name': str(lora), 'weight': 1.0})
                print(f"MetaMan Extract Components: Using simplified enhanced LoRAs ({len(loras_list)} LoRAs)")
            elif 'loras' in metadata and isinstance(metadata['loras'], list):
                # Simplify regular LoRAs to only include name (as real_name) and weight
                raw_loras = metadata['loras']
                for lora in raw_loras:
                    if isinstance(lora, dict):
                        simplified_lora = {
                            'real_name': lora.get('name', ''),
                            'weight': lora.get('weight', 1.0)
                        }
                        loras_list.append(simplified_lora)
                    else:
                        # Handle non-dict LoRAs (fallback)
                        loras_list.append({'real_name': str(lora), 'weight': 1.0})
                print(f"MetaMan Extract Components: Using simplified regular LoRAs ({len(loras_list)} LoRAs)")
            
            # Format embeddings as a list
            embeddings_list = []
            if 'embeddings' in metadata and isinstance(metadata['embeddings'], list):
                embeddings_list = metadata['embeddings']  # Return the actual list
            
            print(f"MetaMan Extract Components: Successfully extracted {len([x for x in [positive_prompt, negative_prompt, model_name, vae_name] if x])} text components")
            print(f"MetaMan Extract Components: Extracted {len(loras_list)} simplified LoRAs and {len(embeddings_list)} embeddings")
            
            return (positive_prompt, negative_prompt, steps, cfg_scale, sampler, scheduler, seed, width, height, model_name, vae_name, loras_list, embeddings_list, denoising_strength)
            
        except Exception as e:
            print(f"MetaMan Extract Components Error: {e}")
            return self._return_empty_components()
    
    def _return_empty_components(self):
        """Return empty/default values for all components"""
        return ("", "", 20, 7.0, "euler", "normal", -1, 512, 512, "", "", [], [], 1.0)


class MetaManEmbedAndSave:
    """
    Embed metadata into image and save as PNG file to output directory
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "metadata_json": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "MetaMan_converted"})
            }
        }
    
    CATEGORY = "MetaMan"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "embed_and_save"
    OUTPUT_NODE = True
    DESCRIPTION = "Embed metadata into image and save as PNG file"
    
    def embed_and_save(self, image, metadata_json, filename_prefix="MetaMan_converted"):
        """
        Embed metadata into image and save as PNG file to output directory
        """
        try:
            # Parse metadata JSON to extract target format
            if metadata_json and metadata_json.strip():
                try:
                    metadata_data = json.loads(metadata_json)
                    # Extract target metadata or use universal metadata
                    if 'target_metadata' in metadata_data:
                        metadata_text = metadata_data['target_metadata']
                        print(f"MetaMan Embed & Save: Using target metadata ({len(metadata_text)} chars)")
                    elif 'metadata' in metadata_data:
                        # Convert metadata dict to A1111 format
                        metadata_text = self._format_metadata_for_embedding(metadata_data['metadata'])
                        print(f"MetaMan Embed & Save: Converted metadata to A1111 format ({len(metadata_text)} chars)")
                    else:
                        metadata_text = json.dumps(metadata_data, indent=2)
                        print(f"MetaMan Embed & Save: Using raw JSON ({len(metadata_text)} chars)")
                except Exception as e:
                    print(f"MetaMan Embed & Save: Error parsing metadata JSON: {e}")
                    metadata_text = metadata_json  # Fallback to raw input
            else:
                metadata_text = "MetaMan processed image"
                print(f"MetaMan Embed & Save: No metadata provided, using default")
            
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                img_tensor = image[0]
                img_array = (img_tensor.cpu().numpy() * 255).astype('uint8')
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = image
            
            # Save to ComfyUI output directory with metadata
            output_dir = folder_paths.get_output_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Create PNG info with metadata
            png_info = PngInfo()
            png_info.add_text("parameters", metadata_text)
            png_info.add_text("metaman_converted", datetime.now().isoformat())
            png_info.add_text("metaman_version", "1.0.0")
            
            # Save image with embedded metadata
            pil_image.save(filepath, "PNG", pnginfo=png_info)
            
            print(f"MetaMan Embed & Save: Successfully saved image with metadata to {filepath}")
            print(f"MetaMan Embed & Save: File size: {os.path.getsize(filepath)} bytes")
            
            return {"ui": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}}
            
        except Exception as e:
            print(f"MetaMan Embed & Save Error: {e}")
            return {"ui": {"text": [f"Error: {str(e)}"]}}
    
    def _format_metadata_for_embedding(self, metadata: dict) -> str:
        """Convert metadata dict to A1111 format for embedding"""
        lines = []
        
        # Positive prompt
        if 'prompt' in metadata:
            lines.append(metadata['prompt'])
        
        # Negative prompt
        if 'negative_prompt' in metadata:
            lines.append(f"Negative prompt: {metadata['negative_prompt']}")
        
        # Parameters line
        params = []
        param_order = ["steps", "sampler", "cfg_scale", "seed", "width", "height", "model_name", "scheduler"]
        
        for param in param_order:
            if param in metadata and metadata[param] is not None:
                if param == "cfg_scale":
                    params.append(f"CFG scale: {metadata[param]}")
                elif param == "model_name":
                    params.append(f"Model: {metadata[param]}")
                elif param == "width" and "height" in metadata:
                    params.append(f"Size: {metadata['width']}x{metadata['height']}")
                elif param != "height":  # Skip height since it's handled with width
                    params.append(f"{param.title()}: {metadata[param]}")
        
        if params:
            lines.append(", ".join(params))
        
        return "\n".join(lines)


class MetaManLoadImage:
    """
    Custom Load Image node that preserves PNG metadata
    Designed to work with MetaMan Universal node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            }
        }
    
    CATEGORY = "MetaMan"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata_json")
    FUNCTION = "load_image_with_metadata"
    DESCRIPTION = "Load image while preserving PNG metadata for MetaMan"
    
    def load_image_with_metadata(self, image):
        """
        Load image and extract metadata, returning both
        """
        try:
            input_dir = folder_paths.get_input_directory()
            image_path = folder_paths.get_annotated_filepath(image, input_dir)
            
            print(f"MetaMan Load Image: Loading {image_path}")
            
            # Load image with PIL to preserve metadata
            pil_image = Image.open(image_path)
            print(f"MetaMan Load Image: Image size: {pil_image.size}")
            print(f"MetaMan Load Image: Image format: {pil_image.format}")
            
            # Extract metadata from the original file
            metadata = self._extract_metadata_from_image(pil_image, image_path)
            
            # Convert PIL image to tensor for ComfyUI
            import numpy as np
            img_array = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            # Create metadata JSON output
            metadata_json = {
                "source_file_path": image_path,
                "image_size": pil_image.size,
                "image_format": pil_image.format,
                "metadata": metadata,
                "extracted_at": datetime.now().isoformat()
            }
            
            print(f"MetaMan Load Image: Extracted metadata keys: {list(metadata.keys())}")
            
            return (img_tensor, json.dumps(metadata_json, indent=2))
            
        except Exception as e:
            print(f"MetaMan Load Image Error: {e}")
            # Return empty tensor and error message
            empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            error_json = {"error": str(e), "source_file_path": image_path if 'image_path' in locals() else "unknown"}
            return (empty_tensor, json.dumps(error_json, indent=2))
    
    def _extract_metadata_from_image(self, pil_image, file_path):
        """Extract all available metadata from PIL image"""
        metadata = {}
        
        try:
            # PNG text chunks
            if hasattr(pil_image, 'text') and pil_image.text:
                print(f"MetaMan Load Image: Found PNG text chunks: {list(pil_image.text.keys())}")
                
                for key, value in pil_image.text.items():
                    print(f"MetaMan Load Image: Chunk '{key}': {len(value)} characters")
                    
                    # Store raw chunk data
                    metadata[f"png_chunk_{key}"] = value
                    
                    # Parse specific formats
                    if key == 'parameters':
                        # A1111/Civitai parameters
                        try:
                            parsed_params = self._parse_a1111_parameters(value)
                            metadata.update(parsed_params)
                            print(f"MetaMan Load Image: Parsed A1111 parameters: {list(parsed_params.keys())}")
                        except Exception as e:
                            print(f"MetaMan Load Image: Error parsing A1111 parameters: {e}")
                    
                    elif key in ['workflow', 'prompt']:
                        # ComfyUI workflow/prompt data
                        try:
                            json_data = json.loads(value)
                            metadata[f"comfyui_{key}"] = json_data
                            print(f"MetaMan Load Image: Parsed ComfyUI {key}: {len(json_data)} nodes")
                            
                            # If this is prompt data, extract parameters immediately
                            if key == 'prompt':
                                extracted_params = self._extract_params_from_comfyui_prompt(json_data)
                                metadata.update(extracted_params)
                                
                                # Phase: De-obfuscate Tensor.AI model names if available (after parameter extraction)
                                if extracted_params:
                                    deobfuscated_info = self._deobfuscate_tensor_ai_models(metadata, extracted_params.get('loras', []))
                                    if deobfuscated_info:
                                        metadata.update(deobfuscated_info)
                                        print(f"MetaMan Load Image: De-obfuscated Tensor.AI models: {list(deobfuscated_info.keys())}")
                                
                                print(f"MetaMan Load Image: Extracted ComfyUI params: {list(extracted_params.keys())}")
                                
                        except Exception as e:
                            print(f"MetaMan Load Image: Error parsing ComfyUI {key}: {e}")
                    
                    elif key == 'meta':
                        # MetaMan universal format
                        try:
                            meta_data = json.loads(value)
                            metadata.update(meta_data)
                            print(f"MetaMan Load Image: Parsed MetaMan meta chunk")
                        except Exception as e:
                            print(f"MetaMan Load Image: Error parsing meta chunk: {e}")
            else:
                print(f"MetaMan Load Image: No PNG text chunks found")
            
            # Check image.info as fallback
            if hasattr(pil_image, 'info') and pil_image.info:
                print(f"MetaMan Load Image: Found image.info keys: {list(pil_image.info.keys())}")
                for key, value in pil_image.info.items():
                    if isinstance(value, str) and len(value) > 50:
                        metadata[f"info_{key}"] = value
                        print(f"MetaMan Load Image: Stored info['{key}']: {len(value)} characters")
            
        except Exception as e:
            print(f"MetaMan Load Image: Error extracting metadata: {e}")
        
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
    
    def _extract_params_from_comfyui_prompt(self, prompt_data: dict) -> dict:
        """Universal dynamic extraction from ComfyUI prompt data - works with any node ecosystem"""
        params = {}
        
        print(f"MetaMan Universal: Starting extraction from {len(prompt_data)} nodes")
        
        try:
            # Phase 1: Extract core parameters universally
            self._extract_core_parameters_universal(prompt_data, params)
            
            # Phase 2: Collect ALL text content from all nodes
            all_text_content = self._scan_all_text_content_universal(prompt_data)
            print(f"MetaMan Universal: Found {len(all_text_content)} text candidates")
            
            # Phase 3: Classify text content using improved heuristics
            classified_prompts = self._classify_prompts_intelligent(all_text_content)
            
            # Phase 4: Extract LoRAs from all text content
            all_loras = self._extract_loras_universal(all_text_content)
            print(f"MetaMan Universal: DEBUG - LoRAs extraction returned: {all_loras}")
            
            # Phase 5: Extract embeddings from all text content
            all_embeddings = self._extract_embeddings_universal(all_text_content)
            print(f"MetaMan Universal: DEBUG - Embeddings extraction returned: {all_embeddings}")
            
            # Phase 6: Resolve model names by tracing connections
            model_info = self._resolve_model_names(prompt_data)
            if model_info:
                params.update(model_info)
                print(f"MetaMan Universal: Resolved models: {model_info}")
                
            # Phase 6.7: Extract VAE information
            vae_info = self._extract_vae_info(prompt_data)
            if vae_info:
                params.update(vae_info)
                print(f"MetaMan Universal: Extracted VAE: {vae_info}")
                
            # Phase 6.8: Determine model type from VAE (helps identify SD1.5 vs SDXL vs Flux)
            model_type = self._determine_model_type(prompt_data)
            if model_type:
                params['model_type'] = model_type
                print(f"MetaMan Universal: Detected model type: {model_type}")
            
            # Phase 7: Compile final results
            if classified_prompts['positive']:
                params['prompt'] = classified_prompts['positive']['text']
                print(f"MetaMan Universal: POSITIVE from {classified_prompts['positive']['source']}: {params['prompt'][:100]}...")
            
            if classified_prompts['negative']:
                params['negative_prompt'] = classified_prompts['negative']['text']
                print(f"MetaMan Universal: NEGATIVE from {classified_prompts['negative']['source']}: {params['negative_prompt'][:100]}...")
            
            if all_loras:
                params['loras'] = all_loras
                print(f"MetaMan Universal: Extracted {len(all_loras)} LoRAs: {[lora['name'] for lora in all_loras]}")
            
            if all_embeddings:
                params['embeddings'] = all_embeddings
                print(f"MetaMan Universal: Extracted {len(all_embeddings)} embeddings: {[emb['name'] for emb in all_embeddings]}")
            
            print(f"MetaMan Universal: Extraction complete: {list(params.keys())}")
            
        except Exception as e:
            print(f"MetaMan Universal Extraction Error: {e}")
            import traceback
            traceback.print_exc()
        
        return params
    
    def _extract_core_parameters_universal(self, prompt_data: dict, params: dict):
        """Extract core parameters from all nodes universally"""
        print(f"MetaMan Universal: Extracting core parameters...")
        
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
                
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            self._extract_core_parameters_from_node(inputs, class_type, params, node_id)
    
    def _extract_core_parameters_from_node(self, inputs: dict, class_type: str, params: dict, node_id: str):
        """Extract core parameters from any node type"""
        # Universal parameter mapping with broader coverage
        param_mappings = {
            'steps': ['steps', 'sampling_steps', 'num_steps', 'inference_steps'],
            'cfg_scale': ['cfg', 'cfg_scale', 'guidance_scale', 'guidance'],
            'sampler': ['sampler_name', 'sampler', 'sampling_method', 'sampler_type'],
            'scheduler': ['scheduler', 'scheduler_name', 'noise_schedule', 'scheduler_type'],
            'seed': ['seed', 'noise_seed', 'random_seed', 'generator_seed'],
            'denoising_strength': ['denoise', 'denoising_strength', 'strength', 'denoise_strength'],
            'width': ['width', 'image_width', 'w', 'img_width'],
            'height': ['height', 'image_height', 'h', 'img_height']
        }
        
        for param_key, possible_fields in param_mappings.items():
            if param_key not in params:  # Don't overwrite existing values
                for field in possible_fields:
                    if field in inputs and inputs[field] is not None:
                        params[param_key] = inputs[field]
                        print(f"MetaMan Universal: Found {param_key} = {inputs[field]} in {node_id} ({class_type})")
                        break
        
        # Special handling for model_name - don't extract node references here
        # Model resolution will be handled separately in _resolve_model_names()
    

    def _extract_loras_from_text(self, text: str) -> list:
        """Extract LoRA information from text using regex"""
        import re
        
        loras = []
        
        # Pattern for <lora:name:weight> or <lora:name>
        lora_pattern = r'<lora:([^>:]+)(?::([0-9.]+))?[^>]*>'
        matches = re.findall(lora_pattern, text, re.IGNORECASE)
        
        for match in matches:
            name = match[0].strip()
            weight = float(match[1]) if match[1] else 1.0
            
            loras.append({
                'name': name,
                'weight': weight
            })
        
        return loras
    
    def _scan_all_text_content_universal(self, prompt_data: dict) -> list:
        """Scan ALL nodes for text content with enhanced detection"""
        text_candidates = []
        
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
                
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            # Enhanced text field detection
            for field_name, field_value in inputs.items():
                if isinstance(field_value, str) and len(field_value.strip()) > 5:
                    text_candidates.append({
                        'text': field_value,
                        'source': f"{node_id}.{field_name}",
                        'node_id': node_id,
                        'field_name': field_name,
                        'node_type': class_type,
                        'length': len(field_value),
                        'is_text_field': self._is_likely_text_field(field_name),
                        'is_processed_field': self._is_processed_text_field(field_name)
                    })
                    
                    print(f"MetaMan Universal: Found text in {node_id}.{field_name} ({class_type}): {len(field_value)} chars")
        
        return text_candidates
    
    def _is_likely_text_field(self, field_name: str) -> bool:
        """Check if field name indicates text content"""
        text_indicators = [
            'text', 'prompt', 'positive', 'negative', 'content', 'description',
            'caption', 'wildcard', 'populated', 'formatted', 'input', 'output'
        ]
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in text_indicators)
    
    def _is_processed_text_field(self, field_name: str) -> bool:
        """Check if field name indicates processed/formatted text"""
        processed_indicators = ['populated', 'formatted', 'processed', 'final', 'output', 'result']
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in processed_indicators)
    
    def _classify_prompts_intelligent(self, text_candidates: list) -> dict:
        """Intelligent classification of text content into positive/negative prompts"""
        classified = {'positive': None, 'negative': None}
        
        # Score all candidates for positive/negative likelihood
        positive_candidates = []
        negative_candidates = []
        
        for candidate in text_candidates:
            if len(candidate['text']) < 20:  # Skip very short text
                continue
                
            pos_score = self._score_as_positive_prompt(candidate)
            neg_score = self._score_as_negative_prompt(candidate)
            
            candidate['positive_score'] = pos_score
            candidate['negative_score'] = neg_score
            
            print(f"MetaMan Universal: {candidate['source']} - Pos: {pos_score:.1f}, Neg: {neg_score:.1f}")
            
            if pos_score > 0.5:
                positive_candidates.append(candidate)
            if neg_score > 0.5:
                negative_candidates.append(candidate)
        
        # Select best positive prompt
        if positive_candidates:
            positive_candidates.sort(key=lambda x: x['positive_score'], reverse=True)
            classified['positive'] = positive_candidates[0]
            print(f"MetaMan Universal: Best positive: {classified['positive']['source']} (score: {classified['positive']['positive_score']:.1f})")
        
        # Select best negative prompt  
        if negative_candidates:
            negative_candidates.sort(key=lambda x: x['negative_score'], reverse=True)
            classified['negative'] = negative_candidates[0]
            print(f"MetaMan Universal: Best negative: {classified['negative']['source']} (score: {classified['negative']['negative_score']:.1f})")
        
        return classified
    
    def _score_as_positive_prompt(self, candidate: dict) -> float:
        """Score text as likely positive prompt with enhanced detection for processed fields"""
        score = 0.0
        text = candidate['text']
        text_lower = text.lower()
        field_name = candidate['field_name'].lower()
        node_type = candidate['node_type']
        
        # CRITICAL: Exclude LoRA-only content from prompt scoring
        if self._is_lora_only_content(text, node_type):
            print(f"MetaMan Universal: EXCLUDING LoRA-only content from prompt scoring: {candidate['source']}")
            return 0.0  # LoRA-only content should not be considered as positive prompt
        
        # CRITICAL: Strong negative indicators should disqualify from positive scoring
        if self._is_clearly_negative_content(text):
            print(f"MetaMan Universal: EXCLUDING clearly negative content from positive scoring: {candidate['source']}")
            return 0.0  # Obviously negative content should not be positive prompt
        
        # ENHANCED: Field name indicators (strong signals)
        if 'positive' in field_name or 'prompt' in field_name:
            score += 3.0
        # PRIORITY: Processed/populated fields are very likely to be the main prompt
        if 'populated' in field_name or 'formatted' in field_name or 'processed' in field_name:
            score += 4.0  # Increased from 2.0 - these are usually the complete prompt
            print(f"MetaMan Universal: HIGH PRIORITY processed field detected: {candidate['source']}")
        # Custom node field detection
        if 'wildcard' in field_name or 'impact' in node_type.lower():
            score += 3.0  # ImpactWildcardEncode is a common complete prompt source
        if 'text' in field_name:
            score += 1.0
        
        # Content indicators
        positive_keywords = [
            'score_9', 'score_8', 'masterpiece', 'best quality', 'high resolution',
            'detailed', 'beautiful', 'professional', 'cinematic', 'photorealistic',
            'ultra', 'highly detailed', 'intricate', 'sharp focus', 'realistic'
        ]
        
        keyword_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        score += keyword_count * 0.5
        
        # LoRA presence (moderate indicator of main prompt, but not if it's LoRA-only)
        if '<lora:' in text:
            # Count human-readable content vs LoRA content
            human_content = self._extract_human_readable_content(text)
            if len(human_content) > 20:  # Has substantial human content
                lora_count = text.count('<lora:')
                score += lora_count * 1.0  # Reduced from 2.0 - LoRAs are good but not overwhelming
                print(f"MetaMan Universal: Found {lora_count} LoRAs with human content in {candidate['source']} (+{lora_count} score)")
            else:
                print(f"MetaMan Universal: LoRAs found but minimal human content in {candidate['source']} (no LoRA bonus)")
        
        # ENHANCED: Length scoring with better thresholds (but capped to prevent negative dominance)
        text_length = len(text)
        if text_length > 1000:
            score += 1.5  # Reduced for very long text - might be negative prompt
        elif text_length > 500:
            score += 2.5  # Reduced
        elif text_length > 200:
            score += 2.0
        elif text_length > 100:
            score += 1.0
        elif text_length > 50:
            score += 0.5
        
        # BREAK indicators (common in ComfyUI prompts)
        if 'BREAK' in text.upper():
            score += 1.0  # BREAK is common in detailed positive prompts
        
        # Enhanced negative indicators (reduce score more aggressively)
        negative_keywords = ['worst', 'low quality', 'bad', 'ugly', 'blurry', 'deformed']
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        if negative_count > 2:
            score -= 3.0  # Increased penalty - likely negative prompt
        if negative_count > 5:
            score -= 5.0  # Very likely negative prompt
        
        return max(0.0, score)
    
    def _is_clearly_negative_content(self, text: str) -> bool:
        """Detect obviously negative content that should never be positive prompt"""
        text_lower = text.lower()
        
        # Check for embedding syntax
        if 'embedding:' in text_lower:
            print(f"MetaMan Universal: Found embedding syntax - likely negative prompt")
            return True
        
        # Well-known negative embedding/model names
        negative_embeddings = [
            'baddream', 'fastnegative', 'easynegative', 'verybadimagenegative',
            'ng_deepnegative', 'negative_hand', 'bad-hands', 'badhandv4',
            'unrealengine', 'badprompt', 'bad-artist', 'bad_prompt_version'
        ]
        
        for neg_embed in negative_embeddings:
            if neg_embed in text_lower:
                print(f"MetaMan Universal: Found negative embedding '{neg_embed}' - clearly negative content")
                return True
        
        # Count negative keywords as percentage of content
        negative_keywords = [
            'worst', 'low quality', 'bad', 'ugly', 'blurry', 'deformed', 'distorted',
            'mutation', 'mutated', 'disfigured', 'dismembered', 'malformed',
            'poorly drawn', 'extra', 'missing', 'cropped', 'watermark', 'text',
            'signature', 'username', 'lowres', 'artifacts', 'duplicate', 'morbid',
            'mutilated', 'fused fingers', 'unclear eyes', 'bad anatomy', 'bad hands'
        ]
        
        neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        # If text has high concentration of negative keywords, it's clearly negative
        if neg_count > 10:  # Lots of negative keywords
            print(f"MetaMan Universal: High negative keyword density ({neg_count}) - clearly negative content")
            return True
        
        # If negative keywords make up significant portion of unique words
        words = set(text_lower.split())
        if len(words) > 0 and neg_count / len(words) > 0.3:  # 30% negative keywords
            print(f"MetaMan Universal: High negative keyword ratio ({neg_count}/{len(words)}) - clearly negative content")
            return True
        
        return False
    
    def _is_lora_only_content(self, text: str, node_type: str) -> bool:
        """Check if text contains only LoRA tags and no meaningful human content"""
        # Check node type first - LoraTagLoader is specifically for LoRA loading
        if 'lora' in node_type.lower() and 'tag' in node_type.lower():
            return True
            
        # Check if text is primarily LoRA tags
        human_content = self._extract_human_readable_content(text)
        lora_content_length = len(text) - len(human_content)
        
        # If most of the content is LoRA tags and little human content, consider it LoRA-only
        if len(human_content.strip()) < 10 and lora_content_length > len(human_content) * 2:
            return True
            
        return False
    
    def _extract_human_readable_content(self, text: str) -> str:
        """Extract human-readable content by removing LoRA tags"""
        import re
        # Remove LoRA tags
        no_loras = re.sub(r'<lora:[^>]*>', '', text, flags=re.IGNORECASE)
        # Remove extra whitespace and commas
        cleaned = re.sub(r'[,\s]+', ' ', no_loras).strip()
        return cleaned
    
    def _score_as_negative_prompt(self, candidate: dict) -> float:
        """Score text as likely negative prompt"""
        score = 0.0
        text = candidate['text']
        text_lower = text.lower()
        field_name = candidate['field_name'].lower()
        
        # Field name indicators
        if 'negative' in field_name:
            score += 5.0  # Very strong indicator
        elif 'bad' in field_name or 'unwanted' in field_name:
            score += 3.0
        
        # Content analysis - negative keywords
        negative_keywords = [
            'worst quality', 'low quality', 'bad', 'ugly', 'blurry', 'deformed',
            'distorted', 'mutation', 'error', 'artifact', 'disfigured',
            'extra limbs', 'missing', 'cropped', 'watermark', 'text', 'signature',
            'username', 'lowres', 'jpeg artifacts', 'duplicate', 'morbid',
            'mutilated', 'poorly drawn', 'extra fingers', 'fused fingers',
            'too many fingers', 'unclear eyes', 'lowers', 'bad anatomy',
            'bad hands', 'missing fingers', 'extra digit', 'fewer digits'
        ]
        
        keyword_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        score += keyword_count * 0.7
        
        # Pattern analysis - negative prompts often have parentheses with weights
        parentheses_content = re.findall(r'\([^)]+:[0-9.]+\)', text)
        if parentheses_content:
            score += len(parentheses_content) * 0.5
        
        # Short text with negative words likely negative prompt
        if len(text) < 200 and keyword_count > 1:
            score += 1.0
        
        # Positive indicators (reduce score)
        positive_keywords = ['masterpiece', 'best quality', 'detailed', 'beautiful']
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        if positive_count > 1:
            score -= 1.5  # Likely positive prompt
        
        # LoRA presence (usually in positive prompts)
        if '<lora:' in text:
            score -= 2.0  # LoRAs typically in positive prompts
        
        return max(0.0, score)
    
    def _extract_loras_from_text(self, text: str) -> list:
        """Extract LoRA information from text using regex"""
        import re
        
        loras = []
        
        # Pattern for <lora:name:weight> or <lora:name>
        lora_pattern = r'<lora:([^>:]+)(?::([0-9.]+))?[^>]*>'
        matches = re.findall(lora_pattern, text, re.IGNORECASE)
        
        for match in matches:
            name = match[0].strip()
            weight = float(match[1]) if match[1] else 1.0
            
            loras.append({
                'name': name,
                'weight': weight
            })
        
        return loras
    
    def _extract_loras_universal(self, text_candidates: list) -> list:
        """Extract LoRAs from all text content"""
        all_loras = []
        seen_loras = set()
        
        for candidate in text_candidates:
            loras_in_text = self._extract_loras_from_text(candidate['text'])
            for lora in loras_in_text:
                lora_key = lora['name'].lower()
                if lora_key not in seen_loras:
                    seen_loras.add(lora_key)
                    all_loras.append(lora)
                    print(f"MetaMan Universal: Found LoRA '{lora['name']}' (weight: {lora['weight']}) in {candidate['source']}")
        
        return all_loras
        
    def _extract_embeddings_universal(self, text_candidates: list) -> list:
        """Extract embeddings from all text content"""
        all_embeddings = []
        seen_embeddings = set()
        
        for candidate in text_candidates:
            embeddings_in_text = self._extract_embeddings_from_text(candidate['text'])
            for embedding in embeddings_in_text:
                embedding_key = embedding['name'].lower()
                if embedding_key not in seen_embeddings:
                    seen_embeddings.add(embedding_key)
                    all_embeddings.append(embedding)
                    print(f"MetaMan Universal: Found embedding '{embedding['name']}' in {candidate['source']}")
        
        return all_embeddings
    
    def _extract_embeddings_from_text(self, text: str) -> list:
        """Extract embedding information from text using regex and known patterns"""
        import re
        
        embeddings = []
        text_lower = text.lower()
        
        # Pattern 1: embedding:name syntax
        embedding_pattern = r'embedding:([\w\-_\.]+)'
        matches = re.findall(embedding_pattern, text, re.IGNORECASE)
        
        for match in matches:
            name = match.strip()
            embeddings.append({
                'name': name,
                'type': 'embedding'
            })
        
        # Pattern 2: Well-known negative embedding names (even without embedding: prefix)
        negative_embeddings = [
            'baddream', 'fastnegative', 'fastnegativev2', 'easynegative', 'verybadimagenegative',
            'ng_deepnegative', 'negative_hand', 'bad-hands', 'badhandv4', 'badhandv5',
            'unrealengine', 'badprompt', 'bad-artist', 'bad_prompt_version', 'badpromptversion',
            'ng_deepnegative_v1_75t', 'negativexl', 'ac_neg1', 'ac_neg2'
        ]
        
        for neg_embed in negative_embeddings:
            # Look for exact word matches (not partial)
            pattern = r'\b' + re.escape(neg_embed) + r'\b'
            if re.search(pattern, text_lower):
                embedding_name = neg_embed
                # If found via pattern, use the actual case from text if possible
                actual_match = re.search(pattern, text, re.IGNORECASE)
                if actual_match:
                    embedding_name = actual_match.group()
                
                embeddings.append({
                    'name': embedding_name,
                    'type': 'negative_embedding'
                })
        
        return embeddings
    
    def _resolve_model_names(self, prompt_data: dict) -> dict:
        """Resolve actual model names by tracing node connections from samplers"""
        model_info = {}
        
        try:
            # Find all sampler nodes (they determine which models are actually used)
            sampler_nodes = []
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue
                class_type = node_data.get('class_type', '')
                if 'sampler' in class_type.lower() or class_type == 'KSampler':
                    sampler_nodes.append((node_id, node_data))
                    print(f"MetaMan Universal: Found sampler node {node_id} ({class_type})")
            
            if not sampler_nodes:
                print(f"MetaMan Universal: No sampler nodes found")
                return model_info
            
            # For each sampler, trace the model connection
            used_models = []
            for node_id, node_data in sampler_nodes:
                inputs = node_data.get('inputs', {})
                
                # Look for model input (usually a node reference)
                model_input = inputs.get('model')
                if model_input and isinstance(model_input, list) and len(model_input) >= 2:
                    source_node_id = str(model_input[0])
                    print(f"MetaMan Universal: Sampler {node_id} uses model from node {source_node_id}")
                    
                    # Find the source node and extract the model name
                    model_name = self._trace_model_source(prompt_data, source_node_id)
                    if model_name:
                        used_models.append(model_name)
                        print(f"MetaMan Universal: Traced to model: {model_name}")
            
            # Remove duplicates and set model info
            unique_models = []
            for model in used_models:
                # Skip None values and node references that weren't resolved
                if model and isinstance(model, str) and not model.startswith('['):
                    if model not in unique_models:
                        unique_models.append(model)
            
            if unique_models:
                if len(unique_models) == 1:
                    model_info['model_name'] = unique_models[0]
                else:
                    model_info['model_name'] = unique_models[0]  # Primary model
                    model_info['additional_models'] = unique_models[1:]  # Additional models
                
                print(f"MetaMan Universal: Final model resolution: {unique_models}")
            else:
                print(f"MetaMan Universal: No resolvable model names found")
            
        except Exception as e:
            print(f"MetaMan Universal: Error resolving model names: {e}")
        
        return model_info
    
    def _trace_model_source(self, prompt_data: dict, node_id: str) -> str:
        """Trace a node to find the actual model name"""
        try:
            if node_id not in prompt_data:
                return None
            
            node_data = prompt_data[node_id]
            if not isinstance(node_data, dict):
                return None
            
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            print(f"MetaMan Universal: Tracing model source in node {node_id} ({class_type})")
            
            # Check different model loader types
            model_fields = [
                'ckpt_name', 'model_name', 'checkpoint_name', 
                'unet_name', 'model', 'checkpoint'
            ]
            
            for field in model_fields:
                if field in inputs and inputs[field]:
                    model_value = inputs[field]
                    # Check if this is a string (actual model name) or a list (node reference)
                    if isinstance(model_value, str):
                        print(f"MetaMan Universal: Found model '{model_value}' in {node_id}.{field}")
                        return model_value
                    elif isinstance(model_value, list) and len(model_value) >= 2:
                        # This is a node reference, trace further
                        source_node_id = str(model_value[0])
                        print(f"MetaMan Universal: Node {node_id}.{field} references node {source_node_id}, tracing further...")
                        return self._trace_model_source(prompt_data, source_node_id)
            
            # If this node doesn't have a direct model name, look for model input that might be a reference to another node
            model_input = inputs.get('model')
            if model_input and isinstance(model_input, list) and len(model_input) >= 2:
                source_node_id = str(model_input[0])
                print(f"MetaMan Universal: Node {node_id} model input references node {source_node_id}, tracing further...")
                return self._trace_model_source(prompt_data, source_node_id)
            
            return None
            
        except Exception as e:
            print(f"MetaMan Universal: Error tracing model source for node {node_id}: {e}")
            return None
    
    def _determine_model_type(self, prompt_data: dict) -> str:
        """Determine model type (SD1.5, SDXL, Flux, etc.) by analyzing VAE and other indicators"""
        try:
            # Look for VAE nodes and their model names
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue
                    
                class_type = node_data.get('class_type', '')
                inputs = node_data.get('inputs', {})
                
                # Check VAE loaders
                if 'vae' in class_type.lower():
                    vae_name = inputs.get('vae_name', '')
                    if vae_name:
                        vae_lower = vae_name.lower()
                        if 'ae.safetensors' in vae_lower or 'flux' in vae_lower:
                            return 'Flux'
                        elif 'sdxl' in vae_lower or 'xl' in vae_lower:
                            return 'SDXL'
                        elif any(sd15_indicator in vae_lower for sd15_indicator in ['sd15', 'v1-5', 'vae-ft']):
                            return 'SD1.5'
                
                # Check for SD3/Flux specific nodes
                if class_type in ['EmptySD3LatentImage', 'SD3LatentImage']:
                    return 'SD3/Flux'
                    
                # Check DualCLIPLoader (Flux indicator)
                if class_type == 'DualCLIPLoader':
                    clip_name1 = inputs.get('clip_name1', '')
                    clip_name2 = inputs.get('clip_name2', '')
                    if 't5xxl' in clip_name2.lower():
                        return 'Flux'
                
                # Check for UNet loaders (often Flux GGUF)
                if 'unet' in class_type.lower():
                    unet_name = inputs.get('unet_name', '')
                    if 'flux' in unet_name.lower():
                        return 'Flux'
            
            # Fallback: look at latent image dimensions
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue
                    
                class_type = node_data.get('class_type', '')
                inputs = node_data.get('inputs', {})
                
                if 'latent' in class_type.lower() and 'empty' in class_type.lower():
                    width = inputs.get('width', 0)
                    height = inputs.get('height', 0)
                    
                    # Flux typically uses 1024x1024 or similar
                    if width >= 1024 or height >= 1024:
                        return 'Flux/SDXL'  # Could be either
                    elif width <= 512 and height <= 512:
                        return 'SD1.5'
            
            return 'Unknown'
            
        except Exception as e:
            print(f"MetaMan Universal: Error determining model type: {e}")
            return 'Unknown'
    
    def _deobfuscate_tensor_ai_models(self, metadata: dict, loras: list) -> dict:
        """De-obfuscate Tensor.AI EMS model codes using generation_data chunk"""
        deobfuscated = {}
        
        try:
            # Look for Tensor.AI generation_data chunk
            generation_data_raw = metadata.get('png_chunk_generation_data') or metadata.get('info_generation_data')
            if not generation_data_raw:
                print(f"MetaMan De-obfuscation: No generation_data found")
                return deobfuscated
            
            print(f"MetaMan De-obfuscation: Found generation_data ({len(generation_data_raw)} chars)")
            
            # Clean the JSON string - remove null bytes and strip whitespace
            cleaned_data = generation_data_raw.replace('\u0000', '').replace('\x00', '').strip()
            print(f"MetaMan De-obfuscation: Cleaned data ({len(cleaned_data)} chars)")
            
            # Parse the generation data JSON
            import json
            generation_data = json.loads(cleaned_data)
            
            # Extract model mappings from generation_data
            models_data = generation_data.get('models', [])
            base_model_data = generation_data.get('baseModel', {})
            
            print(f"MetaMan De-obfuscation: Found {len(models_data)} LoRAs and base model in generation_data")
            
            # Create EMS to real name mapping
            ems_mapping = {}
            
            # Map LoRAs
            for i, model_data in enumerate(models_data):
                if model_data.get('type') == 'LORA':
                    real_name = model_data.get('modelFileName', '')
                    model_hash = model_data.get('hash', '')
                    weight = model_data.get('weight', 1.0)
                    label = model_data.get('label', '')
                    
                    # Store mapping info
                    ems_mapping[real_name] = {
                        'hash': model_hash,
                        'weight': weight,
                        'label': label,
                        'type': 'lora',
                        'index': i
                    }
                    
                    print(f"MetaMan De-obfuscation: LoRA {i} mapping - {real_name} (hash: {model_hash[:16]}...)")
            
            # Map base model
            if base_model_data:
                real_name = base_model_data.get('modelFileName', '')
                model_hash = base_model_data.get('hash', '')
                label = base_model_data.get('label', '')
                
                if real_name:
                    ems_mapping[real_name] = {
                        'hash': model_hash,
                        'label': label,
                        'type': 'checkpoint'
                    }
                    
                    # Set the real checkpoint name
                    deobfuscated['model_name_real'] = real_name
                    deobfuscated['model_hash'] = model_hash
                    deobfuscated['model_label'] = label
                    
                    print(f"MetaMan De-obfuscation: Checkpoint mapping - {real_name} (hash: {model_hash[:16]}...)")
            
            # Apply de-obfuscation to LoRAs
            if loras and models_data:
                deobfuscated_loras = []
                
                print(f"MetaMan De-obfuscation: Mapping {len(loras)} LoRAs with {len(models_data)} model data entries")
                
                # Map LoRAs by order/index since EMS names don't match real names
                for i, lora in enumerate(loras):
                    lora_name = lora.get('name', '') if isinstance(lora, dict) else str(lora)
                    lora_weight = lora.get('weight', 1.0) if isinstance(lora, dict) else 1.0
                    
                    # Find corresponding model data by index
                    if i < len(models_data):
                        model_data = models_data[i]
                        if model_data.get('type') == 'LORA':
                            real_name = model_data.get('modelFileName', '')
                            enhanced_lora = {
                                'name': lora_name,  # Keep original EMS name
                                'real_name': real_name,  # Add real name
                                'weight': lora_weight,  # Use weight from extraction
                                'hash': model_data.get('hash', ''),
                                'label': model_data.get('label', '')
                            }
                            deobfuscated_loras.append(enhanced_lora)
                            print(f"MetaMan De-obfuscation: Enhanced LoRA {i} - {lora_name} → {real_name}")
                        else:
                            deobfuscated_loras.append(lora)
                            print(f"MetaMan De-obfuscation: Model data {i} not LoRA type, keeping original: {lora_name}")
                    else:
                        # Keep original if no mapping found
                        deobfuscated_loras.append(lora)
                        print(f"MetaMan De-obfuscation: No model data for LoRA {i}, keeping original: {lora_name}")
                
                if deobfuscated_loras:
                    deobfuscated['loras_enhanced'] = deobfuscated_loras
                    print(f"MetaMan De-obfuscation: Created {len(deobfuscated_loras)} enhanced LoRAs")
            
            # Store the EMS mapping for future reference
            if ems_mapping:
                deobfuscated['tensor_ai_mappings'] = ems_mapping
                print(f"MetaMan De-obfuscation: Stored {len(ems_mapping)} mappings")
                
        except json.JSONDecodeError as e:
            print(f"MetaMan De-obfuscation JSON Error: {e}")
            print(f"MetaMan De-obfuscation: Problematic data around char {e.pos}: '{generation_data_raw[max(0, e.pos-50):e.pos+50]}'")
        except Exception as e:
            print(f"MetaMan De-obfuscation Error: {e}")
        
        return deobfuscated
    
    def _extract_vae_info(self, prompt_data: dict) -> dict:
        """Extract VAE information from workflow"""
        vae_info = {}
        
        try:
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue
                    
                class_type = node_data.get('class_type', '')
                inputs = node_data.get('inputs', {})
                
                # Check VAE loaders
                if 'vae' in class_type.lower() and 'loader' in class_type.lower():
                    vae_name = inputs.get('vae_name', '')
                    if vae_name:
                        vae_info['vae_name'] = vae_name
                        print(f"MetaMan Universal: Found VAE: {vae_name}")
                        break
            
        except Exception as e:
            print(f"MetaMan Universal: Error extracting VAE info: {e}")
        
        return vae_info


class MetaManUniversalNodeV2:
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
                "metadata_json": ("STRING", {"default": "", "multiline": True}),
                "target_service": (cls.SUPPORTED_SERVICES, {"default": "automatic1111"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_workflow", "json_workflow")
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

    def process_metadata(self, image, metadata_json, target_service):
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
            
            # Check if metadata was provided from MetaManLoadImage node
            source_metadata = {}
            if metadata_json and metadata_json.strip():
                try:
                    metadata_data = json.loads(metadata_json)
                    source_metadata = metadata_data.get("metadata", {})
                    print(f"MetaMan Debug: Using metadata from MetaManLoadImage: {list(source_metadata.keys())}")
                    print(f"MetaMan Debug: Source file: {metadata_data.get('source_file_path', 'unknown')}")
                except Exception as e:
                    print(f"MetaMan Debug: Error parsing metadata_json: {e}")
                    return (image, json.dumps({"error": f"Invalid metadata_json format: {e}"}, indent=2))
            else:
                print(f"MetaMan Debug: No metadata_json provided - use MetaManLoadImage node for best results")
            
            # If no metadata from load image node, try extracting from PIL image
            if not source_metadata:
                print(f"MetaMan Debug: Falling back to PIL image extraction...")
                source_metadata = self._extract_source_metadata(pil_image)
                
                # If still no metadata found, try to find and read the original file
                if not source_metadata:
                    print(f"MetaMan Debug: No metadata in PIL image, attempting direct file reading...")
                    original_file_metadata = self._try_read_original_file(pil_image)
                    if original_file_metadata:
                        source_metadata = original_file_metadata
                        print(f"MetaMan Debug: Successfully read metadata from original file")
                    else:
                        print(f"MetaMan Debug: No metadata found anywhere - recommend using MetaManLoadImage node")
            
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
        
        print(f"MetaMan Debug: Starting metadata extraction...")
        print(f"MetaMan Debug: Image type: {type(image)}")
        print(f"MetaMan Debug: Image mode: {getattr(image, 'mode', 'N/A')}")
        print(f"MetaMan Debug: Image format: {getattr(image, 'format', 'N/A')}")
        print(f"MetaMan Debug: Image size: {getattr(image, 'size', 'N/A')}")
        
        # Check if image has filename attribute (from file loading)
        if hasattr(image, 'filename'):
            print(f"MetaMan Debug: Image filename: {image.filename}")
        
        # Check PNG text chunks
        if hasattr(image, 'text') and image.text:
            print(f"MetaMan Debug: Found PNG text chunks: {list(image.text.keys())}")
            for key, value in image.text.items():
                print(f"MetaMan Debug: Chunk '{key}': {len(value)} characters")
            
            # A1111/Civitai parameters format
            if 'parameters' in image.text:
                print(f"MetaMan Debug: Found 'parameters' chunk")
                metadata.update(self._parse_a1111_parameters(image.text['parameters']))
            
            # ComfyUI workflow format  
            if 'workflow' in image.text:
                print(f"MetaMan Debug: Found 'workflow' chunk")
                try:
                    workflow_data = json.loads(image.text['workflow'])
                    metadata['comfyui_workflow'] = workflow_data
                    print(f"MetaMan Debug: Successfully parsed workflow with {len(workflow_data.get('nodes', []))} nodes")
                except Exception as e:
                    print(f"MetaMan Debug: Error parsing workflow: {e}")
            
            # ComfyUI prompt format
            if 'prompt' in image.text:
                print(f"MetaMan Debug: Found 'prompt' chunk")
                try:
                    prompt_data = json.loads(image.text['prompt'])
                    metadata['comfyui_prompt'] = prompt_data
                    print(f"MetaMan Debug: Successfully parsed prompt with {len(prompt_data)} nodes")
                    
                    # Extract basic parameters from prompt data immediately
                    extracted_params = self._extract_params_from_comfyui_prompt(prompt_data)
                    metadata.update(extracted_params)
                    print(f"MetaMan Debug: Extracted params from prompt: {list(extracted_params.keys())}")
                except Exception as e:
                    print(f"MetaMan Debug: Error parsing prompt: {e}")
            
            # Custom "meta" chunk (our universal format)
            if 'meta' in image.text:
                print(f"MetaMan Debug: Found 'meta' chunk")
                try:
                    meta_data = json.loads(image.text['meta'])
                    metadata.update(meta_data)
                except Exception as e:
                    print(f"MetaMan Debug: Error parsing meta chunk: {e}")
        else:
            print(f"MetaMan Debug: No PNG text chunks found or image.text is empty")
            print(f"MetaMan Debug: image.text value: {getattr(image, 'text', 'ATTRIBUTE_MISSING')}")
            
            # Try alternative PNG info access
            if hasattr(image, 'info'):
                print(f"MetaMan Debug: Image.info keys: {list(image.info.keys())}")
                for key, value in image.info.items():
                    if isinstance(value, str) and len(value) > 50:
                        print(f"MetaMan Debug: Info '{key}': {len(value)} characters")
                        # Try to parse as potential metadata
                        if key.lower() in ['parameters', 'workflow', 'prompt', 'meta']:
                            print(f"MetaMan Debug: Found potential metadata in info['{key}']")
                            if key == 'parameters':
                                metadata.update(self._parse_a1111_parameters(value))
                            elif key in ['workflow', 'prompt']:
                                try:
                                    data = json.loads(value)
                                    metadata[f'comfyui_{key}'] = data
                                    if key == 'prompt':
                                        extracted_params = self._extract_params_from_comfyui_prompt(data)
                                        metadata.update(extracted_params)
                                except:
                                    pass
        
        # Check EXIF data
        if hasattr(image, '_getexif') and image._getexif():
            print(f"MetaMan Debug: Found EXIF data")
            exif_data = image._getexif()
            metadata.update(self._parse_exif_metadata(exif_data))
        else:
            print(f"MetaMan Debug: No EXIF data found")
        
        print(f"MetaMan Debug: Final extracted metadata keys: {list(metadata.keys())}")
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
        print(f"MetaMan Debug: Detecting service for metadata keys: {list(metadata.keys())}")
        
        # ComfyUI indicators (check multiple possible indicators)
        comfyui_indicators = [
            'comfyui_workflow' in metadata,
            'comfyui_prompt' in metadata,
            'workflow' in metadata,  # Raw workflow chunk
            'prompt' in metadata and isinstance(metadata.get('prompt'), dict),  # ComfyUI prompt dict
            any(key.startswith('comfyui_') for key in metadata.keys()),
            # Check if we have typical ComfyUI node data
            any(isinstance(value, dict) and 'class_type' in str(value) for value in metadata.values())
        ]
        
        if any(comfyui_indicators):
            print(f"MetaMan Debug: Detected ComfyUI service (indicators: {comfyui_indicators})")
            return 'comfyui'
        
        # A1111/Civitai indicators
        elif 'model_hash' in metadata and 'sampler' in metadata:
            print(f"MetaMan Debug: Detected A1111/Civitai service")
            return 'automatic1111'
        
        # Other services
        elif 'tensor_ai_style' in metadata:
            print(f"MetaMan Debug: Detected Tensor.AI service")
            return 'tensor.ai'
        elif 'leonardo_preset' in metadata:
            print(f"MetaMan Debug: Detected Leonardo.AI service")
            return 'leonardo.ai'
        elif 'midjourney_version' in metadata:
            print(f"MetaMan Debug: Detected Midjourney service")
            return 'midjourney'
        
        # If we have basic generation params, try to infer
        elif 'prompt' in metadata and isinstance(metadata['prompt'], str):
            # Check for A1111-style parameters
            if any(key in metadata for key in ['steps', 'cfg_scale', 'seed']):
                print(f"MetaMan Debug: Detected generic A1111-style metadata")
                return 'automatic1111'
        
        print(f"MetaMan Debug: Could not detect service, using generic")
        return 'generic'
    
    def _convert_to_universal(self, source_metadata: dict, source_service: str) -> dict:
        """Convert source metadata to universal format"""
        print(f"MetaMan Debug: Converting to universal format from {source_service}")
        print(f"MetaMan Debug: Source metadata keys: {list(source_metadata.keys())}")
        
        universal = {
            "schema_version": self.universal_schema["schema_version"],
            "source_service": source_service,
            "creation_time": datetime.now().isoformat(),
            "metaman_version": "1.0.0",
            "metadata": {}
        }
        
        # First, copy all source metadata directly
        for key, value in source_metadata.items():
            universal["metadata"][key] = value
            print(f"MetaMan Debug: Copied direct field {key}: {type(value)}")
        
        # Then map source fields to universal schema (for compatibility)
        mapped_count = 0
        for field_name, field_config in self.universal_schema["fields"].items():
            if source_service in field_config.get("supported_by", []):
                # Direct mapping
                if field_name in source_metadata:
                    if field_name not in universal["metadata"]:
                        universal["metadata"][field_name] = source_metadata[field_name]
                        mapped_count += 1
                        print(f"MetaMan Debug: Mapped schema field {field_name}")
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
                                    mapped_count += 1
                                    print(f"MetaMan Debug: Mapped via schema {field_name}: {source_val} -> {universal_val}")
                                    break
                            else:
                                universal["metadata"][field_name] = source_value
                                mapped_count += 1
                                print(f"MetaMan Debug: Mapped direct {field_name}: {source_value}")
                        else:
                            universal["metadata"][field_name] = source_value
                            mapped_count += 1
                            print(f"MetaMan Debug: Mapped simple {field_name}: {source_value}")
        
        print(f"MetaMan Debug: Universal conversion complete. {len(universal['metadata'])} total fields, {mapped_count} schema-mapped")
        print(f"MetaMan Debug: Final universal metadata keys: {list(universal['metadata'].keys())}")
        
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
        print(f"MetaMan Debug: Formatting A1111 output from metadata: {list(metadata.keys())}")
        lines = []
        
        # Positive prompt
        if 'prompt' in metadata:
            lines.append(metadata['prompt'])
            print(f"MetaMan Debug: Added prompt: {metadata['prompt'][:100]}...")
        else:
            print(f"MetaMan Debug: No prompt found in metadata")
        
        # Negative prompt
        if 'negative_prompt' in metadata:
            lines.append(f"Negative prompt: {metadata['negative_prompt']}")
            print(f"MetaMan Debug: Added negative prompt: {metadata['negative_prompt'][:100]}...")
        
        # Parameters line
        params = []
        param_order = template.get("parameter_order", [
            "steps", "sampler", "cfg_scale", "seed", "width", "height", 
            "model_hash", "clip_skip", "denoising_strength"
        ])
        
        print(f"MetaMan Debug: Processing parameters in order: {param_order}")
        
        for param in param_order:
            if param in metadata and metadata[param] is not None:
                print(f"MetaMan Debug: Found parameter {param}: {metadata[param]}")
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
                    print(f"MetaMan Debug: Added size: {metadata['width']}x{metadata['height']}")
                elif param != "height":  # Skip height since it's handled with width
                    params.append(f"{param.title()}: {metadata[param]}")
            else:
                print(f"MetaMan Debug: Parameter {param} not found or is None")
        
        if params:
            param_line = ", ".join(params)
            lines.append(param_line)
            print(f"MetaMan Debug: Added parameters line: {param_line}")
        else:
            print(f"MetaMan Debug: No parameters to add")
        
        result = "\n".join(lines)
        print(f"MetaMan Debug: Final A1111 output ({len(result)} chars): {result[:200]}...")
        return result
    
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
    
    def _extract_params_from_comfyui_prompt(self, prompt_data: dict) -> dict:
        """Extract generation parameters from ComfyUI prompt data"""
        params = {}
        
        print(f"MetaMan Debug: Extracting params from ComfyUI prompt with {len(prompt_data)} nodes")
        
        try:
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue
                    
                class_type = node_data.get('class_type', '')
                inputs = node_data.get('inputs', {})
                
                print(f"MetaMan Debug: Processing node {node_id} of type {class_type}")
                
                # Extract from KSampler nodes
                if class_type == 'KSampler':
                    if 'steps' in inputs:
                        params['steps'] = inputs['steps']
                        print(f"MetaMan Debug: Found steps: {inputs['steps']}")
                    if 'cfg' in inputs:
                        params['cfg_scale'] = inputs['cfg']
                        print(f"MetaMan Debug: Found CFG: {inputs['cfg']}")
                    if 'sampler_name' in inputs:
                        params['sampler'] = inputs['sampler_name']
                        print(f"MetaMan Debug: Found sampler: {inputs['sampler_name']}")
                    if 'scheduler' in inputs:
                        params['scheduler'] = inputs['scheduler']
                        print(f"MetaMan Debug: Found scheduler: {inputs['scheduler']}")
                    if 'seed' in inputs:
                        params['seed'] = inputs['seed']
                        print(f"MetaMan Debug: Found seed: {inputs['seed']}")
                    if 'denoise' in inputs:
                        params['denoising_strength'] = inputs['denoise']
                        print(f"MetaMan Debug: Found denoise: {inputs['denoise']}")
                
                # Extract from CheckpointLoaderSimple
                elif class_type == 'CheckpointLoaderSimple':
                    if 'ckpt_name' in inputs:
                        params['model_name'] = inputs['ckpt_name']
                        print(f"MetaMan Debug: Found model: {inputs['ckpt_name']}")
                
                # Extract from CLIPTextEncode (prompts)
                elif class_type == 'CLIPTextEncode':
                    if 'text' in inputs:
                        # First positive prompt we find
                        if 'prompt' not in params and inputs['text'].strip():
                            params['prompt'] = inputs['text']
                            print(f"MetaMan Debug: Found prompt: {inputs['text'][:100]}...")
                        # If we already have a prompt, this might be negative
                        elif 'negative_prompt' not in params and inputs['text'].strip():
                            # Simple heuristic: if it contains common negative words
                            negative_indicators = ['worst', 'low quality', 'blurry', 'bad', 'ugly', 'deformed']
                            if any(indicator in inputs['text'].lower() for indicator in negative_indicators):
                                params['negative_prompt'] = inputs['text']
                                print(f"MetaMan Debug: Found negative prompt: {inputs['text'][:100]}...")
                
                # Extract from EmptyLatentImage (dimensions)
                elif class_type == 'EmptyLatentImage':
                    if 'width' in inputs:
                        params['width'] = inputs['width']
                        print(f"MetaMan Debug: Found width: {inputs['width']}")
                    if 'height' in inputs:
                        params['height'] = inputs['height']
                        print(f"MetaMan Debug: Found height: {inputs['height']}")
                
                # Extract from LoraLoader nodes
                elif class_type == 'LoraLoader':
                    if 'lora_name' in inputs and 'strength_model' in inputs:
                        if 'loras' not in params:
                            params['loras'] = []
                        lora_info = {
                            'name': inputs['lora_name'],
                            'weight': inputs['strength_model']
                        }
                        if 'strength_clip' in inputs:
                            lora_info['clip_weight'] = inputs['strength_clip']
                        params['loras'].append(lora_info)
                        print(f"MetaMan Debug: Found LoRA: {inputs['lora_name']} @ {inputs['strength_model']}")
        
        except Exception as e:
            print(f"MetaMan Debug: Error extracting ComfyUI params: {e}")
        
        print(f"MetaMan Debug: Final extracted params: {list(params.keys())}")
        return params
    
    def _try_read_original_file(self, pil_image) -> dict:
        """Attempt to find and read the original PNG file with metadata"""
        metadata = {}
        
        try:
            import glob
            
            # Common ComfyUI directories to search
            search_paths = [
                "/Users/pxl8d/Art/ComfyUI/input/*.png",
                "/Users/pxl8d/Art/ComfyUI/output/*.png", 
                "/Users/pxl8d/Art/ComfyUI/temp/*.png"
            ]
            
            # Get image dimensions for matching
            img_size = pil_image.size if hasattr(pil_image, 'size') else None
            print(f"MetaMan Debug: Looking for original file with size {img_size}")
            
            # Search for PNG files in common locations
            candidates = []
            for search_path in search_paths:
                try:
                    files = glob.glob(search_path)
                    candidates.extend(files)
                    print(f"MetaMan Debug: Found {len(files)} PNG files in {search_path}")
                except Exception as e:
                    print(f"MetaMan Debug: Error searching {search_path}: {e}")
            
            # Sort by modification time (newest first)
            candidates.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            print(f"MetaMan Debug: Checking {len(candidates)} candidate files...")
            
            # Try to find a matching file
            for file_path in candidates[:10]:  # Check up to 10 most recent files
                try:
                    print(f"MetaMan Debug: Checking file: {file_path}")
                    
                    # Open file directly with PIL to preserve metadata
                    with Image.open(file_path) as test_image:
                        # Check if dimensions match
                        if img_size and test_image.size != img_size:
                            print(f"MetaMan Debug: Size mismatch: {test_image.size} vs {img_size}")
                            continue
                        
                        print(f"MetaMan Debug: Size match found: {test_image.size}")
                        
                        # Check for PNG text chunks
                        if hasattr(test_image, 'text') and test_image.text:
                            print(f"MetaMan Debug: File has PNG text chunks: {list(test_image.text.keys())}")
                            
                            # Extract metadata from this file
                            file_metadata = self._extract_metadata_from_file_image(test_image)
                            if file_metadata:
                                print(f"MetaMan Debug: Successfully extracted metadata from {file_path}")
                                return file_metadata
                        else:
                            print(f"MetaMan Debug: File has no PNG text chunks")
                            
                except Exception as e:
                    print(f"MetaMan Debug: Error reading {file_path}: {e}")
                    continue
            
            print(f"MetaMan Debug: No matching file with metadata found")
            
        except Exception as e:
            print(f"MetaMan Debug: Error in file search: {e}")
        
        return metadata
    
    def _extract_metadata_from_file_image(self, file_image) -> dict:
        """Extract metadata from a file-loaded PIL image (preserves PNG chunks)"""
        metadata = {}
        
        try:
            # Check PNG text chunks
            if hasattr(file_image, 'text') and file_image.text:
                print(f"MetaMan Debug: File image has text chunks: {list(file_image.text.keys())}")
                
                # A1111/Civitai parameters format
                if 'parameters' in file_image.text:
                    print(f"MetaMan Debug: Found 'parameters' chunk in file")
                    metadata.update(self._parse_a1111_parameters(file_image.text['parameters']))
                
                # ComfyUI workflow format  
                if 'workflow' in file_image.text:
                    print(f"MetaMan Debug: Found 'workflow' chunk in file")
                    try:
                        workflow_data = json.loads(file_image.text['workflow'])
                        metadata['comfyui_workflow'] = workflow_data
                        print(f"MetaMan Debug: Successfully parsed workflow with {len(workflow_data.get('nodes', []))} nodes")
                    except Exception as e:
                        print(f"MetaMan Debug: Error parsing workflow from file: {e}")
                
                # ComfyUI prompt format
                if 'prompt' in file_image.text:
                    print(f"MetaMan Debug: Found 'prompt' chunk in file")
                    try:
                        prompt_data = json.loads(file_image.text['prompt'])
                        metadata['comfyui_prompt'] = prompt_data
                        print(f"MetaMan Debug: Successfully parsed prompt with {len(prompt_data)} nodes")
                        
                        # Extract basic parameters from prompt data
                        extracted_params = self._extract_params_from_comfyui_prompt(prompt_data)
                        metadata.update(extracted_params)
                        print(f"MetaMan Debug: Extracted params from file prompt: {list(extracted_params.keys())}")
                    except Exception as e:
                        print(f"MetaMan Debug: Error parsing prompt from file: {e}")
                
                # Custom "meta" chunk
                if 'meta' in file_image.text:
                    print(f"MetaMan Debug: Found 'meta' chunk in file")
                    try:
                        meta_data = json.loads(file_image.text['meta'])
                        metadata.update(meta_data)
                    except Exception as e:
                        print(f"MetaMan Debug: Error parsing meta chunk from file: {e}")
        
        except Exception as e:
            print(f"MetaMan Debug: Error extracting metadata from file image: {e}")
        
        print(f"MetaMan Debug: File metadata extraction complete. Keys: {list(metadata.keys())}")
        return metadata


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MetaManLoadAndConvert": MetaManLoadAndConvert,
    "MetaManExtractComponents": MetaManExtractComponents,
    "MetaManEmbedAndSave": MetaManEmbedAndSave,
    "MetaManUniversalNodeV2": MetaManUniversalNodeV2,
    "MetaManLoadImage": MetaManLoadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaManLoadAndConvert": "MetaMan Load & Convert",
    "MetaManExtractComponents": "MetaMan Extract Components",
    "MetaManEmbedAndSave": "MetaMan Embed & Save",
    "MetaManUniversalNodeV2": "MetaMan Universal V2",
    "MetaManLoadImage": "MetaMan Load Image"
}
