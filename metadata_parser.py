"""
MetaMan Metadata Parser Module
Handles parsing of metadata from various AI image generation formats
"""

import json
import re
import base64
from typing import Dict, List, Optional, Any, Union
from PIL import Image, PngImagePlugin
import piexif
from datetime import datetime


class MetadataParser:
    """
    Universal metadata parser for AI image generation platforms
    """
    
    def __init__(self):
        self.parsers = {
            'automatic1111': self._parse_a1111_parameters,
            'comfyui': self._parse_comfyui_metadata,
            'civitai': self._parse_civitai_metadata,
            'forge': self._parse_forge_parameters,
            'tensor.ai': self._parse_tensor_ai_metadata,
            'leonardo.ai': self._parse_leonardo_metadata,
            'generic': self._parse_generic_metadata
        }
    
    def extract_all_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract all available metadata from an image
        Returns unified metadata structure
        """
        metadata = {
            'source_detected': 'unknown',
            'raw_metadata': {},
            'parsed_metadata': {},
            'chunks_found': [],
            'confidence': 0.0
        }
        
        # Extract PNG text chunks
        if hasattr(image, 'text') and image.text:
            metadata['raw_metadata'].update(image.text)
            metadata['chunks_found'] = list(image.text.keys())
        
        # Extract EXIF data
        exif_data = self._extract_exif_data(image)
        if exif_data:
            metadata['raw_metadata']['exif'] = exif_data
        
        # Detect source service and parse accordingly
        detected_source, confidence = self._detect_source_service(metadata['raw_metadata'])
        metadata['source_detected'] = detected_source
        metadata['confidence'] = confidence
        
        # Parse using appropriate parser
        if detected_source in self.parsers:
            parsed = self.parsers[detected_source](metadata['raw_metadata'])
            metadata['parsed_metadata'] = parsed
        
        return metadata
    
    def _detect_source_service(self, raw_metadata: Dict) -> tuple[str, float]:
        """
        Detect which service generated the image based on metadata patterns
        Returns (service_name, confidence_score)
        """
        confidence_scores = {}
        
        # ComfyUI detection
        if 'workflow' in raw_metadata:
            confidence_scores['comfyui'] = 0.95
        elif 'prompt' in raw_metadata and self._is_json_like(raw_metadata.get('prompt', '')):
            confidence_scores['comfyui'] = 0.7
        
        # A1111/Civitai detection
        if 'parameters' in raw_metadata:
            params_text = raw_metadata['parameters']
            if self._is_a1111_format(params_text):
                # Check for Civitai-specific indicators
                if any(indicator in params_text.lower() for indicator in ['civitai', 'model id:', 'version id:']):
                    confidence_scores['civitai'] = 0.9
                else:
                    confidence_scores['automatic1111'] = 0.85
        
        # Forge detection (similar to A1111 but with extensions)
        if 'parameters' in raw_metadata:
            params_text = raw_metadata['parameters']
            if any(indicator in params_text.lower() for indicator in ['schedule type:', 'sgm noise multiplier:', 'forge']):
                confidence_scores['forge'] = 0.9
        
        # Tensor.AI detection
        if 'TensorAI' in raw_metadata or any('tensor' in key.lower() for key in raw_metadata.keys()):
            confidence_scores['tensor.ai'] = 0.9
        
        # Leonardo.AI detection
        if 'LeonardoAI' in raw_metadata or any('leonardo' in key.lower() for key in raw_metadata.keys()):
            confidence_scores['leonardo.ai'] = 0.9
        
        # Custom MetaMan format
        if 'meta' in raw_metadata:
            try:
                meta_data = json.loads(raw_metadata['meta'])
                if 'source_service' in meta_data:
                    source = meta_data['source_service']
                    confidence_scores[source] = 0.95
            except:
                pass
        
        # Return highest confidence detection
        if confidence_scores:
            best_match = max(confidence_scores.items(), key=lambda x: x[1])
            return best_match[0], best_match[1]
        else:
            return 'generic', 0.1
    
    def _is_a1111_format(self, text: str) -> bool:
        """Check if text matches A1111 parameter format"""
        if not text:
            return False
        
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Last line should contain parameters with colons
        last_line = lines[-1]
        return ':' in last_line and any(param in last_line.lower() for param in ['steps', 'cfg scale', 'seed', 'sampler'])
    
    def _is_json_like(self, text: str) -> bool:
        """Check if text looks like JSON"""
        try:
            json.loads(text)
            return True
        except:
            return False
    
    def _extract_exif_data(self, image: Image.Image) -> Optional[Dict]:
        """Extract EXIF data from image"""
        try:
            exif_dict = piexif.load(image.info.get('exif', b''))
            return {
                'GPS': exif_dict.get('GPS', {}),
                'Exif': exif_dict.get('Exif', {}),
                'Image': exif_dict.get('0th', {}),
                'Thumbnail': exif_dict.get('thumbnail'),
                'Interop': exif_dict.get('Interop', {})
            }
        except:
            return None
    
    def _parse_a1111_parameters(self, raw_metadata: Dict) -> Dict:
        """Parse Automatic1111 parameter format"""
        if 'parameters' not in raw_metadata:
            return {}
        
        params_text = raw_metadata['parameters']
        return self._parse_a1111_text(params_text)
    
    def _parse_a1111_text(self, params_text: str) -> Dict:
        """Parse A1111 parameter text format"""
        metadata = {}
        lines = params_text.strip().split('\n')
        
        if not lines:
            return metadata
        
        # First line is typically the positive prompt
        metadata['prompt'] = lines[0].strip()
        
        # Process subsequent lines
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line:
                continue
            
            # Check for negative prompt
            if line.startswith('Negative prompt:'):
                metadata['negative_prompt'] = line.replace('Negative prompt:', '').strip()
                continue
            
            # Check if this is a parameters line (contains key: value pairs)
            if ':' in line and ',' in line:
                params = self._parse_parameter_line(line)
                metadata.update(params)
        
        return metadata
    
    def _parse_parameter_line(self, line: str) -> Dict:
        """Parse A1111 parameter line with key: value pairs"""
        params = {}
        
        # Split by comma, respecting quoted values
        parts = self._smart_split(line, ',')
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
            
            key, value = part.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            # Convert to appropriate type
            params[key] = self._convert_value_type(key, value)
        
        # Handle special cases
        if 'size' in params:
            size_parts = params['size'].split('x')
            if len(size_parts) == 2:
                try:
                    params['width'] = int(size_parts[0])
                    params['height'] = int(size_parts[1])
                except:
                    pass
        
        return params
    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter, respecting quoted sections"""
        parts = []
        current = ""
        in_quotes = False
        i = 0
        
        while i < len(text):
            char = text[i]
            
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_quotes = not in_quotes
                current += char
            elif char == delimiter and not in_quotes:
                parts.append(current)
                current = ""
            else:
                current += char
            
            i += 1
        
        if current:
            parts.append(current)
        
        return [part.strip() for part in parts if part.strip()]
    
    def _convert_value_type(self, key: str, value: str) -> Union[str, int, float, bool]:
        """Convert string value to appropriate Python type"""
        # Boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Integer fields
        if key in ['steps', 'width', 'height', 'seed', 'clip_skip', 'batch_size', 'batch_pos']:
            try:
                return int(value)
            except:
                return value
        
        # Float fields
        if key in ['cfg_scale', 'denoising_strength', 'eta', 'subseed_strength']:
            try:
                return float(value)
            except:
                return value
        
        # Return as string by default
        return value
    
    def _parse_comfyui_metadata(self, raw_metadata: Dict) -> Dict:
        """Parse ComfyUI metadata format"""
        metadata = {}
        
        # Parse workflow
        if 'workflow' in raw_metadata:
            try:
                workflow = json.loads(raw_metadata['workflow'])
                metadata['comfyui_workflow'] = workflow
                
                # Extract basic parameters from workflow
                extracted = self._extract_params_from_comfyui_workflow(workflow)
                metadata.update(extracted)
            except:
                pass
        
        # Parse prompt data
        if 'prompt' in raw_metadata:
            try:
                prompt_data = json.loads(raw_metadata['prompt'])
                metadata['comfyui_prompt'] = prompt_data
                
                # Extract parameters from prompt data
                extracted = self._extract_params_from_comfyui_prompt(prompt_data)
                metadata.update(extracted)
            except:
                pass
        
        # Fall back to A1111 format if present
        if 'parameters' in raw_metadata and not metadata:
            metadata.update(self._parse_a1111_text(raw_metadata['parameters']))
        
        return metadata
    
    def _extract_params_from_comfyui_workflow(self, workflow: Dict) -> Dict:
        """Extract generation parameters from ComfyUI workflow"""
        params = {}
        
        nodes = workflow.get('nodes', [])
        
        for node in nodes:
            node_type = node.get('type', '')
            widgets_values = node.get('widgets_values', [])
            
            # Extract from KSampler nodes
            if node_type in ['KSampler', 'KSamplerAdvanced']:
                if len(widgets_values) >= 4:
                    params['seed'] = widgets_values[0] if isinstance(widgets_values[0], int) else None
                    params['steps'] = widgets_values[1] if isinstance(widgets_values[1], int) else None
                    params['cfg_scale'] = widgets_values[2] if isinstance(widgets_values[2], (int, float)) else None
                    params['sampler'] = widgets_values[3] if isinstance(widgets_values[3], str) else None
                    if len(widgets_values) > 4:
                        params['scheduler'] = widgets_values[4] if isinstance(widgets_values[4], str) else None
            
            # Extract from text nodes
            elif node_type == 'CLIPTextEncode' and widgets_values:
                text = widgets_values[0] if isinstance(widgets_values[0], str) else ''
                # Determine if this is positive or negative prompt based on context
                # This is a simplified approach - real implementation would need node connection analysis
                if not params.get('prompt'):
                    params['prompt'] = text
                elif text != params.get('prompt', ''):
                    params['negative_prompt'] = text
            
            # Extract from checkpoint loader
            elif node_type == 'CheckpointLoaderSimple' and widgets_values:
                params['model_name'] = widgets_values[0] if isinstance(widgets_values[0], str) else None
            
            # Extract from image size nodes
            elif node_type == 'EmptyLatentImage' and len(widgets_values) >= 2:
                params['width'] = widgets_values[0] if isinstance(widgets_values[0], int) else None
                params['height'] = widgets_values[1] if isinstance(widgets_values[1], int) else None
        
        return params
    
    def _extract_params_from_comfyui_prompt(self, prompt_data: Dict) -> Dict:
        """Extract parameters from ComfyUI prompt data"""
        params = {}
        
        # ComfyUI prompt data contains node execution info
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            # Extract from samplers
            if class_type in ['KSampler', 'KSamplerAdvanced']:
                params.update({
                    'seed': inputs.get('seed'),
                    'steps': inputs.get('steps'),
                    'cfg_scale': inputs.get('cfg'),
                    'sampler': inputs.get('sampler_name'),
                    'scheduler': inputs.get('scheduler'),
                    'denoising_strength': inputs.get('denoise', 1.0)
                })
            
            # Extract text
            elif class_type == 'CLIPTextEncode':
                text = inputs.get('text', '')
                if not params.get('prompt'):
                    params['prompt'] = text
                elif text != params.get('prompt', ''):
                    params['negative_prompt'] = text
            
            # Extract model
            elif class_type == 'CheckpointLoaderSimple':
                params['model_name'] = inputs.get('ckpt_name')
            
            # Extract dimensions
            elif class_type == 'EmptyLatentImage':
                params.update({
                    'width': inputs.get('width'),
                    'height': inputs.get('height'),
                    'batch_size': inputs.get('batch_size', 1)
                })
        
        # Clean up None values
        return {k: v for k, v in params.items() if v is not None}
    
    def _parse_civitai_metadata(self, raw_metadata: Dict) -> Dict:
        """Parse Civitai metadata (extends A1111 format)"""
        metadata = self._parse_a1111_parameters(raw_metadata)
        
        # Parse Civitai-specific chunk if present
        if 'civt' in raw_metadata:
            try:
                civitai_data = json.loads(raw_metadata['civt'])
                if 'data' in civitai_data:
                    metadata.update(civitai_data['data'])
            except:
                pass
        
        return metadata
    
    def _parse_forge_parameters(self, raw_metadata: Dict) -> Dict:
        """Parse Forge metadata (extends A1111 format)"""
        metadata = self._parse_a1111_parameters(raw_metadata)
        
        # Parse Forge-specific chunk if present
        if 'forge' in raw_metadata:
            try:
                forge_data = json.loads(raw_metadata['forge'])
                metadata.update(forge_data)
            except:
                pass
        
        return metadata
    
    def _parse_tensor_ai_metadata(self, raw_metadata: Dict) -> Dict:
        """Parse Tensor.AI metadata format"""
        metadata = {}
        
        if 'TensorAI' in raw_metadata:
            try:
                tensor_data = json.loads(raw_metadata['TensorAI'])
                
                # Map Tensor.AI format to universal format
                generation_params = tensor_data.get('generation_params', {})
                tensor_params = tensor_data.get('tensor_ai_params', {})
                
                metadata.update({
                    'prompt': generation_params.get('prompt'),
                    'negative_prompt': generation_params.get('negative_prompt'),
                    'steps': generation_params.get('num_inference_steps'),
                    'cfg_scale': generation_params.get('guidance_scale'),
                    'width': generation_params.get('width'),
                    'height': generation_params.get('height'),
                    'seed': generation_params.get('seed'),
                    'sampler': generation_params.get('scheduler'),
                    'tensor_ai_style': tensor_params.get('style'),
                    'model_name': tensor_params.get('model_id')
                })
            except:
                pass
        
        return metadata
    
    def _parse_leonardo_metadata(self, raw_metadata: Dict) -> Dict:
        """Parse Leonardo.AI metadata format"""
        metadata = {}
        
        if 'LeonardoAI' in raw_metadata:
            try:
                leonardo_data = json.loads(raw_metadata['LeonardoAI'])
                
                # Map Leonardo.AI format to universal format
                metadata.update({
                    'prompt': leonardo_data.get('prompt'),
                    'negative_prompt': leonardo_data.get('negative_prompt'),
                    'leonardo_preset': leonardo_data.get('modelId'),
                    'width': leonardo_data.get('width'),
                    'height': leonardo_data.get('height'),
                    'cfg_scale': leonardo_data.get('guidance_scale'),
                    'steps': leonardo_data.get('num_inference_steps'),
                    'seed': leonardo_data.get('seed')
                })
            except:
                pass
        
        return metadata
    
    def _parse_generic_metadata(self, raw_metadata: Dict) -> Dict:
        """Parse generic/unknown metadata format"""
        metadata = {}
        
        # Try to extract common fields from any available data
        for key, value in raw_metadata.items():
            if key in ['prompt', 'negative_prompt', 'steps', 'cfg_scale', 'seed', 'width', 'height']:
                if isinstance(value, str) and self._is_json_like(value):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            metadata.update(parsed)
                        else:
                            metadata[key] = parsed
                    except:
                        metadata[key] = value
                else:
                    metadata[key] = value
        
        return metadata


# Global parser instance
metadata_parser = MetadataParser()
