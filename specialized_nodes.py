"""
MetaMan Workflow Saver Node
Specialized node for saving and embedding workflows in images
"""

import torch
import json
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
import folder_paths
from .metadata_parser import metadata_parser


class MetaManWorkflowSaver:
    """
    Dedicated node for saving workflows with various output options
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "save_format": ([
                    "embed_in_image", "json_file", "both", "workflow_only", "prompt_only"
                ], {"default": "embed_in_image"}),
                "filename_prefix": ("STRING", {"default": "MetaMan_workflow"}),
            },
            "optional": {
                "workflow_data": ("STRING", {"default": ""}),
                "include_metadata": ("BOOLEAN", {"default": True}),
                "compress_data": ("BOOLEAN", {"default": True}),
                "preserve_original": ("BOOLEAN", {"default": True}),
                "output_directory": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "workflow_info", "file_path")
    FUNCTION = "save_workflow"
    CATEGORY = "MetaMan"
    DESCRIPTION = "Save workflows as files or embed in images"

    def save_workflow(self, image, save_format, filename_prefix, 
                     workflow_data="", include_metadata=True, compress_data=True,
                     preserve_original=True, output_directory=""):
        """
        Save workflow data in various formats
        """
        try:
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                img_tensor = image[0]
                img_array = (img_tensor.cpu().numpy() * 255).astype('uint8')
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = image
            
            # Determine output directory
            if not output_directory:
                output_directory = folder_paths.get_output_directory()
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{filename_prefix}_{timestamp}"
            
            workflow_info = ""
            file_path = ""
            
            # Extract or prepare workflow data
            if not workflow_data:
                workflow_data = self._extract_workflow_from_image(pil_image)
            
            if save_format == "embed_in_image":
                result_image, info = self._embed_workflow_in_image(
                    pil_image, workflow_data, include_metadata, compress_data, preserve_original
                )
                # Save the image with embedded workflow
                image_path = os.path.join(output_directory, f"{base_filename}.png")
                result_image.save(image_path, "PNG")
                
                workflow_info = info
                file_path = image_path
                output_image = result_image
                
            elif save_format == "json_file":
                json_path = os.path.join(output_directory, f"{base_filename}.json")
                workflow_info = self._save_workflow_json(workflow_data, json_path, include_metadata)
                file_path = json_path
                output_image = pil_image
                
            elif save_format == "both":
                # Embed in image
                result_image, embed_info = self._embed_workflow_in_image(
                    pil_image, workflow_data, include_metadata, compress_data, preserve_original
                )
                image_path = os.path.join(output_directory, f"{base_filename}.png")
                result_image.save(image_path, "PNG")
                
                # Save JSON file
                json_path = os.path.join(output_directory, f"{base_filename}.json")
                json_info = self._save_workflow_json(workflow_data, json_path, include_metadata)
                
                workflow_info = f"Image: {embed_info}\nJSON: {json_info}"
                file_path = f"Image: {image_path}, JSON: {json_path}"
                output_image = result_image
                
            elif save_format == "workflow_only":
                workflow_path = os.path.join(output_directory, f"{base_filename}_workflow.json")
                workflow_info = self._save_workflow_only(workflow_data, workflow_path)
                file_path = workflow_path
                output_image = pil_image
                
            elif save_format == "prompt_only":
                prompt_path = os.path.join(output_directory, f"{base_filename}_prompt.json")
                workflow_info = self._save_prompt_only(workflow_data, prompt_path)
                file_path = prompt_path
                output_image = pil_image
            
            # Convert back to tensor if needed
            if isinstance(image, torch.Tensor):
                if isinstance(output_image, Image.Image):
                    img_array = torch.from_numpy(output_image.__array__()).float() / 255.0
                    output_tensor = img_array.unsqueeze(0)
                    return (output_tensor, workflow_info, file_path)
            
            return (output_image, workflow_info, file_path)
            
        except Exception as e:
            error_msg = f"MetaMan Workflow Saver Error: {str(e)}"
            return (image, error_msg, "")
    
    def _extract_workflow_from_image(self, image: Image.Image) -> Dict:
        """Extract existing workflow data from image"""
        extracted = metadata_parser.extract_all_metadata(image)
        
        workflow_data = {}
        
        # Get ComfyUI workflow if available
        if 'comfyui_workflow' in extracted['parsed_metadata']:
            workflow_data['workflow'] = extracted['parsed_metadata']['comfyui_workflow']
        
        # Get ComfyUI prompt if available
        if 'comfyui_prompt' in extracted['parsed_metadata']:
            workflow_data['prompt'] = extracted['parsed_metadata']['comfyui_prompt']
        
        # Include all parsed metadata
        workflow_data['metadata'] = extracted['parsed_metadata']
        workflow_data['source_service'] = extracted['source_detected']
        workflow_data['extracted_at'] = datetime.now().isoformat()
        
        return workflow_data
    
    def _embed_workflow_in_image(self, image: Image.Image, workflow_data: Dict, 
                                include_metadata: bool, compress_data: bool, 
                                preserve_original: bool) -> tuple[Image.Image, str]:
        """Embed workflow data into image PNG chunks"""
        
        # Create new image or copy original
        if preserve_original and hasattr(image, 'copy'):
            result_image = image.copy()
        else:
            result_image = image
        
        # Prepare PNG info
        png_info = PngInfo()
        
        # Copy existing metadata if preserving original
        if preserve_original and hasattr(image, 'text'):
            for key, value in image.text.items():
                png_info.add_text(key, value)
        
        chunks_added = []
        
        # Add workflow data
        if isinstance(workflow_data, dict):
            if 'workflow' in workflow_data:
                workflow_json = json.dumps(workflow_data['workflow'], separators=(',', ':'))
                png_info.add_text("workflow", workflow_json, zip=compress_data)
                chunks_added.append("workflow")
            
            if 'prompt' in workflow_data:
                prompt_json = json.dumps(workflow_data['prompt'], separators=(',', ':'))
                png_info.add_text("prompt", prompt_json, zip=compress_data)
                chunks_added.append("prompt")
            
            # Add universal metadata if requested
            if include_metadata and 'metadata' in workflow_data:
                # Create MetaMan universal chunk
                universal_meta = {
                    'schema_version': '1.0.0',
                    'created_at': datetime.now().isoformat(),
                    'metaman_version': '1.0.0',
                    'source_service': workflow_data.get('source_service', 'unknown'),
                    'metadata': workflow_data['metadata']
                }
                meta_json = json.dumps(universal_meta, separators=(',', ':'))
                png_info.add_text("meta", meta_json, zip=compress_data)
                chunks_added.append("meta")
        
        # Create new image with metadata
        if chunks_added:
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
        
        info_message = f"Embedded chunks: {', '.join(chunks_added)}"
        
        return result_image, info_message
    
    def _save_workflow_json(self, workflow_data: Dict, file_path: str, include_metadata: bool) -> str:
        """Save complete workflow data as JSON file"""
        
        output_data = {
            'metaman_export': {
                'version': '1.0.0',
                'exported_at': datetime.now().isoformat(),
                'export_type': 'complete_workflow'
            }
        }
        
        if isinstance(workflow_data, dict):
            output_data.update(workflow_data)
        else:
            output_data['workflow_data'] = workflow_data
        
        if not include_metadata:
            output_data.pop('metadata', None)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return f"Saved complete workflow to {os.path.basename(file_path)}"
    
    def _save_workflow_only(self, workflow_data: Dict, file_path: str) -> str:
        """Save only the workflow graph data"""
        
        if isinstance(workflow_data, dict) and 'workflow' in workflow_data:
            workflow_only = workflow_data['workflow']
        else:
            workflow_only = workflow_data
        
        output_data = {
            'metaman_export': {
                'version': '1.0.0',
                'exported_at': datetime.now().isoformat(),
                'export_type': 'workflow_only'
            },
            'workflow': workflow_only
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return f"Saved workflow graph to {os.path.basename(file_path)}"
    
    def _save_prompt_only(self, workflow_data: Dict, file_path: str) -> str:
        """Save only the prompt execution data"""
        
        if isinstance(workflow_data, dict) and 'prompt' in workflow_data:
            prompt_only = workflow_data['prompt']
        else:
            prompt_only = {}
        
        output_data = {
            'metaman_export': {
                'version': '1.0.0',
                'exported_at': datetime.now().isoformat(),
                'export_type': 'prompt_only'
            },
            'prompt': prompt_only
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return f"Saved prompt data to {os.path.basename(file_path)}"


class MetaManDependencyResolver:
    """
    Specialized node for resolving model dependencies and finding download sources
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "search_platforms": ([
                    "all", "civitai_only", "huggingface_only", "civitai_then_hf"
                ], {"default": "all"}),
                "output_format": ([
                    "detailed_json", "download_urls", "summary_text", "dependency_list"
                ], {"default": "detailed_json"}),
            },
            "optional": {
                "include_confidence": ("BOOLEAN", {"default": True}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_results_per_model": ("INT", {"default": 3, "min": 1, "max": 10}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "dependencies", "summary")
    FUNCTION = "resolve_dependencies"
    CATEGORY = "MetaMan"
    DESCRIPTION = "Resolve model dependencies and find download sources"

    def resolve_dependencies(self, image, search_platforms, output_format,
                           include_confidence=True, min_confidence=0.3,
                           max_results_per_model=3, cache_results=True):
        """
        Resolve model dependencies from image metadata
        """
        try:
            from .api_integration import model_resolver
            
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                img_tensor = image[0]
                img_array = (img_tensor.cpu().numpy() * 255).astype('uint8')
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = image
            
            # Extract metadata
            extracted = metadata_parser.extract_all_metadata(pil_image)
            metadata = extracted['parsed_metadata']
            
            # Find dependencies
            dependencies = self._find_all_dependencies(metadata)
            
            # Resolve each dependency
            resolved_dependencies = []
            
            for dep in dependencies:
                if search_platforms in ["all", "civitai_only", "civitai_then_hf"]:
                    sources = model_resolver.find_model_sources(
                        dep['name'], 
                        dep.get('hash', ''), 
                        dep['type']
                    )
                    
                    # Filter by confidence and limit results
                    filtered_sources = [
                        s for s in sources 
                        if s.get('confidence', 0) >= min_confidence
                    ][:max_results_per_model]
                    
                    dep['sources'] = filtered_sources
                    dep['search_status'] = 'found' if filtered_sources else 'not_found'
                else:
                    dep['sources'] = []
                    dep['search_status'] = 'skipped'
                
                resolved_dependencies.append(dep)
            
            # Format output
            if output_format == "detailed_json":
                output = self._format_detailed_json(resolved_dependencies, include_confidence)
            elif output_format == "download_urls":
                output = self._format_download_urls(resolved_dependencies)
            elif output_format == "summary_text":
                output = self._format_summary_text(resolved_dependencies)
            elif output_format == "dependency_list":
                output = self._format_dependency_list(resolved_dependencies)
            else:
                output = json.dumps(resolved_dependencies, indent=2)
            
            # Generate summary
            summary = self._generate_summary(resolved_dependencies)
            
            return (image, output, summary)
            
        except Exception as e:
            error_msg = f"MetaMan Dependency Resolver Error: {str(e)}"
            return (image, error_msg, "")
    
    def _find_all_dependencies(self, metadata: Dict) -> List[Dict]:
        """Find all model dependencies in metadata"""
        dependencies = []
        
        # Main model
        if 'model_name' in metadata and metadata['model_name']:
            dep = {
                'type': 'checkpoint',
                'name': metadata['model_name'],
                'hash': metadata.get('model_hash', ''),
                'required': True,
                'weight': None
            }
            dependencies.append(dep)
        
        # LoRAs
        if 'loras' in metadata and isinstance(metadata['loras'], list):
            for lora in metadata['loras']:
                if isinstance(lora, dict) and lora.get('name'):
                    dep = {
                        'type': 'lora',
                        'name': lora['name'],
                        'hash': lora.get('hash', ''),
                        'weight': lora.get('weight', 1.0),
                        'required': True
                    }
                    dependencies.append(dep)
        
        # Embeddings
        if 'embeddings' in metadata and isinstance(metadata['embeddings'], list):
            for embedding in metadata['embeddings']:
                if isinstance(embedding, dict) and embedding.get('name'):
                    dep = {
                        'type': 'embedding',
                        'name': embedding['name'],
                        'hash': embedding.get('hash', ''),
                        'weight': None,
                        'required': True
                    }
                    dependencies.append(dep)
        
        # VAE
        if 'vae' in metadata and metadata['vae']:
            dep = {
                'type': 'vae',
                'name': metadata['vae'],
                'hash': '',
                'weight': None,
                'required': False
            }
            dependencies.append(dep)
        
        # ControlNet
        if 'controlnet' in metadata and isinstance(metadata['controlnet'], list):
            for controlnet in metadata['controlnet']:
                if isinstance(controlnet, dict) and controlnet.get('model'):
                    dep = {
                        'type': 'controlnet',
                        'name': controlnet['model'],
                        'hash': '',
                        'weight': controlnet.get('weight', 1.0),
                        'required': False
                    }
                    dependencies.append(dep)
        
        return dependencies
    
    def _format_detailed_json(self, dependencies: List[Dict], include_confidence: bool) -> str:
        """Format as detailed JSON"""
        output = {
            'metaman_dependencies': {
                'version': '1.0.0',
                'generated_at': datetime.now().isoformat(),
                'total_dependencies': len(dependencies),
                'found_count': len([d for d in dependencies if d.get('search_status') == 'found']),
            },
            'dependencies': dependencies
        }
        
        if not include_confidence:
            # Remove confidence scores
            for dep in output['dependencies']:
                for source in dep.get('sources', []):
                    source.pop('confidence', None)
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def _format_download_urls(self, dependencies: List[Dict]) -> str:
        """Format as simple download URL list"""
        urls = []
        
        for dep in dependencies:
            for source in dep.get('sources', []):
                if source.get('download_url'):
                    urls.append(f"{dep['name']}: {source['download_url']}")
        
        return '\n'.join(urls) if urls else "No download URLs found"
    
    def _format_summary_text(self, dependencies: List[Dict]) -> str:
        """Format as human-readable summary"""
        lines = []
        lines.append("Model Dependencies Summary")
        lines.append("=" * 30)
        
        for dep in dependencies:
            status = "✅" if dep.get('search_status') == 'found' else "❌"
            lines.append(f"{status} {dep['type'].title()}: {dep['name']}")
            
            if dep.get('hash'):
                lines.append(f"   Hash: {dep['hash'][:16]}...")
            
            if dep.get('weight') is not None:
                lines.append(f"   Weight: {dep['weight']}")
            
            sources = dep.get('sources', [])
            if sources:
                lines.append(f"   Sources found: {len(sources)}")
                for i, source in enumerate(sources[:2], 1):
                    lines.append(f"   {i}. {source['platform']}: {source.get('confidence', 0):.2f} confidence")
            else:
                lines.append("   No sources found")
            
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_dependency_list(self, dependencies: List[Dict]) -> str:
        """Format as simple dependency list"""
        lines = []
        
        for dep in dependencies:
            line = f"{dep['type']}: {dep['name']}"
            if dep.get('weight') is not None:
                line += f" (weight: {dep['weight']})"
            if dep.get('hash'):
                line += f" [hash: {dep['hash'][:8]}...]"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _generate_summary(self, dependencies: List[Dict]) -> str:
        """Generate a brief summary"""
        total = len(dependencies)
        found = len([d for d in dependencies if d.get('search_status') == 'found'])
        
        summary = f"Dependencies: {found}/{total} found"
        
        # Break down by type
        by_type = {}
        for dep in dependencies:
            dep_type = dep['type']
            if dep_type not in by_type:
                by_type[dep_type] = {'total': 0, 'found': 0}
            by_type[dep_type]['total'] += 1
            if dep.get('search_status') == 'found':
                by_type[dep_type]['found'] += 1
        
        type_summaries = []
        for dep_type, counts in by_type.items():
            type_summaries.append(f"{dep_type}: {counts['found']}/{counts['total']}")
        
        if type_summaries:
            summary += f" ({', '.join(type_summaries)})"
        
        return summary


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS.update({
    "MetaManWorkflowSaver": MetaManWorkflowSaver,
    "MetaManDependencyResolver": MetaManDependencyResolver
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "MetaManWorkflowSaver": "MetaMan Workflow Saver",
    "MetaManDependencyResolver": "MetaMan Dependency Resolver"
})
