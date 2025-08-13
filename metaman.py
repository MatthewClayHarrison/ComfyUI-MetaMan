"""
MetaMan Node for ComfyUI
Universal metadata management for AI image generation platforms
Supports A1111, ComfyUI, Civitai, Tensor.ai, Forge, and other services
"""

import torch
import json
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import re
from datetime import datetime


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
                                # Store current metadata for workflow scanning
                                self._current_metadata = metadata
                                
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
                    
                    elif key == 'generation_data':
                        # Tensor.AI generation data
                        try:
                            # Clean null bytes before parsing JSON
                            cleaned_data = value.replace('\u0000', '').replace('\x00', '').strip()
                            metadata[f"png_chunk_{key}"] = cleaned_data
                            print(f"MetaMan Load Image: Stored Tensor.AI generation_data")
                        except Exception as e:
                            print(f"MetaMan Load Image: Error handling generation_data: {e}")
                    
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
            
            # Phase 2: Collect ALL text content from all nodes (from prompt data)
            all_text_content = self._scan_all_text_content_universal(prompt_data)
            print(f"MetaMan Universal: Found {len(all_text_content)} text candidates from prompt data")
            
            # Phase 2.5: EMERGENCY - Direct search for embedding text anywhere in metadata
            emergency_embeddings = self._emergency_scan_for_embeddings()
            if emergency_embeddings:
                print(f"MetaMan EMERGENCY: Found {len(emergency_embeddings)} embeddings via direct scan!")
            
            # Phase 2.6: CRITICAL - Direct workflow scanning for embeddings (simplified approach)
            workflow_embeddings = self._extract_embeddings_from_workflow_direct()
            if workflow_embeddings:
                print(f"MetaMan Universal: Found {len(workflow_embeddings)} embeddings directly from workflow data")
            
            # Phase 2.7: Also try the old workflow scanning method as backup
            workflow_text_content = self._scan_workflow_widget_values()
            if workflow_text_content:
                all_text_content.extend(workflow_text_content)
                print(f"MetaMan Universal: Added {len(workflow_text_content)} text candidates from workflow data")
            
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
            
            # Combine all embedding sources
            final_embeddings = all_embeddings.copy()
            
            # Add emergency embeddings first (highest priority)
            if emergency_embeddings:
                for emb in emergency_embeddings:
                    emb_name = emb['name'].lower()
                    if not any(existing['name'].lower() == emb_name for existing in final_embeddings):
                        final_embeddings.append(emb)
                        print(f"MetaMan Universal: Added EMERGENCY embedding: {emb['name']}")
            
            # Add workflow embeddings next
            if workflow_embeddings:
                for workflow_emb in workflow_embeddings:
                    emb_name = workflow_emb['name'].lower()
                    if not any(existing['name'].lower() == emb_name for existing in final_embeddings):
                        final_embeddings.append(workflow_emb)
                        print(f"MetaMan Universal: Added workflow embedding: {workflow_emb['name']}")
            
            if final_embeddings:
                params['embeddings'] = final_embeddings
                print(f"MetaMan Universal: Final extracted {len(final_embeddings)} embeddings: {[emb['name'] for emb in final_embeddings]}")
            
            print(f"MetaMan Universal: Extraction complete: {list(params.keys())}")
            
        except Exception as e:
            print(f"MetaMan Universal Extraction Error: {e}")
            import traceback
            traceback.print_exc()
            # Initialize variables as empty if there was an error
            workflow_embeddings = []
            emergency_embeddings = []
        
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
    
    def _scan_all_text_content_universal(self, prompt_data: dict) -> list:
        """Scan ALL nodes for text content with enhanced detection"""
        text_candidates = []
        
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
                
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            # Enhanced text field detection from inputs
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
    
    def _emergency_scan_for_embeddings(self) -> list:
        """Emergency function to scan ALL metadata for embedding: text"""
        embeddings = []
        found_embedding_texts = []
        
        try:
            if not hasattr(self, '_current_metadata'):
                print(f"MetaMan EMERGENCY: No metadata available")
                return embeddings
            
            print(f"MetaMan EMERGENCY: Scanning ALL metadata for 'embedding:' text...")
            
            def scan_any_value(value, path=""):
                """Recursively scan any value for embedding text"""
                if isinstance(value, str):
                    if 'embedding:' in value and value not in ['CHOOSE', 'Select']:
                        print(f"MetaMan EMERGENCY: Found embedding text at {path}: '{value[:100]}...'")
                        found_embedding_texts.append({'path': path, 'text': value})
                elif isinstance(value, dict):
                    for k, v in value.items():
                        scan_any_value(v, f"{path}.{k}")
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        scan_any_value(v, f"{path}[{i}]")
            
            # Scan EVERYTHING in current metadata
            for key, value in self._current_metadata.items():
                scan_any_value(value, key)
            
            print(f"MetaMan EMERGENCY: Found {len(found_embedding_texts)} embedding texts")
            
            # Extract embeddings from all found texts
            for item in found_embedding_texts:
                text_embeddings = self._extract_embeddings_from_text(item['text'])
                for emb in text_embeddings:
                    # Avoid duplicates
                    if not any(existing['name'].lower() == emb['name'].lower() for existing in embeddings):
                        emb['source_path'] = item['path']  # Add source info
                        embeddings.append(emb)
                        print(f"MetaMan EMERGENCY: Extracted '{emb['name']}' from {item['path']}")
            
            return embeddings
            
        except Exception as e:
            print(f"MetaMan EMERGENCY Error: {e}")
            return embeddings
    
    def _extract_embeddings_from_workflow_direct(self) -> list:
        """Direct extraction of embeddings from workflow data - simplified approach"""
        embeddings = []
        
        try:
            # Direct access to workflow data from current metadata
            if not hasattr(self, '_current_metadata'):
                print(f"MetaMan Direct: No current metadata available")
                return embeddings
            
            # Get workflow data directly
            workflow_data = None
            if 'comfyui_workflow' in self._current_metadata:
                workflow_data = self._current_metadata['comfyui_workflow']
                print(f"MetaMan Direct: Found workflow data in comfyui_workflow")
            elif 'png_chunk_workflow' in self._current_metadata:
                try:
                    workflow_data = json.loads(self._current_metadata['png_chunk_workflow'])
                    print(f"MetaMan Direct: Parsed workflow data from png_chunk_workflow")
                except:
                    print(f"MetaMan Direct: Failed to parse png_chunk_workflow")
            
            if not workflow_data or not isinstance(workflow_data, dict):
                print(f"MetaMan Direct: No valid workflow data found")
                return embeddings
            
            nodes = workflow_data.get('nodes', [])
            if not nodes:
                print(f"MetaMan Direct: No nodes in workflow data")
                return embeddings
            
            print(f"MetaMan Direct: Scanning {len(nodes)} nodes for embeddings")
            
            # Scan each node's widgets_values for embeddings
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                    
                node_id = node.get('id', 'unknown')
                node_type = node.get('type', 'unknown')
                widgets_values = node.get('widgets_values', [])
                
                if isinstance(widgets_values, list) and widgets_values:
                    for i, widget_value in enumerate(widgets_values):
                        if isinstance(widget_value, str) and 'embedding:' in widget_value:
                            print(f"MetaMan Direct: Found embedding text in node {node_id}.widget_{i}: {widget_value[:50]}...")
                            
                            # Extract embeddings from this widget value
                            widget_embeddings = self._extract_embeddings_from_text(widget_value)
                            for emb in widget_embeddings:
                                # Avoid duplicates
                                if not any(existing['name'].lower() == emb['name'].lower() for existing in embeddings):
                                    embeddings.append(emb)
                                    print(f"MetaMan Direct: Added embedding '{emb['name']}' from node {node_id}")
            
            print(f"MetaMan Direct: Final embeddings found: {len(embeddings)}")
            return embeddings
            
        except Exception as e:
            print(f"MetaMan Direct Error: {e}")
            import traceback
            traceback.print_exc()
            return embeddings
    
    def _scan_workflow_widget_values(self) -> list:
        """Enhanced workflow data scanning for widgets_values - critical for Power Prompt embeddings!"""
        text_candidates = []
        
        try:
            # Get workflow data from the instance - we need to store it during metadata extraction
            if not hasattr(self, '_current_metadata'):
                print(f"MetaMan Workflow Debug: No current metadata available for workflow scanning")
                return text_candidates
            
            # Look for workflow data in current metadata
            workflow_data = None
            for key in ['comfyui_workflow', 'png_chunk_workflow', 'info_workflow']:
                if key in self._current_metadata:
                    if isinstance(self._current_metadata[key], dict):
                        workflow_data = self._current_metadata[key]
                        print(f"MetaMan Workflow Debug: Found workflow data in {key}")
                        break
                    elif isinstance(self._current_metadata[key], str):
                        try:
                            workflow_data = json.loads(self._current_metadata[key])
                            print(f"MetaMan Workflow Debug: Parsed workflow data from {key}")
                            break
                        except:
                            pass
            
            if not workflow_data:
                print(f"MetaMan Workflow Debug: No workflow data found")
                return text_candidates
            
            # Extract nodes from workflow data
            nodes = workflow_data.get('nodes', [])
            if not nodes:
                print(f"MetaMan Workflow Debug: No nodes found in workflow data")
                return text_candidates
            
            print(f"MetaMan Workflow Debug: Scanning {len(nodes)} workflow nodes for widgets_values")
            
            # Scan each node for widgets_values
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                    
                node_id = str(node.get('id', 'unknown'))
                node_type = node.get('type', 'unknown')
                widgets_values = node.get('widgets_values', [])
                
                if isinstance(widgets_values, list) and len(widgets_values) > 0:
                    for i, widget_value in enumerate(widgets_values):
                        if isinstance(widget_value, str) and len(widget_value.strip()) > 5:
                            text_candidates.append({
                                'text': widget_value,
                                'source': f"workflow.{node_id}.widget_{i}",
                                'node_id': node_id,
                                'field_name': f"widget_{i}",
                                'node_type': node_type,
                                'length': len(widget_value),
                                'is_text_field': True,
                                'is_processed_field': False
                            })
                            
                            print(f"MetaMan Workflow: Found text in workflow.{node_id}.widget_{i} ({node_type}): {len(widget_value)} chars")
            
            print(f"MetaMan Workflow Debug: Found {len(text_candidates)} text candidates from workflow data")
            
        except Exception as e:
            print(f"MetaMan Workflow Debug Error: {e}")
            import traceback
            traceback.print_exc()
        
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
            
            # Parse the JSON from MetaManLoadImage
            metadata_data = json.loads(metadata_json)
            
            # Check if this is from our Load Image node or raw metadata
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
            
            # Extract width/height from image_size if available, otherwise from metadata
            width = int(metadata.get('width', 512))
            height = int(metadata.get('height', 512))
            
            # Check for image_size at top level of metadata_data (from MetaManLoadImage)
            if 'image_size' in metadata_data and isinstance(metadata_data['image_size'], list) and len(metadata_data['image_size']) >= 2:
                width = int(metadata_data['image_size'][0])
                height = int(metadata_data['image_size'][1])
                print(f"MetaMan Extract Components: Using image_size dimensions: {width}x{height}")
            elif 'width' in metadata and 'height' in metadata:
                print(f"MetaMan Extract Components: Using metadata dimensions: {width}x{height}")
            else:
                print(f"MetaMan Extract Components: Using default dimensions: {width}x{height}")
            
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
                "target_service": (["automatic1111", "comfyui", "civitai", "forge", "tensor.ai", "leonardo.ai"], {"default": "automatic1111"})
            },
            "optional": {
                "directory": ("STRING", {"default": "MetaMan converted"})
            }
        }
    
    CATEGORY = "MetaMan"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "embed_and_save"
    OUTPUT_NODE = True
    DESCRIPTION = "Embed metadata into image and save as PNG file"
    
    def embed_and_save(self, image, metadata_json, target_service, directory="MetaMan converted"):
        """
        Embed metadata into image and save as PNG file to output directory
        """
        try:
            # Parse metadata JSON and convert to target service format
            if metadata_json and metadata_json.strip():
                try:
                    metadata_data = json.loads(metadata_json)
                    
                    # Extract the source metadata
                    source_metadata = {}
                    if 'metadata' in metadata_data:
                        source_metadata = metadata_data['metadata']
                    else:
                        source_metadata = metadata_data
                    
                    # Convert to target service format
                    metadata_text = self._convert_metadata_to_target_format(source_metadata, target_service)
                    print(f"MetaMan Embed & Save: Converted metadata to {target_service} format ({len(metadata_text)} chars)")
                    
                except Exception as e:
                    print(f"MetaMan Embed & Save: Error parsing/converting metadata JSON: {e}")
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
            
            # Create subdirectory and save with metadata
            output_dir = folder_paths.get_output_directory()
            target_dir = os.path.join(output_dir, directory)
            
            # Create subdirectory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            print(f"MetaMan Embed & Save: Created/confirmed directory: {target_dir}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"MetaMan_{target_service}_{timestamp}.png"
            filepath = os.path.join(target_dir, filename)
            
            # Create PNG info with metadata
            png_info = PngInfo()
            png_info.add_text("parameters", metadata_text)
            png_info.add_text("metaman_converted", datetime.now().isoformat())
            png_info.add_text("metaman_version", "1.0.0")
            
            # Save image with embedded metadata
            pil_image.save(filepath, "PNG", pnginfo=png_info)
            
            print(f"MetaMan Embed & Save: Successfully saved image with metadata to {filepath}")
            print(f"MetaMan Embed & Save: File size: {os.path.getsize(filepath)} bytes")
            
            return {"ui": {"images": [{"filename": filename, "subfolder": directory, "type": "output"}]}}
            
        except Exception as e:
            print(f"MetaMan Embed & Save Error: {e}")
            return {"ui": {"text": [f"Error: {str(e)}"]}}
    
    def _convert_metadata_to_target_format(self, metadata: dict, target_service: str) -> str:
        """
        Convert metadata to the specified target service format
        """
        try:
            if target_service == "automatic1111":
                return self._format_a1111_output(metadata)
            elif target_service == "comfyui":
                return self._format_comfyui_output(metadata)
            elif target_service == "civitai":
                return self._format_civitai_output(metadata)
            elif target_service == "forge":
                return self._format_forge_output(metadata)
            elif target_service == "tensor.ai":
                return self._format_tensor_ai_output(metadata)
            elif target_service == "leonardo.ai":
                return self._format_leonardo_ai_output(metadata)
            else:
                # Generic JSON format for unknown services
                return json.dumps(metadata, indent=2)
        except Exception as e:
            print(f"MetaMan Embed & Save: Error converting to {target_service}: {e}")
            return json.dumps(metadata, indent=2)  # Fallback to JSON
    
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
    
    def _format_comfyui_output(self, metadata: dict) -> str:
        """Format metadata for ComfyUI compatibility"""
        # For ComfyUI, return structured JSON with workflow info
        comfyui_meta = {
            "prompt": metadata.get('prompt', ''),
            "negative_prompt": metadata.get('negative_prompt', ''),
            "steps": metadata.get('steps', 20),
            "cfg_scale": metadata.get('cfg_scale', 7.0),
            "sampler": metadata.get('sampler', 'euler'),
            "scheduler": metadata.get('scheduler', 'normal'),
            "seed": metadata.get('seed', -1),
            "width": metadata.get('width', 512),
            "height": metadata.get('height', 512),
            "model_name": metadata.get('model_name', ''),
            "loras": metadata.get('loras', []),
            "embeddings": metadata.get('embeddings', []),
            "metaman_converted": True,
            "conversion_time": datetime.now().isoformat()
        }
        return json.dumps(comfyui_meta, indent=2)
    
    def _format_civitai_output(self, metadata: dict) -> str:
        """Format metadata for Civitai compatibility (enhanced A1111)"""
        a1111_output = self._format_a1111_output(metadata)
        
        # Add Civitai-specific information
        civitai_additions = []
        
        # Add model hash if available
        if 'model_hash' in metadata:
            civitai_additions.append(f"Model hash: {metadata['model_hash']}")
        
        # Add LoRA information
        if 'loras' in metadata and metadata['loras']:
            lora_strings = []
            for lora in metadata['loras']:
                if isinstance(lora, dict):
                    lora_name = lora.get('real_name', lora.get('name', ''))
                    lora_weight = lora.get('weight', 1.0)
                    lora_strings.append(f"<lora:{lora_name}:{lora_weight}>")
            if lora_strings:
                civitai_additions.append(f"Lora hashes: {', '.join(lora_strings)}")
        
        if civitai_additions:
            return a1111_output + "\n" + ", ".join(civitai_additions)
        else:
            return a1111_output
    
    def _format_forge_output(self, metadata: dict) -> str:
        """Format metadata for Forge compatibility (A1111-based)"""
        return self._format_a1111_output(metadata)  # Forge uses A1111 format
    
    def _format_tensor_ai_output(self, metadata: dict) -> str:
        """Format metadata for Tensor.AI compatibility"""
        tensor_meta = {
            "prompt": metadata.get('prompt', ''),
            "negativePrompt": metadata.get('negative_prompt', ''),
            "steps": metadata.get('steps', 20),
            "guidanceScale": metadata.get('cfg_scale', 7.0),
            "sampler": metadata.get('sampler', 'euler'),
            "seed": metadata.get('seed', -1),
            "width": metadata.get('width', 512),
            "height": metadata.get('height', 512),
            "model": metadata.get('model_name', ''),
            "style": metadata.get('tensor_ai_style', 'Default'),
            "metaman_converted": True
        }
        return json.dumps(tensor_meta, indent=2)
    
    def _format_leonardo_ai_output(self, metadata: dict) -> str:
        """Format metadata for Leonardo.AI compatibility"""
        leonardo_meta = {
            "prompt": metadata.get('prompt', ''),
            "negative_prompt": metadata.get('negative_prompt', ''),
            "num_images": 1,
            "width": metadata.get('width', 512),
            "height": metadata.get('height', 512),
            "num_inference_steps": metadata.get('steps', 20),
            "guidance_scale": metadata.get('cfg_scale', 7.0),
            "modelId": metadata.get('model_name', ''),
            "preset_style": metadata.get('leonardo_preset', 'GENERAL'),
            "scheduler": metadata.get('sampler', 'EULER_DISCRETE'),
            "seed": metadata.get('seed', -1),
            "metaman_converted": True
        }
        return json.dumps(leonardo_meta, indent=2)
