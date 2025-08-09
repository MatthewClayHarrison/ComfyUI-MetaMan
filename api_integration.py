"""
MetaMan API Integration Module
Handles model dependency resolution across various platforms
"""

import requests
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote


class ModelDependencyResolver:
    """
    Resolves model dependencies by searching across multiple platforms
    Priority: Civitai → HuggingFace → Other sources
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MetaMan/1.0.0 (ComfyUI Custom Node)'
        })
        
        # Rate limiting
        self.last_request_time = {}
        self.min_interval = {
            'civitai': 1.0,      # 1 second between requests
            'huggingface': 0.5,  # 0.5 seconds between requests
            'other': 2.0         # 2 seconds for other services
        }
    
    def _rate_limit(self, service: str):
        """Apply rate limiting for API requests"""
        now = time.time()
        last_time = self.last_request_time.get(service, 0)
        min_interval = self.min_interval.get(service, 1.0)
        
        time_since_last = now - last_time
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time[service] = time.time()
    
    def find_model_sources(self, name: str, hash_value: str = "", model_type: str = "checkpoint") -> List[Dict]:
        """
        Find download sources for a model across all platforms
        """
        sources = []
        
        # Search Civitai first (highest priority)
        civitai_results = self._search_civitai(name, hash_value, model_type)
        sources.extend(civitai_results)
        
        # Search HuggingFace if no exact match found
        if not any(s.get('confidence', 0) >= 0.9 for s in sources):
            hf_results = self._search_huggingface(name, model_type)
            sources.extend(hf_results)
        
        # TODO: Add other sources (Tensor.AI, archives, etc.)
        
        # Sort by confidence score
        sources.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return sources[:5]  # Return top 5 results
    
    def _search_civitai(self, name: str, hash_value: str, model_type: str) -> List[Dict]:
        """Search Civitai for models"""
        sources = []
        
        try:
            self._rate_limit('civitai')
            
            # Search by name
            search_url = "https://civitai.com/api/v1/models"
            params = {
                'query': name,
                'type': self._map_model_type_civitai(model_type),
                'sort': 'Highest Rated',
                'limit': 10
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for model in data.get('items', []):
                    confidence = self._calculate_confidence_civitai(model, name, hash_value)
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        # Get the latest version or find by hash
                        best_version = self._find_best_version_civitai(model, hash_value)
                        
                        if best_version:
                            source = {
                                'platform': 'civitai',
                                'model_id': model['id'],
                                'version_id': best_version['id'],
                                'name': model['name'],
                                'download_url': best_version.get('downloadUrl', ''),
                                'hash': best_version.get('files', [{}])[0].get('hashes', {}).get('SHA256', ''),
                                'confidence': confidence,
                                'model_page': f"https://civitai.com/models/{model['id']}",
                                'file_size': best_version.get('files', [{}])[0].get('sizeKB', 0),
                                'created_at': best_version.get('createdAt', ''),
                                'tags': model.get('tags', [])
                            }
                            sources.append(source)
            
        except Exception as e:
            print(f"MetaMan: Error searching Civitai: {e}")
        
        return sources
    
    def _search_huggingface(self, name: str, model_type: str) -> List[Dict]:
        """Search HuggingFace for models"""
        sources = []
        
        try:
            self._rate_limit('huggingface')
            
            # HuggingFace search API
            search_url = "https://huggingface.co/api/models"
            params = {
                'search': name,
                'filter': self._map_model_type_huggingface(model_type),
                'sort': 'downloads',
                'direction': -1,
                'limit': 10
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                models = response.json()
                
                for model in models:
                    confidence = self._calculate_confidence_huggingface(model, name)
                    
                    if confidence > 0.3:
                        source = {
                            'platform': 'huggingface',
                            'model_id': model['modelId'],
                            'name': model['modelId'],
                            'download_url': f"https://huggingface.co/{model['modelId']}/resolve/main/",
                            'confidence': confidence,
                            'model_page': f"https://huggingface.co/{model['modelId']}",
                            'downloads': model.get('downloads', 0),
                            'likes': model.get('likes', 0),
                            'created_at': model.get('createdAt', ''),
                            'tags': model.get('tags', [])
                        }
                        sources.append(source)
            
        except Exception as e:
            print(f"MetaMan: Error searching HuggingFace: {e}")
        
        return sources
    
    def _map_model_type_civitai(self, model_type: str) -> str:
        """Map universal model type to Civitai format"""
        mapping = {
            'checkpoint': 'Checkpoint',
            'lora': 'LORA',
            'embedding': 'TextualInversion',
            'vae': 'VAE',
            'controlnet': 'Controlnet'
        }
        return mapping.get(model_type, 'Checkpoint')
    
    def _map_model_type_huggingface(self, model_type: str) -> str:
        """Map universal model type to HuggingFace filter"""
        mapping = {
            'checkpoint': 'text-to-image',
            'lora': 'text-to-image', 
            'embedding': 'feature-extraction',
            'vae': 'text-to-image',
            'controlnet': 'text-to-image'
        }
        return mapping.get(model_type, 'text-to-image')
    
    def _calculate_confidence_civitai(self, model: Dict, name: str, hash_value: str) -> float:
        """Calculate confidence score for Civitai match"""
        confidence = 0.0
        
        # Exact hash match = maximum confidence
        if hash_value:
            for version in model.get('modelVersions', []):
                for file in version.get('files', []):
                    if file.get('hashes', {}).get('SHA256', '').lower() == hash_value.lower():
                        return 1.0
        
        # Name similarity
        model_name = model.get('name', '').lower()
        search_name = name.lower()
        
        if model_name == search_name:
            confidence += 0.8
        elif search_name in model_name or model_name in search_name:
            confidence += 0.6
        elif any(word in model_name for word in search_name.split()):
            confidence += 0.4
        
        # Boost confidence for popular models
        stats = model.get('stats', {})
        downloads = stats.get('downloadCount', 0)
        rating = stats.get('rating', 0)
        
        if downloads > 10000:
            confidence += 0.1
        if rating > 4.0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_confidence_huggingface(self, model: Dict, name: str) -> float:
        """Calculate confidence score for HuggingFace match"""
        confidence = 0.0
        
        # Name similarity
        model_name = model.get('modelId', '').lower()
        search_name = name.lower()
        
        if model_name == search_name:
            confidence += 0.7  # Slightly lower than Civitai since no hash matching
        elif search_name in model_name or model_name in search_name:
            confidence += 0.5
        elif any(word in model_name for word in search_name.split()):
            confidence += 0.3
        
        # Boost confidence for popular models
        downloads = model.get('downloads', 0)
        likes = model.get('likes', 0)
        
        if downloads > 1000:
            confidence += 0.1
        if likes > 100:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _find_best_version_civitai(self, model: Dict, hash_value: str) -> Optional[Dict]:
        """Find the best version of a Civitai model"""
        versions = model.get('modelVersions', [])
        
        if not versions:
            return None
        
        # If hash is provided, try to find exact match
        if hash_value:
            for version in versions:
                for file in version.get('files', []):
                    if file.get('hashes', {}).get('SHA256', '').lower() == hash_value.lower():
                        return version
        
        # Otherwise, return the latest version
        return versions[0]  # Civitai returns versions sorted by creation date
    
    def get_download_info(self, model_id: str, version_id: str, platform: str) -> Optional[Dict]:
        """Get detailed download information for a specific model version"""
        if platform == 'civitai':
            return self._get_civitai_download_info(model_id, version_id)
        elif platform == 'huggingface':
            return self._get_huggingface_download_info(model_id)
        
        return None
    
    def _get_civitai_download_info(self, model_id: str, version_id: str) -> Optional[Dict]:
        """Get download info from Civitai"""
        try:
            self._rate_limit('civitai')
            
            url = f"https://civitai.com/api/v1/models/{model_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Find the specific version
                for version in data.get('modelVersions', []):
                    if str(version['id']) == str(version_id):
                        return {
                            'download_url': version.get('downloadUrl', ''),
                            'files': version.get('files', []),
                            'requirements': version.get('trainedWords', []),
                            'training_info': version.get('trainingDetails', {}),
                            'images': version.get('images', [])
                        }
            
        except Exception as e:
            print(f"MetaMan: Error getting Civitai download info: {e}")
        
        return None
    
    def _get_huggingface_download_info(self, model_id: str) -> Optional[Dict]:
        """Get download info from HuggingFace"""
        try:
            self._rate_limit('huggingface')
            
            # Get model info
            url = f"https://huggingface.co/api/models/{model_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get file list
                files_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
                files_response = self.session.get(files_url, timeout=10)
                
                files = []
                if files_response.status_code == 200:
                    files = files_response.json()
                
                return {
                    'download_url': f"https://huggingface.co/{model_id}/resolve/main/",
                    'files': files,
                    'model_info': data,
                    'git_url': f"https://huggingface.co/{model_id}.git"
                }
            
        except Exception as e:
            print(f"MetaMan: Error getting HuggingFace download info: {e}")
        
        return None


class ModelHashVerifier:
    """
    Utilities for verifying model file hashes and integrity
    """
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """Calculate hash of a local file"""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def verify_hash(file_path: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify if file hash matches expected value"""
        calculated_hash = ModelHashVerifier.calculate_file_hash(file_path, algorithm)
        return calculated_hash.lower() == expected_hash.lower()
    
    @staticmethod
    def extract_hash_from_filename(filename: str) -> Optional[str]:
        """Extract hash from filename if present"""
        # Common patterns: model_name_[hash].safetensors, model_name.[hash].ckpt
        import re
        
        # Look for hex strings of typical hash lengths
        patterns = [
            r'[a-fA-F0-9]{64}',  # SHA256
            r'[a-fA-F0-9]{40}',  # SHA1  
            r'[a-fA-F0-9]{32}',  # MD5
            r'[a-fA-F0-9]{16}',  # Short hash
            r'[a-fA-F0-9]{8}'    # Very short hash
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(0)
        
        return None


# Global instance for use by MetaMan node
model_resolver = ModelDependencyResolver()
hash_verifier = ModelHashVerifier()
