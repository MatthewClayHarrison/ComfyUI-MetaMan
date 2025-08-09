# MetaMan - Universal AI Image Metadata Manager

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/downloads/)
[![Supported Platforms](https://img.shields.io/badge/Platforms-6%2B-brightgreen)](#supported-platforms)

A comprehensive ComfyUI custom node for managing, converting, and standardizing metadata across all major AI image generation platforms.

## 🌟 Features

MetaMan bridges the metadata gap between AI image generation services, enabling seamless workflow sharing and model dependency tracking across platforms.

### Core Capabilities

- **Universal Metadata Extraction**: Read metadata from any AI image generation service
- **Cross-Platform Conversion**: Convert metadata between A1111, ComfyUI, Civitai, Tensor.AI, Leonardo.AI, and more
- **Model Dependency Tracking**: Automatically identify and resolve model dependencies with download URLs
- **Workflow Preservation**: Save and restore complete workflows across platforms
- **Template-Based Extensibility**: Easy addition of new services via YAML templates

### Supported Platforms

✅ **Automatic1111 (A1111)** - Full parameter format support  
✅ **ComfyUI** - Complete workflow preservation  
✅ **Civitai** - Enhanced resource tracking with model IDs  
✅ **Forge** - A1111-compatible with extensions  
✅ **Tensor.AI** - JSON-based metadata format  
✅ **Leonardo.AI** - Style and preset management  
🔄 **SeaArt.AI** - Coming soon  
🔄 **Midjourney** - Coming soon  

## 🚀 Installation

1. **Clone to ComfyUI custom nodes:**
   ```bash
   cd /path/to/ComfyUI/custom_nodes/
   git clone <repository-url> MetaMan
   ```

2. **Install dependencies:**
   ```bash
   cd MetaMan
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI**

The MetaMan Universal node will appear in the **MetaMan** category.

## 📖 How It Works

### Universal Metadata Standard

MetaMan uses a comprehensive **universal metadata schema** that encompasses all possible metadata fields from every supported platform. The system:

1. **Extracts** metadata from source images regardless of origin platform
2. **Converts** to universal format with field mapping and compatibility tracking  
3. **Transforms** to target platform format using service-specific templates
4. **Preserves** original workflows and enables cross-platform reproduction

### Template System

The power of MetaMan lies in its **template-driven architecture**:

```
templates/
├── universal_schema.yaml     # Defines all possible metadata fields
└── services/                 # Service-specific output templates
    ├── automatic1111.yaml
    ├── comfyui.yaml
    ├── civitai.yaml
    ├── tensor.ai.yaml
    └── leonardo.ai.yaml
```

### Custom "meta" PNG Chunk

MetaMan introduces a standardized **"meta" PNG chunk** containing universal metadata in JSON format, enabling perfect cross-platform compatibility while maintaining service-specific formats.

## 🎯 Usage Examples

### Basic Metadata Conversion

```python
# Extract universal metadata from any image
operation: "extract_universal"
target_service: "automatic1111"
# → Outputs: universal JSON + A1111 parameter string

# Convert ComfyUI workflow to A1111 format  
operation: "convert_to_service"
target_service: "automatic1111"
# → Generates A1111-compatible parameters
```

### Model Dependency Tracking

```python
operation: "generate_dependencies"
dependency_sources: "all"  # Searches Civitai, HuggingFace, etc.
# → Returns JSON with download URLs and model information
```

### Workflow Preservation

```python
operation: "save_workflow" 
output_format: "embed_in_image"
# → Embeds complete workflow back into the original image
```

## 🔧 Node Interface

### Inputs

**Required:**
- `image` - Input image with metadata
- `operation` - Operation type:
  - `extract_universal` - Extract to universal format
  - `convert_to_service` - Convert to specific service
  - `save_workflow` - Save workflow data
  - `generate_dependencies` - Create dependency list
  - `export_metadata` - Export in various formats
- `target_service` - Target platform for conversion

**Optional:**
- `output_format` - Output format (png_chunk, json_file, txt_file, embed_in_image)
- `template_override` - Custom template path
- `include_workflow` - Include workflow data
- `include_dependencies` - Include model dependencies
- `dependency_sources` - Dependency search sources

### Outputs

- `image` - Processed image (possibly with embedded metadata)
- `universal_metadata` - Complete universal format JSON
- `service_metadata` - Target service formatted metadata  
- `dependencies` - Model dependency information with download URLs

## 🎨 Advanced Features

### Service-Specific Optimizations

Each service template includes:
- **Field mappings** - Automatic parameter translation
- **Validation rules** - Ensure platform compatibility
- **Format specifications** - Proper encoding and chunk handling
- **Model resolution** - Platform-specific model identification

### Extensibility

Adding new platforms is straightforward:

1. Create a new YAML template in `templates/services/`
2. Define field mappings and output format
3. Add to `SUPPORTED_SERVICES` list
4. MetaMan automatically incorporates the new service

### Model Dependency Resolution

MetaMan tracks and resolves:
- **Checkpoints/Base Models** with SHA256 hashes
- **LoRA models** with weights and trigger words
- **Embeddings/Textual Inversions** 
- **VAE models**
- **ControlNet models**

Priority search order: Civitai → HuggingFace → Other repositories

## 🛠 Technical Implementation

### Metadata Flow

```
Source Image → Extract Native Metadata → Universal Schema → Target Template → Output Format
```

### Schema Validation

- **Type checking** for all metadata fields
- **Range validation** for numeric parameters  
- **Format verification** for strings and arrays
- **Compatibility checking** across services

### PNG Chunk Handling

- **Automatic encoding detection** (Latin-1 vs UTF-8)
- **Compression for large data** (workflows, dependency lists)
- **Fallback mechanisms** for incompatible content
- **Multiple chunk support** (parameters + workflow + meta)

## 🔍 Troubleshooting

### Common Issues

**Metadata not detected:**
- Ensure image contains compatible metadata chunks
- Check PNG format (some services use proprietary formats)

**Conversion errors:**
- Verify source service compatibility
- Check template configuration for target service

**Missing dependencies:**
- Install all requirements: `pip install -r requirements.txt`
- Restart ComfyUI after installation

### Debug Mode

Enable verbose logging by setting environment variable:
```bash
export METAMAN_DEBUG=1
```

## 🤝 Contributing

We welcome contributions to expand platform support and improve metadata handling!

### Adding New Services

1. Research the service's metadata format
2. Create a YAML template in `templates/services/`
3. Update the universal schema if new fields are needed
4. Test with sample images from the service
5. Submit a pull request

### Template Structure

Service templates should include:
- Field mappings and transformations
- Output format specifications  
- Validation rules
- PNG chunk configuration
- API integration hints (for future automation)

## 📊 Compatibility Matrix

| Feature | A1111 | ComfyUI | Civitai | Forge | Tensor.AI | Leonardo.AI |
|---------|-------|---------|---------|--------|-----------|-------------|
| Basic Parameters | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Workflow Data | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Model Hashes | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ |
| LoRA Support | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Dependency URLs | ⚠️ | ⚠️ | ✅ | ⚠️ | ✅ | ✅ |

✅ Full Support | ⚠️ Partial Support | ❌ Not Supported

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- ComfyUI team for the excellent node-based interface
- A1111 community for establishing metadata standards  
- Civitai for advancing model sharing and compatibility
- All AI image generation platforms for pushing the boundaries of creativity

---

**MetaMan**: Making AI image metadata universal, accessible, and interoperable across all platforms.
