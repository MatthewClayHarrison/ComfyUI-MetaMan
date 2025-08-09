# MetaMan Project Context Bootstrap

## Project Overview
**MetaMan** is a comprehensive ComfyUI custom node system for universal metadata management across AI image generation platforms. Built in August 2025, it solves the critical interoperability problem between different AI image generation services.

## Core Concept & Architecture
- **Universal Metadata Standard**: Custom "meta" PNG chunk containing JSON-structured metadata
- **Template-Driven System**: YAML-based service definitions for easy platform extension
- **Cross-Platform Workflow Generation**: Convert ANY platform's metadata into working ComfyUI workflows that recreate identical images
- **Bidirectional Conversion**: Translate metadata between platforms while preserving generation capability

## User Requirements Implemented

### Original User Stories (All ✅ Completed):
A. **Rewrite Metadata on A1111/Civitai Workflow Standards** - Converts any source to A1111/Civitai-compatible formats
B. **Generate new PNG with A1111/Civitai Standards from existing** - Creates new images with embedded target-format metadata  
C. **Extract Metadata for use in existing workflow** - Universal extraction with service-specific conversions
D. **Extract Metadata and create/update text file** - Multiple export formats (JSON, TXT, parameters)
E. **Output Metadata as string** - All nodes return formatted metadata strings
F. **Create dependency list with download URLs** - Automated model dependency resolution via Civitai/HuggingFace APIs

### Key Design Decisions:
1. **Single Service Selection**: User chooses target service rather than storing multiple formats per image
2. **Custom "meta" PNG Chunk**: Universal format with all discrete data elements from all services
3. **Template-Based Extensibility**: Service support via YAML templates, no code changes needed
4. **Workflow Embedding**: Save workflows as JSON files or embed back into original images

## Technical Architecture

### Core Components:
- **`metaman.py`** (renamed from metamann.py): MetaManUniversalNode - primary metadata extraction & conversion
- **`specialized_nodes.py`**: MetaManWorkflowSaver & MetaManDependencyResolver nodes
- **`metadata_parser.py`**: Universal parser for all platform formats with service auto-detection
- **`api_integration.py`**: Model repository APIs (Civitai, HuggingFace) with rate limiting

### Template System:
- **`templates/universal_schema.yaml`**: Defines 60+ metadata fields across all platforms
- **`templates/services/`**: Platform-specific conversion templates:
  - `automatic1111.yaml` - A1111 parameter format
  - `comfyui.yaml` - ComfyUI workflow + prompt format  
  - `civitai.yaml` - Enhanced A1111 with resource tracking
  - `forge.yaml` - Forge extensions and advanced sampling
  - `tensor.ai.yaml` - JSON-based metadata
  - `leonardo.ai.yaml` - Leonardo.AI preset system

### Supported Platforms:
✅ **Automatic1111 (A1111)** - Full parameter format support  
✅ **ComfyUI** - Complete workflow preservation  
✅ **Civitai** - Enhanced resource tracking with model IDs  
✅ **Forge** - A1111-compatible with extensions  
✅ **Tensor.AI** - JSON-based metadata format  
✅ **Leonardo.AI** - Style and preset management  
🔄 **SeaArt.AI** - Planned  
🔄 **Midjourney** - Planned  

## Key Terminology & Concepts

### Workflow Types:
- **Simplified Workflow**: Functional workflows generated from any platform metadata that recreate identical images
- **Complex Workflow Preservation**: Full-fidelity preservation of ComfyUI node graphs and custom arrangements

### Metadata Formats:
- **Universal Format**: MetaMan's comprehensive JSON schema covering all platforms
- **Service-Specific Format**: Platform-native metadata (A1111 parameters, ComfyUI workflow JSON, etc.)
- **PNG Chunks**: Text metadata embedded in PNG files (parameters, workflow, meta)

### Dependencies:
- **Model Dependencies**: Checkpoints, LoRAs, embeddings, VAE, ControlNet models
- **Dependency Resolution**: Multi-platform search with confidence scoring
- **Download URLs**: Direct links to model files with source platform info

## Project Structure
```
MetaMan/
├── Core Nodes
│   ├── __init__.py                 # Package entry point
│   ├── metaman.py                  # Universal metadata node
│   └── specialized_nodes.py        # Workflow saver & dependency resolver
├── Utilities  
│   ├── metadata_parser.py          # Universal format parser
│   ├── api_integration.py          # Model repository APIs
│   └── validation.py               # Testing framework
├── Templates
│   ├── universal_schema.yaml       # Universal metadata definition
│   └── services/                   # Platform-specific templates (6 services)
├── Documentation
│   ├── README.md                   # Main documentation with badges
│   ├── SETUP.md                    # Installation & configuration
│   ├── CONTRIBUTING.md             # Community contribution guidelines
│   └── CHANGELOG.md                # Version history
├── Testing & Project Management
│   ├── test_installation.py        # Installation verification
│   ├── validation.py               # Template and functionality testing
│   ├── requirements.txt            # Python dependencies
│   ├── LICENSE                     # MIT License
│   └── .gitignore                  # Git ignore rules
```

## Current Status (August 2025)
- ✅ **Complete codebase** ready for deployment
- ✅ **All user stories implemented** and tested
- ✅ **Documentation complete** with setup guides
- ✅ **Git repository prepared** for public release
- 🚀 **Ready for GitHub deployment** - just needs repo creation and push

## Technical Specifications

### Dependencies:
- Pillow >= 9.0.0 (image processing)
- PyYAML >= 6.0.0 (template processing)  
- piexif >= 1.1.3 (EXIF metadata)
- requests >= 2.28.0 (API integration)
- jsonschema >= 4.17.0 (validation)
- torch (provided by ComfyUI)

### Node Interface:
**MetaMan Universal Node Operations:**
- `extract_universal` - Extract to universal format
- `convert_to_service` - Convert to specific service format
- `save_workflow` - Save workflow data
- `generate_dependencies` - Find model dependencies  
- `export_metadata` - Export in various formats

**Input/Output Types:**
- Input: IMAGE, operation selection, service selection, optional parameters
- Output: IMAGE, universal_metadata (STRING), service_metadata (STRING), dependencies (STRING)

## Key Research & Terminology

### Platform-Specific Details:
- **A1111 Format**: Text-based parameters in PNG tEXt chunks, Latin-1 encoding
- **ComfyUI Format**: JSON workflow graphs with node connections and execution data
- **Civitai Extensions**: A1111 format + resource tracking with SHA256 hashes and model IDs
- **PNG Chunks**: tEXt (Latin-1), iTXt (UTF-8), custom "meta" chunk proposal

### Model Repository APIs:
- **Civitai API**: REST endpoints for model search, hash matching, download URLs
- **HuggingFace API**: Git-based model repositories with commit hashes
- **Search Priority**: Civitai → HuggingFace → Archives → Other sources
- **Rate Limiting**: Respect platform limits (1-2 seconds between requests)

### Sampler/Scheduler Mappings:
Complex mappings between platform naming conventions (e.g., A1111's "DPM++ 2M Karras" = ComfyUI's "dpmpp_2m" + "karras" scheduler)

## Important Clarifications Made

### Workflow Conversion Capabilities:
- **✅ ANY platform → ComfyUI workflow generation** (creates identical images)
- **✅ ComfyUI → ANY platform parameter conversion** (simplified but equivalent)
- **⚠️ Complex ComfyUI workflows lose node structure** when converted to linear platforms
- **🎯 Image reproduction is identical** across all conversion types

### Compatibility Matrix Meanings:
- **Simplified Workflow** ✅ = Can generate functional workflow that recreates exact image
- **Complex Workflow Preservation** ✅ = Can preserve original node arrangements and custom routing
- **Basic Parameters** ✅ = Standard generation parameters (prompt, steps, seed, etc.)

## File Locations
- **Project**: `/Users/pxl8d/Projects/MetaMan/`
- **ComfyUI Target**: `/Users/pxl8d/Art/ComfyUI/custom_nodes/`
- **Git Status**: Initialized repository ready for remote push

## Next Steps
1. Create GitHub repository (suggested name: `ComfyUI-MetaMan`)
2. Push codebase to public repository
3. Share with ComfyUI community
4. Consider ComfyUI-Manager integration
5. Community feedback and platform additions

## Research Terms & Context Covered
- A1111/Civitai metadata standards and PNG chunk formats
- ComfyUI workflow architecture and node-based systems  
- Cross-platform compatibility challenges in AI image generation
- Model dependency tracking and resolution systems
- PNG metadata embedding techniques and encoding standards
- API rate limiting and repository search strategies
- Template-driven software architecture for extensibility