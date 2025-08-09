# MetaMan Setup and Configuration Guide

This guide helps you install, configure, and troubleshoot MetaMan for ComfyUI.

## Quick Start

### 1. Installation

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

3. **Test installation:**
   ```bash
   python test_installation.py
   ```

4. **Restart ComfyUI**

### 2. First Use

1. Start ComfyUI
2. Look for "MetaMan" category in node browser
3. Add "MetaMan Universal" node to your workflow
4. Connect an image with AI metadata
5. Set operation to "extract_universal"
6. Run the workflow to see extracted metadata

## Detailed Setup

### Dependencies

MetaMan requires these Python packages:
- `Pillow` >= 9.0.0 (image processing)
- `PyYAML` >= 6.0.0 (template loading)
- `piexif` >= 1.1.3 (EXIF handling)
- `requests` >= 2.28.0 (API integration)
- `jsonschema` >= 4.17.0 (validation)

These are usually already available in ComfyUI environments:
- `torch` (tensor operations)
- `json` (metadata processing)

### Directory Structure

After installation, your MetaMan directory should look like:

```
MetaMan/
├── __init__.py                 # Main package file
├── metamann.py                 # Universal metadata node
├── specialized_nodes.py        # Workflow saver & dependency resolver
├── metadata_parser.py          # Metadata parsing utilities
├── api_integration.py          # Model repository APIs
├── validation.py               # Testing and validation
├── test_installation.py        # Installation test script
├── requirements.txt            # Python dependencies
├── README.md                   # Main documentation
├── SETUP.md                    # This file
├── templates/
│   ├── universal_schema.yaml   # Universal metadata schema
│   └── services/               # Service-specific templates
│       ├── automatic1111.yaml
│       ├── comfyui.yaml
│       ├── civitai.yaml
│       ├── forge.yaml
│       ├── tensor.ai.yaml
│       └── leonardo.ai.yaml
└── examples/
    └── README.md               # Usage examples
```

## Configuration

### Adding New Services

To add support for a new AI image generation service:

1. **Create service template:**
   ```bash
   cp templates/services/automatic1111.yaml templates/services/newservice.yaml
   ```

2. **Edit the template:**
   - Update `service_name`
   - Define `output_format`
   - Configure field mappings
   - Set PNG chunk specifications

3. **Update universal schema (if needed):**
   - Add new fields to `templates/universal_schema.yaml`
   - Specify which services support each field

4. **Test the new service:**
   ```bash
   python validation.py
   ```

### Customizing Templates

Templates are YAML files that define how metadata is converted between formats. Key sections:

- **Field mappings:** How universal fields map to service-specific formats
- **Output format:** Text, JSON, or mixed format specifications
- **PNG chunks:** How metadata is embedded in images
- **Validation rules:** Required fields and format constraints

### API Configuration

For model dependency resolution, you can configure:

- **Rate limiting:** Adjust `min_interval` in `api_integration.py`
- **Search sources:** Enable/disable specific model repositories
- **Confidence thresholds:** Minimum confidence for accepting matches

## Available Nodes

### MetaMan Universal
- **Purpose:** Main metadata extraction and conversion node
- **Operations:**
  - `extract_universal` - Extract to universal format
  - `convert_to_service` - Convert to specific service format
  - `save_workflow` - Save workflow data
  - `generate_dependencies` - Find model dependencies
  - `export_metadata` - Export in various formats

### MetaMan Workflow Saver
- **Purpose:** Specialized workflow preservation
- **Formats:**
  - `embed_in_image` - Embed workflow in PNG chunks
  - `json_file` - Save as standalone JSON
  - `both` - Both embedded and file
  - `workflow_only` - ComfyUI workflow graph only
  - `prompt_only` - ComfyUI prompt data only

### MetaMan Dependency Resolver
- **Purpose:** Find and resolve model dependencies
- **Sources:**
  - `all` - Search all available sources
  - `civitai_only` - Civitai models only
  - `huggingface_only` - HuggingFace models only
  - `civitai_then_hf` - Civitai first, then HuggingFace

## Troubleshooting

### Common Issues

**1. "MetaMan nodes not appearing in ComfyUI"**
- Check ComfyUI console for import errors
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Run installation test: `python test_installation.py`
- Restart ComfyUI completely

**2. "Import errors on startup"**
```bash
# Install missing dependencies
pip install PyYAML piexif requests jsonschema

# For conda environments
conda install -c conda-forge pyyaml piexif requests jsonschema
```

**3. "Metadata not detected from images"**
- Ensure image contains PNG text chunks with metadata
- Check supported formats in `metadata_parser.py`
- Some services use proprietary formats not yet supported

**4. "Template loading errors"**
- Verify YAML syntax in template files
- Run validation: `python validation.py`
- Check file permissions on templates directory

**5. "API integration not working"**
- Check internet connection for model repository access
- Verify rate limiting settings if getting blocked
- Some APIs require authentication (future feature)

### Debug Mode

Enable verbose logging:

```bash
export METAMAN_DEBUG=1
```

Or in Python:
```python
import os
os.environ['METAMAN_DEBUG'] = '1'
```

### Log Files

ComfyUI logs MetaMan activities to the console. Look for:
- `MetaMan:` prefixed messages
- Import errors during startup
- Node execution errors during workflow runs

### Getting Help

1. **Check the documentation:** README.md and examples/
2. **Run diagnostics:** `python test_installation.py`
3. **Validate templates:** `python validation.py`
4. **Check ComfyUI console:** Look for error messages
5. **File an issue:** Include error messages and system info

## Advanced Configuration

### Custom Universal Schema

You can extend the universal schema to support new metadata fields:

1. Edit `templates/universal_schema.yaml`
2. Add new field definitions with:
   - Field type (string, integer, float, boolean, array, object)
   - Supported services list
   - Validation rules (range, options, format)
   - Default values

3. Update service templates to handle new fields
4. Test with validation script

### Performance Tuning

For large workflows or batch processing:

- **Enable compression:** Set `compress_data=True` for PNG chunks
- **Limit API calls:** Adjust rate limiting in `api_integration.py`
- **Cache results:** Enable `cache_results=True` for dependency resolution
- **Reduce confidence threshold:** Lower `min_confidence` for faster matching

### Network Configuration

If behind a firewall or proxy:

```python
# Add to api_integration.py
session.proxies = {
    'http': 'http://proxy.company.com:8080',
    'https': 'https://proxy.company.com:8080'
}
```

## Development

### Running Tests

```bash
# Full validation suite
python validation.py

# Installation test only
python test_installation.py

# Test specific template
python validation.py templates/services/automatic1111.yaml
```

### Adding New Features

1. **New metadata fields:** Update universal schema first
2. **New services:** Create template file and test
3. **New operations:** Add to `MetaManUniversalNode.INPUT_TYPES`
4. **New nodes:** Follow pattern in `specialized_nodes.py`

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

---

For more information, see the main README.md or visit the project repository.
