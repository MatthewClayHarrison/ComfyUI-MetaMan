# MetaMan Examples

This directory contains example metadata files and usage scenarios to demonstrate MetaMan's capabilities.

## Example Metadata Formats

### A1111 Parameters Example
```
a beautiful landscape, mountains, sunset, highly detailed, photorealistic
Negative prompt: blurry, low quality, oversaturated
Steps: 25, Sampler: DPM++ 2M Karras, CFG scale: 7.5, Seed: 1234567890, Size: 1024x768, Model hash: a1b2c3d4e5f6, Model: dreamshaper_8, Clip skip: 2
```

### ComfyUI Workflow Example
```json
{
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "widgets_values": ["dreamshaper_8.safetensors"]
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "widgets_values": ["a beautiful landscape, mountains, sunset"]
    }
  ]
}
```

### Civitai Enhanced Example
```
a beautiful landscape, mountains, sunset, highly detailed, photorealistic
Negative prompt: blurry, low quality, oversaturated
Steps: 25, Sampler: DPM++ 2M Karras, CFG scale: 7.5, Seed: 1234567890, Size: 1024x768, Model hash: a1b2c3d4e5f6, Model: dreamshaper_8, Model ID: 4384, Version ID: 128713, Resources: [{"name": "landscape_lora", "type": "lora", "weight": 0.8, "hash": "abc123"}]
```

## Usage Examples

### Basic Metadata Extraction
1. Load an image with AI-generated metadata
2. Add MetaMan Universal node
3. Set operation to "extract_universal"
4. Choose target service (e.g., "automatic1111")
5. View extracted metadata in universal format and converted format

### Cross-Platform Conversion
1. Load ComfyUI-generated image
2. Set operation to "convert_to_service"
3. Set target_service to "automatic1111"
4. Get A1111-compatible parameter string

### Dependency Resolution
1. Load image with model references
2. Add MetaMan Dependency Resolver node
3. Set search_platforms to "all"
4. Get download URLs and model information

### Workflow Preservation
1. Load ComfyUI workflow image
2. Add MetaMan Workflow Saver node
3. Set save_format to "embed_in_image"
4. Get image with preserved workflow data

## Test Cases

See individual files in this directory for specific test cases and expected outputs.
