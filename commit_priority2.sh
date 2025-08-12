#!/bin/bash
cd /Users/pxl8d/Projects/MetaMan

echo "=== Committing Priority 2 Implementation ==="

# Add all changes
git add -A

# Commit with descriptive message
git commit -m "Implement MetaMan Extract Components node (Priority 2 - Metadata Provider)

Features implemented:
- MetaManExtractComponents: Extract individual workflow components from metadata JSON
- MetaManLoadAndConvert: Combined load + conversion functionality  
- MetaManEmbedAndSave: Clean metadata embedding and save functionality
- Updated node registrations in __init__.py

Priority 2 delivers the most valuable workflow: 'stealing' specific parameters 
from any image to use directly in ComfyUI workflows via individual typed outputs:
- positive_prompt, negative_prompt (STRING)
- steps, seed, width, height (INT) 
- cfg_scale, denoising_strength (FLOAT)
- sampler, scheduler, model_name, loras (STRING)

Ready for testing with ComfyUI_00070_.png test case."

# Push to dev branch
git push origin dev

echo "=== Priority 2 Implementation Committed and Pushed ==="
