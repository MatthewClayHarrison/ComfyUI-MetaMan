# Changelog

All notable changes to MetaMan will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-08

### Added
- **Universal Metadata Management System**
  - Custom "meta" PNG chunk standard for cross-platform compatibility
  - Universal schema supporting 60+ metadata fields across all major platforms
  - Template-driven architecture for easy platform extension

- **Platform Support**
  - Automatic1111 (A1111) - Full parameter format support
  - ComfyUI - Complete workflow and prompt preservation
  - Civitai - Enhanced resource tracking with model IDs
  - Forge - A1111-compatible with advanced sampling extensions
  - Tensor.AI - JSON-based metadata format
  - Leonardo.AI - Style and preset management system

- **ComfyUI Nodes**
  - **MetaMan Universal** - Primary metadata extraction and conversion
  - **MetaMan Workflow Saver** - Dedicated workflow preservation with multiple output formats
  - **MetaMan Dependency Resolver** - Model dependency tracking with download URLs

- **Core Features**
  - Intelligent service detection with confidence scoring
  - Bidirectional metadata conversion between platforms
  - Model dependency resolution via Civitai and HuggingFace APIs
  - Comprehensive validation and testing system
  - PNG metadata embedding with compression support
  - Rate-limited API integration respecting platform limits

- **Template System**
  - YAML-based service templates for easy extension
  - Universal metadata schema with field compatibility tracking
  - Service-specific output formatters with validation rules
  - Automatic field mapping and type conversion

- **Documentation**
  - Comprehensive README with usage examples
  - Detailed setup and configuration guide (SETUP.md)
  - Contributing guidelines for community development
  - Installation test script and validation system

### Technical Details
- **Python 3.8+ support** with modern type hints
- **Pillow integration** for robust PNG metadata handling
- **SHA256 hash-based** model identification for accuracy
- **Extensible architecture** supporting future platforms
- **Memory-efficient** processing with lazy loading
- **Cross-platform compatibility** (Windows, macOS, Linux)

### Dependencies
- Pillow >= 9.0.0 (image processing)
- PyYAML >= 6.0.0 (template processing)
- piexif >= 1.1.3 (EXIF metadata handling)
- requests >= 2.28.0 (API integration)
- jsonschema >= 4.17.0 (validation)
- torch (provided by ComfyUI environment)

## [Unreleased]

### Planned Features
- SeaArt.AI template support
- Midjourney metadata extraction (where available)
- Batch processing optimization
- Advanced workflow analysis tools
- Model hash verification utilities
- Plugin system for custom extractors
- GUI configuration interface
- Automatic model downloading integration
- Workflow sharing and marketplace integration

---

## Version History Notes

- **Major versions** (X.0.0): Breaking changes, new architecture, major platform additions
- **Minor versions** (0.X.0): New platforms, significant features, template additions  
- **Patch versions** (0.0.X): Bug fixes, template updates, documentation improvements

For upgrade instructions between major versions, see SETUP.md.
