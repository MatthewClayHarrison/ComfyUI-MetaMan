# MetaMan TODO List

## 🚀 Future Enhancements

### **Platform Support Expansion**
- [ ] **Add SeaArt.AI support** - Metadata extraction and format conversion
- [ ] **Add Midjourney support** - Parameter extraction from Midjourney images
- [ ] **Add Runway ML support** - Video generation metadata handling
- [ ] **Add Stability AI support** - StableVideo and other Stability products
- [ ] **Add Adobe Firefly support** - Adobe's AI generation metadata

### **Critical Bug Fixes**
- [ ] **VAE not found in Tensor.AI Flux workflows** - Flux workflows missing VAE extraction
- [ ] **Embeddings from Power Prompt not found** - Power Prompt (rgthree) embedding detection failing
- [ ] **ControlNet model extraction** - Missing ControlNet model detection in complex workflows
- [ ] **Refiner model detection** - SDXL refiner models not being identified

### **New Node Development**
- [ ] **MetaMan Model Finder** - New node to find missing models and suggest downloads
  - Search Civitai, HuggingFace, and other repositories
  - Hash-based model identification
  - Missing model dependency resolution
  - Download link generation
- [ ] **MetaMan Batch Processor** - Process multiple images at once
- [ ] **MetaMan Workflow Validator** - Validate extracted workflows for completeness

### **Feature Enhancements**
- [ ] **Enhanced Tensor.AI support** - Better EMS code detection and mapping
- [ ] **Advanced LoRA detection** - Support for LoRA tags in non-standard formats
- [ ] **Model hash verification** - Cross-reference model hashes for accuracy
- [ ] **Workflow reconstruction** - Generate functional ComfyUI workflows from any metadata
- [ ] **Custom node compatibility** - Better support for community custom nodes

### **User Experience Improvements**
- [ ] **Progress indicators** - Show progress for long-running operations
- [ ] **Error reporting** - Better error messages and debugging information
- [ ] **Documentation** - Video tutorials and comprehensive usage guides
- [ ] **Community database** - Crowdsourced EMS→real name mappings for Tensor.AI

### **Performance Optimizations**
- [ ] **Caching system** - Cache API results and model lookups
- [ ] **Batch operations** - Process multiple files efficiently
- [ ] **Memory optimization** - Reduce memory usage for large workflows
- [ ] **Async processing** - Non-blocking operations for better UX

### **Integration & Compatibility**
- [ ] **ComfyUI-Manager integration** - Easy installation via ComfyUI-Manager
- [ ] **API endpoints** - REST API for external tool integration
- [ ] **Plugin system** - Allow community plugins for new platforms
- [ ] **Workflow templates** - Pre-built templates for common conversions

---

## 🎯 Priority Levels

**🔥 High Priority (Next Release)**
- VAE not found in Tensor.AI Flux workflows
- Embeddings from Power Prompt not found
- MetaMan Model Finder node

**⚡ Medium Priority**
- Add SeaArt.AI support  
- Enhanced Tensor.AI support
- Workflow reconstruction

**💎 Nice to Have**
- Batch Processor node
- Community database
- Plugin system

---

## 📝 Notes

- **Backward Compatibility**: All enhancements should maintain compatibility with existing workflows
- **Testing**: Each new feature requires thorough testing with real-world images
- **Documentation**: New features must include clear documentation and examples
- **Community Feedback**: Prioritize features based on user requests and bug reports

---

**Last Updated**: August 2025  
**Version**: 1.0.0  
**Architecture**: Three-node core system (Load Image → Extract Components → Embed & Save)
