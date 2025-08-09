# Contributing to MetaMan

Thank you for considering contributing to MetaMan! This project aims to create universal metadata interoperability across AI image generation platforms.

## 🤝 How to Contribute

### Types of Contributions Welcome

- **New Platform Support**: Add templates for new AI image generation services
- **Bug Fixes**: Fix parsing errors, conversion issues, or template problems  
- **Feature Enhancements**: Improve existing functionality or add new capabilities
- **Documentation**: Improve setup guides, examples, or API documentation
- **Testing**: Add test cases, validation scripts, or example workflows
- **Optimization**: Performance improvements or code quality enhancements

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MetaMan.git
   cd MetaMan
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests** to ensure everything works:
   ```bash
   python test_installation.py
   python validation.py
   ```

### Development Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following these guidelines:
   - Follow existing code style and conventions
   - Add docstrings to new functions and classes
   - Update templates and schema as needed
   - Add validation tests for new features

3. **Test your changes**:
   ```bash
   python test_installation.py
   python validation.py
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: your descriptive commit message"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## 🔧 Adding New Platform Support

To add support for a new AI image generation service:

### 1. Create Service Template

Copy an existing template and modify:
```bash
cp templates/services/automatic1111.yaml templates/services/newservice.yaml
```

Key sections to update:
- `service_name`: Unique identifier for the service
- `output_format`: How metadata should be formatted
- `field_mappings`: How universal fields map to service fields
- `png_chunk`: How metadata is stored in PNG files
- `validation`: Service-specific validation rules

### 2. Update Universal Schema (if needed)

If the new service has unique metadata fields:
1. Add fields to `templates/universal_schema.yaml`
2. Specify `supported_by` list including your new service
3. Define field types, validation rules, and mappings

### 3. Test Your Template

Run validation to ensure your template is correct:
```bash
python validation.py
```

### 4. Add Examples

Create example metadata files in the `examples/` directory showing:
- Typical metadata from your service
- Expected conversion output
- Any special handling requirements

## 📝 Code Style Guidelines

- **Python**: Follow PEP 8 with 4-space indentation
- **YAML**: Use 2-space indentation, consistent key ordering
- **Comments**: Document complex logic and platform-specific quirks
- **Error Handling**: Always include try/catch blocks for external API calls
- **Type Hints**: Use type hints for function parameters and return values

## 🧪 Testing

### Running Tests

```bash
# Full test suite
python test_installation.py

# Template validation
python validation.py

# Test specific service template
python validation.py templates/services/yourservice.yaml
```

### Adding Tests

When adding new features:
1. Add test cases to `validation.py`
2. Include both positive and negative test scenarios
3. Test edge cases (malformed metadata, missing fields, etc.)
4. Verify cross-platform conversion accuracy

## 📚 Documentation

### README Updates

When adding new features, update:
- Feature list in main README.md
- Compatibility matrix
- Usage examples
- Installation requirements

### Template Documentation

Each service template should include:
- Clear description of the service and its metadata format
- Examples of typical metadata structures
- Notes about any special handling or limitations
- Links to official service documentation

## 🐛 Bug Reports

When reporting bugs:
1. **Use GitHub Issues** with a descriptive title
2. **Include steps to reproduce** the issue
3. **Provide sample data** (metadata, images, workflows)
4. **Specify your environment** (Python version, OS, ComfyUI version)
5. **Include error messages** and stack traces

### Bug Report Template

```markdown
**Bug Description:**
Brief description of the issue

**Steps to Reproduce:**
1. Load image with metadata from [service]
2. Use MetaMan node with operation [operation]
3. Error occurs when...

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- ComfyUI: [version]
- MetaMan: [version/commit]

**Sample Data:**
[Attach sample image or provide metadata text]
```

## 💡 Feature Requests

For new features:
1. **Check existing issues** to avoid duplicates
2. **Describe the use case** - why is this needed?
3. **Propose implementation** if you have ideas
4. **Consider compatibility** with existing features

## 🔄 Release Process

1. **Version Numbering**: We follow semantic versioning (MAJOR.MINOR.PATCH)
2. **Changelog**: Update CHANGELOG.md with new features and fixes
3. **Testing**: All tests must pass before release
4. **Tagging**: Create Git tags for releases
5. **Documentation**: Update installation and setup guides

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **Issues**: For bug reports and feature requests  
- **Documentation**: Check README.md and SETUP.md first

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special thanks for new platform support

## 📄 License

By contributing to MetaMan, you agree that your contributions will be licensed under the same MIT License as the project.

---

Thank you for helping make AI image metadata universal and interoperable! 🎉
