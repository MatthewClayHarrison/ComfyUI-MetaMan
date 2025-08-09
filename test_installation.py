#!/usr/bin/env python3
"""
MetaMan Installation Test Script
Verifies that MetaMan is properly installed and configured
"""

import os
import sys
import json
from pathlib import Path


def test_imports():
    """Test that all MetaMan modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test basic Python imports
        import yaml
        import json
        from PIL import Image, PngImagePlugin
        print("  ✅ Basic dependencies OK")
        
        # Test MetaMan imports (using absolute imports for testing)
        import sys
        import os
        
        # Add current directory to path for testing
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test importing MetaMan modules
        import metaman
        import specialized_nodes
        import metadata_parser
        import api_integration
        print("  ✅ MetaMan modules imported OK")
        
        # Test node class availability
        from metaman import MetaManUniversalNode
        from specialized_nodes import MetaManWorkflowSaver, MetaManDependencyResolver
        print("  ✅ MetaMan node classes OK")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def test_templates():
    """Test that template files are present and valid"""
    print("Testing templates...")
    
    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    
    if not templates_dir.exists():
        print("  ❌ Templates directory not found")
        return False
    
    # Check universal schema
    schema_file = templates_dir / "universal_schema.yaml"
    if not schema_file.exists():
        print("  ❌ Universal schema file not found")
        return False
    
    try:
        import yaml
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
        print("  ✅ Universal schema loaded OK")
    except Exception as e:
        print(f"  ❌ Schema loading error: {e}")
        return False
    
    # Check service templates
    services_dir = templates_dir / "services"
    if not services_dir.exists():
        print("  ❌ Services templates directory not found")
        return False
    
    service_files = list(services_dir.glob("*.yaml"))
    if len(service_files) == 0:
        print("  ❌ No service template files found")
        return False
    
    print(f"  ✅ Found {len(service_files)} service templates")
    
    # Test loading each template
    for template_file in service_files:
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
        except Exception as e:
            print(f"  ❌ Error loading {template_file.name}: {e}")
            return False
    
    print("  ✅ All service templates loaded OK")
    return True


def test_node_creation():
    """Test that nodes can be instantiated"""
    print("Testing node creation...")
    
    try:
        # Import with absolute imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from metaman import MetaManUniversalNode
        from specialized_nodes import MetaManWorkflowSaver, MetaManDependencyResolver
        
        # Test node instantiation
        universal_node = MetaManUniversalNode()
        workflow_node = MetaManWorkflowSaver()
        dependency_node = MetaManDependencyResolver()
        
        print("  ✅ All nodes created successfully")
        
        # Test INPUT_TYPES methods
        universal_inputs = MetaManUniversalNode.INPUT_TYPES()
        workflow_inputs = MetaManWorkflowSaver.INPUT_TYPES()
        dependency_inputs = MetaManDependencyResolver.INPUT_TYPES()
        
        print("  ✅ INPUT_TYPES methods working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Node creation error: {e}")
        return False


def test_metadata_parsing():
    """Test basic metadata parsing functionality"""
    print("Testing metadata parsing...")
    
    try:
        # Import with absolute imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from metadata_parser import metadata_parser
        
        # Test A1111 format parsing
        test_metadata = {
            'parameters': 'test prompt\nNegative prompt: test negative\nSteps: 25, Sampler: Euler, CFG scale: 7.5, Seed: 123, Size: 512x512'
        }
        
        detected_service, confidence = metadata_parser._detect_source_service(test_metadata)
        
        if detected_service and confidence > 0:
            print(f"  ✅ Service detection working (detected: {detected_service}, confidence: {confidence:.2f})")
        else:
            print("  ❌ Service detection failed")
            return False
        
        # Test parsing
        parsed = metadata_parser._parse_a1111_parameters(test_metadata)
        if 'prompt' in parsed and 'steps' in parsed:
            print("  ✅ A1111 parsing working")
        else:
            print("  ❌ A1111 parsing failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Metadata parsing error: {e}")
        return False


def test_api_integration():
    """Test API integration module"""
    print("Testing API integration...")
    
    try:
        # Import with absolute imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from api_integration import model_resolver, hash_verifier
        
        # Test resolver initialization
        if hasattr(model_resolver, 'find_model_sources'):
            print("  ✅ Model resolver initialized")
        else:
            print("  ❌ Model resolver missing methods")
            return False
        
        # Test hash verifier
        if hasattr(hash_verifier, 'calculate_file_hash'):
            print("  ✅ Hash verifier initialized")
        else:
            print("  ❌ Hash verifier missing methods")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ API integration error: {e}")
        return False


def run_validation():
    """Run comprehensive validation"""
    print("Running validation...")
    
    try:
        # Import with absolute imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from validation import run_validation as validation_func
        
        base_dir = Path(__file__).parent
        templates_dir = str(base_dir / "templates")
        
        report = validation_func(templates_dir)
        print("  ✅ Validation completed")
        print("\nValidation Report:")
        print("-" * 50)
        print(report)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        return False


def main():
    """Run all tests"""
    print("MetaMan Installation Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Templates", test_templates),
        ("Node Creation", test_node_creation),
        ("Metadata Parsing", test_metadata_parsing),
        ("API Integration", test_api_integration),
        ("Validation", run_validation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        print("-" * 20)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed! MetaMan is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
